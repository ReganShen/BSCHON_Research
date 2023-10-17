from __future__ import division
from __future__ import print_function

import argparse
import glob
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch_pso import ParticleSwarmOptimizer
import GDPSO
import AdamPSO
from AdaSwarm import RotatedEMParicleSwarmOptimizer
from models import GAT, SpGAT
from utils import load_data, accuracy, read_json_file,load_data_Citeseer,load_data_Pubmed


class CELossWithPSO(torch.autograd.Function):
    @staticmethod
    def forward(ctx , y, y_pred, sum_cr, eta, gbest):
        ctx.save_for_backward(y, y_pred)
        ctx.sum_cr = sum_cr
        ctx.eta = eta
        ctx.gbest = gbest
        return F.cross_entropy(y,y_pred)

    @staticmethod
    def backward(ctx, grad_output):
        yy, yy_pred= ctx.saved_tensors
        sum_cr = ctx.sum_cr
        eta = ctx.eta
        grad_input = torch.neg((sum_cr/eta) * (ctx.gbest - yy))
        return grad_input, grad_output, None, None, None

class CELoss:
    def __init__(self, y):
        self.y = y
        self.fitness = torch.nn.CrossEntropyLoss()
    def evaluate(self, x):
        return self.fitness(x, self.y)

class GATTrainer:
    def __init__(self,o,s):
    # Training settings
        parser = argparse.ArgumentParser()
        parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
        parser.add_argument('--fastmode', action='store_true', default=True, help='Validate during training pass.')
        parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
        parser.add_argument('--seed', type=int, default=s, help='Random seed.')
        parser.add_argument('--epochs', type=int, default=400, help='Number of epochs to train.')
        parser.add_argument('--lr', type=float, default=0.02, help='Initial learning rate.')
        parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
        parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
        parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
        parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
        parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
        parser.add_argument('--patience', type=int, default=50, help='Patience')

        self.args = parser.parse_args()

        self.args.cuda = not self.args.no_cuda and torch.cuda.is_available()
        self.o = o
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        if self.args.cuda:
            torch.cuda.manual_seed(self.args.seed)
        
        # Load data
        #self.adj, self.features, self.labels, self.idx_train, self.idx_val, self.idx_test = load_data_Citeseer()  #The citeseer dataset
        #self.adj, self.features, self.labels, self.idx_train, self.idx_val, self.idx_test = load_data_Pubmed() #The Pubmed dataset
        self.adj, self.features, self.labels, self.idx_train, self.idx_val, self.idx_test = load_data()  #The cora dataste

        # Model and optimizer
        if self.args.sparse:
            self.model = SpGAT(nfeat=self.features.shape[1], 
                        nhid=self.args.hidden, 
                        nclass=int(self.labels.max()) + 1, 
                        dropout=self.args.dropout, 
                        nheads=self.args.nb_heads, 
                        alpha=self.args.alpha)
        else:
            self.model = GAT(nfeat=self.features.shape[1], 
                        nhid=self.args.hidden, 
                        nclass=int(self.labels.max()) + 1, 
                        dropout=self.args.dropout, 
                        nheads=self.args.nb_heads, 
                        alpha=self.args.alpha)

        # self.optimizer = optim.Adam(self.model.parameters(),
        #                        lr=self.args.lr,
        #                        weight_decay=self.args.weight_decay)

        self.criterion = torch.nn.CrossEntropyLoss()
        if o == "Adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr,weight_decay=self.args.weight_decay)
        elif o == "SGD":
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr)
        elif o == "PSO":
            getData = read_json_file()
            psoParameters = getData["PSO"]
            self.optimizer = ParticleSwarmOptimizer(self.model.parameters(), inertial_weight=psoParameters["inertia_weight"], num_particles=psoParameters["swarm_size"],
                                           cognitive_coefficient=psoParameters["cognitive_parameter"], social_coefficient=psoParameters["social_parameter"], max_param_value=1, min_param_value=-1)
        elif o == "AdaSwarm":
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr,weight_decay=self.args.weight_decay)
            self.criterion = CELossWithPSO.apply
        elif o == "GDPSO":
            getData = read_json_file()
            gdPsoParameters = getData["GDPSO"]
            iterations = gdPsoParameters["iterations_SGD"]
            listOfModelParameters = [self.features.shape[1], self.args.hidden, int(self.labels.max()) + 1, self.args.dropout, self.features, self.adj, self.idx_train, self.labels, self.args.nb_heads,self.args.alpha]
            self.optimizer = GDPSO.GDPSO(listOfModelParameters,self.model.parameters(), inertial_weight=gdPsoParameters["inertia_weight"], num_particles=gdPsoParameters["swarm_size"], cognitive_coefficient =gdPsoParameters["cognitive_parameter"], social_coefficient=gdPsoParameters["social_parameter"], max_param_value=1,min_param_value=-1)
        elif o == "AdamPSO":
            getData = read_json_file()
            adamPSOParameters = getData["AdamPSO"]
            iterations = adamPSOParameters["iterations_SGD"]
            listOfModelParameters = [self.features.shape[1], self.args.hidden, int(self.labels.max()) + 1, self.args.dropout, self.features, self.adj, self.idx_train, self.labels, self.args.nb_heads,self.args.alpha]
            self.optimizer = AdamPSO.AdamPSO(listOfModelParameters,self.model.parameters(), inertial_weight=adamPSOParameters["inertia_weight"], num_particles=adamPSOParameters["swarm_size"], cognitive_coefficient =adamPSOParameters["cognitive_parameter"], social_coefficient=adamPSOParameters["social_parameter"], max_param_value=1,min_param_value=-1)
        if self.args.cuda:
            self.model.cuda()
            self.features = self.features.cuda()
            self.adj = self.adj.cuda()
            self.labels = self.labels.cuda()
            self.idx_train = self.idx_train.cuda()
            self.idx_val = self.idx_val.cuda()
            self.idx_test = self.idx_test.cuda()
        
        self.features, self.adj, self.labels = Variable(self.features), Variable(self.adj), Variable(self.labels)
    
    
    def trainAdam(self,epoch):

        t = time.time()
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(self.features, self.adj)

        # loss_train = F.nll_loss(output[self.idx_train], self.labels[self.idx_train])
        loss_train = self.criterion(output[self.idx_train], self.labels[self.idx_train])
        acc_train = accuracy(output[self.idx_train], self.labels[self.idx_train])
        loss_train.backward()
        self.optimizer.step()
    
        if not self.args.fastmode:
            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
            self.model.eval()
            output = self.model(self.features, self.adj)
    
        # loss_val = F.nll_loss(output[self.idx_val], self.labels[self.idx_val])
        loss_val = self.criterion(output[self.idx_val], self.labels[self.idx_val])
        acc_val = accuracy(output[self.idx_val], self.labels[self.idx_val])
        str = 'Epoch: {:04d}'.format(epoch + 1) + ' loss_train: {:.4f}'.format(
            loss_train.data.item()) + ' acc_train: {:.4f}'.format(acc_train.data.item()) + ' loss_val: {:.4f}'.format(
            loss_val.data.item()) + ' acc_val: {:.4f}'.format(acc_val.data.item()) + ' time: {:.4f}s'.format(
            time.time() - t)
        self.addToFile(str)


        return loss_val.data.item()

    def trainSGD(self, epoch):
        t = time.time()
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(self.features, self.adj)
        # loss_train = F.nll_loss(output[self.idx_train], self.labels[self.idx_train])
        loss_train = self.criterion(output[self.idx_train], self.labels[self.idx_train])
        acc_train = accuracy(output[self.idx_train], self.labels[self.idx_train])
        loss_train.backward()
        self.optimizer.step()

        if not self.args.fastmode:
            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
            self.model.eval()
            output = self.model(self.features, self.adj)

        # loss_val = F.nll_loss(output[self.idx_val], self.labels[self.idx_val])
        loss_val = self.criterion(output[self.idx_val], self.labels[self.idx_val])
        acc_val = accuracy(output[self.idx_val], self.labels[self.idx_val])
        str = 'Epoch: {:04d}'.format(epoch + 1) + ' loss_train: {:.4f}'.format(
            loss_train.data.item()) + ' acc_train: {:.4f}'.format(acc_train.data.item()) + ' loss_val: {:.4f}'.format(
            loss_val.data.item()) + ' acc_val: {:.4f}'.format(acc_val.data.item()) + ' time: {:.4f}s'.format(
            time.time() - t)
        self.addToFile(str)
        # print('Epoch: {:04d}'.format(epoch + 1),
        #       'loss_train: {:.4f}'.format(loss_train.data.item()),
        #       'acc_train: {:.4f}'.format(acc_train.data.item()),
        #       'loss_val: {:.4f}'.format(loss_val.data.item()),
        #       'acc_val: {:.4f}'.format(acc_val.data.item()),
        #       'time: {:.4f}s'.format(time.time() - t))

        return loss_val.data.item()

    def trainPSO(self, epoch):
        t = time.time()
        self.model.train()
        b = ""
        c = ""
        def closure():
            nonlocal b,c
            self.optimizer.zero_grad()
            output = self.model(self.features, self.adj)
            # loss_train = F.nll_loss(output[self.idx_train], self.labels[self.idx_train])
            loss_train = self.criterion(output[self.idx_train], self.labels[self.idx_train])
            b = loss_train
            c = output
            return loss_train
        # loss_train.backward()
        self.optimizer.step(closure)
        loss_train = b
        output = c
        acc_train = accuracy(output[self.idx_train], self.labels[self.idx_train])

        if not self.args.fastmode:
            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
            self.model.eval()
            output = self.model(self.features, self.adj)

        # loss_val = F.nll_loss(output[self.idx_val], self.labels[self.idx_val])
        loss_val = self.criterion(output[self.idx_val], self.labels[self.idx_val])
        acc_val = accuracy(output[self.idx_val], self.labels[self.idx_val])
        # print('Epoch: {:04d}'.format(epoch + 1),
        #       'loss_train: {:.4f}'.format(loss_train.data.item()),
        #       'acc_train: {:.4f}'.format(acc_train.data.item()),
        #       'loss_val: {:.4f}'.format(loss_val.data.item()),
        #       'acc_val: {:.4f}'.format(acc_val.data.item()),
        #       'time: {:.4f}s'.format(time.time() - t))
        str = 'Epoch: {:04d}'.format(epoch + 1) + ' loss_train: {:.4f}'.format(
            loss_train.data.item()) + ' acc_train: {:.4f}'.format(acc_train.data.item()) + ' loss_val: {:.4f}'.format(
            loss_val.data.item()) + ' acc_val: {:.4f}'.format(acc_val.data.item()) + ' time: {:.4f}s'.format(
            time.time() - t)
        self.addToFile(str)
        return loss_val.data.item()



    def trainAdaSwarm(self, epoch, param):
        t = time.time()
        self.model.train()
        y =  self.labels[self.idx_train]
        y.requires_grad = False

        p = RotatedEMParicleSwarmOptimizer(dimensions=param['dimensions'], swarm_size=param['swarm_size'], classes=param['classes'],
                                           true_y=y, options=param['options'], iterations=param['iterations'])
        p.optimize(CELoss(y))

        output = self.model(self.features, self.adj)

        c1r1, c2r2, gbest = p.run(verbosity=False)

        loss_train = self.criterion(output[self.idx_train],
                              y, c1r1 + c2r2, 0.1, gbest)

        self.optimizer.zero_grad()
        loss_train.backward()
        self.optimizer.step()

        acc_train = accuracy(output[self.idx_train], self.labels[self.idx_train])



        if not self.args.fastmode:
            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
            self.model.eval()
            output = self.model(self.features, self.adj)

        loss_val = F.nll_loss(output[self.idx_val], self.labels[self.idx_val])
        acc_val = accuracy(output[self.idx_val], self.labels[self.idx_val])
        # print('Epoch: {:04d}'.format(epoch + 1),
        #       'loss_train: {:.4f}'.format(loss_train.data.item()),
        #       'acc_train: {:.4f}'.format(acc_train.data.item()),
        #       'loss_val: {:.4f}'.format(loss_val.data.item()),
        #       'acc_val: {:.4f}'.format(acc_val.data.item()),
        #       'time: {:.4f}s'.format(time.time() - t))
        str = 'Epoch: {:04d}'.format(epoch + 1) + ' loss_train: {:.4f}'.format(
            loss_train.data.item()) + ' acc_train: {:.4f}'.format(acc_train.data.item()) + ' loss_val: {:.4f}'.format(
            loss_val.data.item()) + ' acc_val: {:.4f}'.format(acc_val.data.item()) + ' time: {:.4f}s'.format(
            time.time() - t)
        self.addToFile(str)
        return loss_val.data.item()
    def trainGDPSO(self, epoch):
        t = time.time()
        self.model.train()
        b = ""
        c = ""

        def closure():
            nonlocal b, c
            self.optimizer.zero_grad()
            output = self.model(self.features, self.adj)
            loss_train = self.criterion(output[self.idx_train], self.labels[self.idx_train])
            b = loss_train
            c = output
            return loss_train

        # loss_train.backward()
        self.optimizer.step(closure)
        loss_train = b
        output = c
        acc_train = accuracy(output[self.idx_train], self.labels[self.idx_train])

        if not self.args.fastmode:
            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
            self.model.eval()
            output = self.model(self.features, self.adj)

        # loss_val = F.nll_loss(output[self.idx_val], self.labels[self.idx_val])
        loss_val = self.criterion(output[self.idx_val], self.labels[self.idx_val])
        acc_val = accuracy(output[self.idx_val], self.labels[self.idx_val])
        # print('Epoch: {:04d}'.format(epoch + 1),
        #       'loss_train: {:.4f}'.format(loss_train.data.item()),
        #       'acc_train: {:.4f}'.format(acc_train.data.item()),
        #       'loss_val: {:.4f}'.format(loss_val.data.item()),
        #       'acc_val: {:.4f}'.format(acc_val.data.item()),
        #       'time: {:.4f}s'.format(time.time() - t))
        str = 'Epoch: {:04d}'.format(epoch + 1) + ' loss_train: {:.4f}'.format(
            loss_train.data.item()) + ' acc_train: {:.4f}'.format(acc_train.data.item()) + ' loss_val: {:.4f}'.format(
            loss_val.data.item()) + ' acc_val: {:.4f}'.format(acc_val.data.item()) + ' time: {:.4f}s'.format(
            time.time() - t)
        self.addToFile(str)
        return loss_val.data.item()

    def trainAdamPSO(self, epoch):
        t = time.time()
        self.model.train()
        b = ""
        c = ""

        def closure():
            nonlocal b, c
            self.optimizer.zero_grad()
            output = self.model(self.features[self.idx_train], self.adj[self.idx_train])
            loss_train = self.criterion(output[self.idx_train], self.labels[self.idx_train])
            b = loss_train
            c = output
            return loss_train

        # loss_train.backward()
        self.optimizer.step(closure)
        loss_train = b
        output = c
        acc_train = accuracy(output[self.idx_train], self.labels[self.idx_train])

        if not self.args.fastmode:
            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
            self.model.eval()
            output = self.model(self.features, self.adj)

        # loss_val = F.nll_loss(output[self.idx_val], self.labels[self.idx_val])
        loss_val = self.criterion(output[self.idx_val], self.labels[self.idx_val])
        acc_val = accuracy(output[self.idx_val], self.labels[self.idx_val])
        # print('Epoch: {:04d}'.format(epoch + 1),
        #       'loss_train: {:.4f}'.format(loss_train.data.item()),
        #       'acc_train: {:.4f}'.format(acc_train.data.item()),
        #       'loss_val: {:.4f}'.format(loss_val.data.item()),
        #       'acc_val: {:.4f}'.format(acc_val.data.item()),
        #       'time: {:.4f}s'.format(time.time() - t))
        str = 'Epoch: {:04d}'.format(epoch + 1) + ' loss_train: {:.4f}'.format(
            loss_train.data.item()) + ' acc_train: {:.4f}'.format(acc_train.data.item()) + ' loss_val: {:.4f}'.format(
            loss_val.data.item()) + ' acc_val: {:.4f}'.format(acc_val.data.item()) + ' time: {:.4f}s'.format(
            time.time() - t)
        self.addToFile(str)
        return loss_val.data.item()


    def compute_test(self):
        self.model.eval()
        output = self.model(self.features, self.adj)
        loss_test = F.nll_loss(output[self.idx_test], self.labels[self.idx_test])
        acc_test = accuracy(output[self.idx_test], self.labels[self.idx_test])
        str = "Test set results:" + "loss= {:.4f}".format(loss_test.data.item()) + "accuracy= {:.4f}".format(acc_test.data.item())
        self.addToFile(str)
        # print("Test set results:",
        #       "loss= {:.4f}".format(loss_test.data.item()),
        #       "accuracy= {:.4f}".format(acc_test.data.item()))
    
    # Train model
    def startEverythiung(self,o):
        t_total = time.time()
        loss_values = []
        bad_counter = 0
        strSeed = "=========================================\n"
        strSeed += "Seed : " + str(self.args.seed)
        self.addToFile(strSeed)
        best = self.args.epochs + 1
        best_epoch = 0
        functionsFields = {"Adam" : self.trainAdam, "SGD" : self.trainSGD, "PSO" : self.trainPSO, "AdaSwarm" : self.trainAdaSwarm, "GDPSO" : self.trainGDPSO,"AdamPSO" : self.trainAdamPSO}
        if o == "AdaSwarm":
            getData = read_json_file()
            adaswarmParameters = getData["AdaSwarm"]
            swarm_size = adaswarmParameters["swarm_size"]
            numberOfIterations = adaswarmParameters["iterations"]
            cognitive_coefficient = adaswarmParameters["cognitive_parameter"]
            social_coefficient = adaswarmParameters["social_parameter"]
            momentum = adaswarmParameters["momentum_factor"]
            options = [cognitive_coefficient, social_coefficient, momentum, 100]
            toSend = {"dimensions" : len(self.idx_train), "swarm_size": swarm_size, 'classes':int(self.labels.max()) + 1, 'options' : options, 'iterations' : numberOfIterations}
        for epoch in range(self.args.epochs):
            if o == "AdaSwarm":
                loss_values.append(self.trainAdaSwarm(epoch,toSend))
            else:
                loss_values.append(functionsFields[o](epoch))
        
            torch.save(self.model.state_dict(), '{}.pkl'.format(epoch))
            if loss_values[-1] < best:
                best = loss_values[-1]
                best_epoch = epoch
                bad_counter = 0
            else:
                bad_counter += 1
        
            if bad_counter == self.args.patience:
                break
        
            files = glob.glob('*.pkl')
            for file in files:
                epoch_nb = int(file.split('.')[0])
                if epoch_nb < best_epoch:
                    os.remove(file)
        
        files = glob.glob('*.pkl')
        for file in files:
            epoch_nb = int(file.split('.')[0])
            if epoch_nb > best_epoch:
                os.remove(file)

        strs = " "
        strs += "Optimization Finished! \n" + "Total time elapsed: {:.4f}s".format(time.time() - t_total)
        self.addToFile(strs)
        # print("Optimization Finished!")
        # print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
        #
        # # Restore best model
        # print('Loading {}th epoch'.format(best_epoch))
        # self.model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))
        
        # Testing
        self.compute_test()

    def addToFile(self, add):
        file = self.o + ".txt"
        f = open(file, "a")
        f.write(str(add + "\n"))
        f.close()
