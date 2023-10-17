import random

import GDPSO
import AdamPSO
import torch
from time import time
from torch_pso import ParticleSwarmOptimizer
from torch_geometric.datasets import KarateClub
from torch_geometric.datasets import GNNBenchmarkDataset
import torch.nn.functional as F
from torch.nn import Linear
from AdaSwarm import RotatedEMParicleSwarmOptimizer
from torch_geometric.nn import GCNConv
from torch_geometric.data import DataLoader
import json
from torch_geometric.datasets import Planetoid
#dataset = KarateClub()
#dataset = GNNBenchmarkDataset("storeData/","TSP")


dataset = Planetoid("storeData/", "Cora")
#dataset = Planetoid("storeData/", "Pubmed")
#dataset = Planetoid("storeData/", "Citeseer")





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
        # print(x, self.y)
        return self.fitness(x, self.y)



class GCN(torch.nn.Module):
    def __init__(self,seed):
        super(GCN, self).__init__()
        torch.manual_seed(seed)
        self.conv1 = GCNConv(dataset.num_features, 4)
        self.conv2 = GCNConv(4, 4)
        self.conv3 = GCNConv(4, 2)
        self.classifier = Linear(2, dataset.num_classes)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        h = self.conv3(h, edge_index)
        h = h.tanh()  # Final GNN embedding space.

        # Apply a final (linear) classifier.
        # Apply a final (linear) classifier.
        out = self.classifier(h)


        return out, h




# optimizer = torch.optim.Adam(self.model.parameters(), lr=0.02)  # Initialize the Adam optimizer.
# optimizer = torch.optim.SGD(self.model.parameters(),lr=0.02, momentum=0.9)
#optimizer = ParticleSwarmOptimizer(self.model.parameters(), inertial_weight=0.3, num_particles=100, max_param_value=1,min_param_value=-1)
class training():
    
    def __init__(self,seed):
        random.seed(seed)
        data = dataset[0]
        self.model = GCN(seed)
        self.criterion = torch.nn.CrossEntropyLoss()  # Initialize the CrossEntropyLoss function.
        self.earlyStoppingIterations = 15
        num_samples = data.num_nodes
        train_size = int(0.6 * num_samples)
        val_size = int(0.2 * num_samples)
        test_size = num_samples - train_size - val_size
        self.train_indices = range(train_size)
        self.val_indices = range(train_size, train_size + val_size)
        self.test_indices = range(train_size + val_size, num_samples)
        x = DataLoader([data], batch_size=1, shuffle=True)
        for d in x:
            self.datas = d
    
    
    def earlyStopping(self,listOfEvaluation):
        preivousLoss = 100
        count = 0
        for loss in listOfEvaluation:
            if loss > preivousLoss:
                count += 1
            preivousLoss = loss
            if count == self.earlyStoppingIterations:
                return True
        return False
    
    
    
    def accuracy(self,pred_y, y):
        return (pred_y == y).sum() / len(y)
    
    def val(self,data):
        out, h = self.model(data.x, data.edge_index)
        loss = self.criterion(out[self.val_indices], data.y[self.val_indices])
        acc = self.accuracy(out[self.val_indices].argmax(dim=1), data.y[self.val_indices])
        return loss, h, acc
    
    def test(self,data):
        out, h = self.model(data.x, data.edge_index)
        loss = self.criterion(out[self.test_indices], data.y[self.test_indices])
        acc = self.accuracy(out[self.test_indices].argmax(dim=1), data.y[self.test_indices])
        return loss, h, acc
    
    def trainAdam(self,optimizer,d):
        optimizer.zero_grad()  # Clear gradients.
    
        out, h = self.model(d.x, d.edge_index)  # Perform a single forward pass.
        loss = self.criterion(out[self.train_indices], d.y[self.train_indices])
        loss.backward()  # Derive gradients.
    
        acc = self.accuracy(out.argmax(dim=1), d.y)
        optimizer.step()  # Update parameters based on gradients.
        # print(loss)
        return loss, h, acc
    
    
    def trainSGD(self,optimizer,data):
        optimizer.zero_grad()  # Clear gradients.
    
        out, h = self.model(data.x, data.edge_index)  # Perform a single forward pass.
        loss = self.criterion(out[self.train_indices], data.y[self.train_indices])
        loss.backward()  # Derive gradients.
    
        acc = self.accuracy(out.argmax(dim=1), data.y)
        optimizer.step()  # Update parameters based on gradients.
        # print(loss)
        return loss, h, acc
    
    def trainPSO(self,optimizer,data):
        b = ""
        self.model.train()
    
        def closure():
            nonlocal b
            optimizer.zero_grad()  # Clear gradients.
            # print("x :" + str(data.x))
            # print("Y : " + str(data.edge_index))
            out, h = self.model.forward(data.x, data.edge_index)  # Perform a single forward pass.
            b = out
    
            loss = self.criterion(out[self.train_indices], data.y[self.train_indices])
            return loss
    
        loss = optimizer.step(closure)  # Update parameters based on gradients.
        acc = self.accuracy(b.argmax(dim=1), data.y)
        # acc = self.accuracy(b.argmax(dim=1), data.y)
        # print(acc)
    
        return loss, 0,acc
    
    def trainAdaSwarm(self,optimizer,data,proposed_loss):
        x = data.x
        y = data.y
        y.requires_grad = False
        dimensions = x.size()[0]
        getData = self.read_json_file()
        adaswarmParameters = getData["AdaSwarm"]
        swarm_size = adaswarmParameters["swarm_size"]
        numberOfIterations = adaswarmParameters["iterations"]
        cognitive_coefficient = adaswarmParameters["cognitive_parameter"]
        social_coefficient = adaswarmParameters["social_parameter"]
        momentum = adaswarmParameters["momentum_factor"]
        options = [cognitive_coefficient, social_coefficient, momentum, 100]
        p = RotatedEMParicleSwarmOptimizer(dimensions = dimensions, swarm_size=swarm_size, classes=dataset.num_classes,  true_y=y,options=options,  iterations=numberOfIterations)
        p.optimize(CELoss(y))
        out, h = self.model(x, data.edge_index)
        # for i in range(10):
        #     c1r1, c2r2, gbest = p.run_one_iter(verbosity=False)
    
        c1r1, c2r2, gbest = p.run(verbosity=False)
    
        loss = proposed_loss(out, y, c1r1 + c2r2, 0.1, gbest)
    
        acc = self.accuracy(out.argmax(dim=1), data.y)
        optimizer.zero_grad()  # Clear gradients.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        check = out.argmax(dim=1)
        return loss, 0, acc
    
    def trainGDPSO(self,optimizer,data):
        b = ""
        def closure():
            nonlocal b
            if torch.is_grad_enabled():
                optimizer.zero_grad()  # Clear gradients.
            out, h = self.model(data.x, data.edge_index)  # Perform a single forward pass.
            b = out
            loss = self.criterion(out[self.train_indices], data.y[self.train_indices])
            return loss
    
        loss = optimizer.step(closure)  # Update parameters based on gradients.
        #I need to pass as a parmeter the self.model, data
        acc = self.accuracy(b.argmax(dim=1), data.y)
        # acc = self.accuracy(b.argmax(dim=1), data.y)
        # print(acc)
        return loss,0, acc
    
    def trainAdamPSO(self,optimizer,data):
        b = ""
        def closure():
            nonlocal b
            if torch.is_grad_enabled():
                optimizer.zero_grad()  # Clear gradients.
            out, h = self.model(data.x, data.edge_index)  # Perform a single forward pass.
            b = out
            loss = self.criterion(out[self.train_indices], data.y[self.train_indices])
            return loss
    
        loss = optimizer.step(closure)  # Update parameters based on gradients.
        #I need to pass as a parmeter the self.model, data
        acc = self.accuracy(b.argmax(dim=1), data.y)
        # acc = self.accuracy(b.argmax(dim=1), data.y)
        # print(acc)
        return loss,0, acc
    
    
    
    
    
    
    
    
    
    
    def runAdam(self,seed):
        
        start = time()
        allevaluation = []
        s = "Seed : " + str(seed) + "\n===========================================================\n"
        self.addToFile("Adam",s)
        listOfSeeds = random.sample(range(0, 100000), 10)
        for index,otherSeed in enumerate(listOfSeeds):
            self.model = GCN(seed)
            v = index+1
            s = "Run : " + str(v) + "\n---------\n"
            self.addToFile("Adam", s)
            random.seed(otherSeed)
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.02)
            for epoch in range(200):
                self.model.train()
                loss,h, acc = self.trainAdam(optimizer,self.datas)
                self.model.eval()  # Switch to evaluation mode for testing.
                with torch.no_grad():
                    test_loss, _, test_acc = self.val(self.datas)
                allevaluation.append(test_loss)
                self.addToFile("Adam", f'Epcoh:{epoch} TrainingLoss:{loss} Trainingself.accuracy:{acc} EvaluationLoss:{test_loss} Evaluationself.accuracy:{test_acc}\n')
                if self.earlyStopping(allevaluation) == True:
                    break
            final_loss, _, final_acc = self.test(self.datas)
            trainTime = time() - start
            self.addToFile("Adam",f'TrainingTime:{trainTime} TestLoss:{final_loss} Testself.accuracy:{final_acc}\n')
    
    def runSGD(self,seed):
        
        start = time()
        allevaluation = []
        s = "Seed : " + str(seed) + "\n===========================================================\n"
        self.addToFile("SGD",s)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.02)
        for epoch in range(200):
            self.model.train()
            loss,h, acc = self.trainSGD(optimizer,self.datas)
            self.model.eval()  # Switch to evaluation mode for testing.
            with torch.no_grad():
                test_loss, _, test_acc =self.val(self.datas)
            allevaluation.append(test_loss)
            self.addToFile("SGD", f'Epcoh:{epoch} TrainingLoss:{loss} Trainingself.accuracy:{acc} EvaluationLoss:{test_loss} Evaluationself.accuracy:{test_acc}\n')
            if self.earlyStopping(allevaluation) == True:
                break
        final_loss, _, final_acc = self.test(self.datas)
        trainTime = time() - start
        self.addToFile("SGD",f'TrainingTime:{trainTime} TestLoss:{final_loss} Testself.accuracy:{final_acc}\n')
    
    def runPSO(self,seed):
        
        getData = self.read_json_file()
        psoParameters = getData["PSO"]
        start = time()
        allevaluation = []
        s = "Seed : " + str(seed) + "\n===========================================================\n"
        self.addToFile("PSO",s)
        optimizer = ParticleSwarmOptimizer(self.model.parameters(), inertial_weight=psoParameters["inertia_weight"], num_particles=psoParameters["swarm_size"],
                                               cognitive_coefficient=psoParameters["cognitive_parameter"], social_coefficient=psoParameters["social_parameter"], max_param_value=1, min_param_value=-1)
        for epoch in range(200):
            self.model.train()
            loss,h, acc = self.trainPSO(optimizer,self.datas)
            self.model.eval()  # Switch to evaluation mode for testing.
            with torch.no_grad():
                test_loss, _, test_acc =self.val(self.datas)
            allevaluation.append(test_loss)
            self.addToFile("PSO", f'Epcoh:{epoch} TrainingLoss:{loss} Trainingself.accuracy:{acc} EvaluationLoss:{test_loss} Evaluationself.accuracy:{test_acc}\n')
            if self.earlyStopping(allevaluation) == True:
                break
        final_loss, _, final_acc = self.test(self.datas)
        trainTime = time() - start
        self.addToFile("PSO",f'TrainingTime:{trainTime} TestLoss:{final_loss} Testself.accuracy:{final_acc}\n')
    
    def runAdaSwarm(self,seed):
        
        start = time()
        allevaluation = []
        s = "Seed : " + str(seed) + "\n===========================================================\n"
        self.addToFile("AdaSwarm",s)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.02)  # Initialize the Adam optimizer.
        proposed_loss = CELossWithPSO.apply
        for epoch in range(200):
            self.model.train()
            loss,h, acc = self.trainAdaSwarm(optimizer,self.datas,proposed_loss)
            self.model.eval()  # Switch to evaluation mode for testing.
            with torch.no_grad():
                test_loss, _, test_acc =self.val(self.datas)
            allevaluation.append(test_loss)
            self.addToFile("AdaSwarm", f'Epcoh:{epoch} TrainingLoss:{loss} Trainingself.accuracy:{acc} EvaluationLoss:{test_loss} Evaluationself.accuracy:{test_acc}\n')
            if self.earlyStopping(allevaluation) == True:
                break
        final_loss, _, final_acc = self.test(self.datas)
        trainTime = time() - start
        self.addToFile("AdaSwarm",f'TrainingTime:{trainTime} TestLoss:{final_loss} Testself.accuracy:{final_acc}\n')
    
    def runGDPSO(self,seed):
        
        getData = self.read_json_file()
        gdPsoParameters = getData["GDPSO"]
        start = time()
        allevaluation = []
        s = "Seed : " + str(seed) + "\n===========================================================\n"
        self.addToFile("GDPSO",s)
        optimizer = GDPSO.GDPSO(self.model,dataset, self.model.parameters(),
                                     inertial_weight=gdPsoParameters["inertia_weight"],
                                     num_particles=gdPsoParameters["swarm_size"],
                                     cognitive_coefficient=gdPsoParameters["cognitive_parameter"],
                                     social_coefficient=gdPsoParameters["social_parameter"], max_param_value=1,
                                     min_param_value=-1)
        for epoch in range(200):
            self.model.train()
            loss,h, acc = self.trainGDPSO(optimizer,self.datas)
            self.model.eval()  # Switch to evaluation mode for testing.
            with torch.no_grad():
                test_loss, _, test_acc =self.val(self.datas)
            allevaluation.append(test_loss)
            self.addToFile("GDPSO", f'Epcoh:{epoch} TrainingLoss:{loss} Trainingself.accuracy:{acc} EvaluationLoss:{test_loss} Evaluationself.accuracy:{test_acc}\n')
            if self.earlyStopping(allevaluation) == True:
                break
        final_loss, _, final_acc = self.test(self.datas)
        trainTime = time() - start
        self.addToFile("GDPSO",f'TrainingTime:{trainTime} TestLoss:{final_loss} Testself.accuracy:{final_acc}\n')
    
    
    def runAdamPSO(self,seed):

        getData = self.read_json_file()
        gdPsoParameters = getData["AdamPSO"]
        start = time()
        allevaluation = []
        s = "Seed : " + str(seed) + "\n===========================================================\n"
        self.addToFile("AdamPSO",s)
        optimizer = AdamPSO.AdamPSO(self.model,dataset, self.model.parameters(),
                                     inertial_weight=gdPsoParameters["inertia_weight"],
                                     num_particles=gdPsoParameters["swarm_size"],
                                     cognitive_coefficient=gdPsoParameters["cognitive_parameter"],
                                     social_coefficient=gdPsoParameters["social_parameter"], max_param_value=1,
                                     min_param_value=-1)
        for epoch in range(200):
            self.model.train()
            loss,h, acc = self.trainAdamPSO(optimizer,self.datas)
            self.model.eval()  # Switch to evaluation mode for testing.
            with torch.no_grad():
                test_loss, _, test_acc =self.val(self.datas)
            allevaluation.append(test_loss)
            self.addToFile("AdamPSO", f'Epcoh:{epoch} TrainingLoss:{loss} Trainingself.accuracy:{acc} EvaluationLoss:{test_loss} Evaluationself.accuracy:{test_acc}\n')
            if self.earlyStopping(allevaluation) == True:
                break
        final_loss, _, final_acc = self.test(self.datas)
        trainTime = time() - start
        self.addToFile("AdamPSO",f'TrainingTime:{trainTime} TestLoss:{final_loss} Testself.accuracy:{final_acc}\n')
    
    
    
    
    def addToFile(self,optimizerFile, add):
        file = optimizerFile + ".txt"
        f = open(file, "a")
        f.write(str(add))
        f.close()
    
    def read_json_file(self):
        with open('parameters.json', 'r') as file:
            data = json.load(file)
        return data