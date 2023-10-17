import gc
import warnings
from time import time,strftime,gmtime

import networkx as nx
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch as th
from sklearn.model_selection import train_test_split
from torch_pso import ParticleSwarmOptimizer
from layer import GCN
import adaswarm.nn
from AdaSwarm import RotatedEMParicleSwarmOptimizer
import GDPSO
import AdamPSO
import sys
sys.path.insert(0,'utils.py')
from utils import read_json_file
from utils import accuracy
from utils import macro_f1
from utils import CudaUse
from utils import EarlyStopping
from utils import LogResult
from utils import parameter_parser
from utils import preprocess_adj
from utils import print_graph_detail
from utils import read_file
from utils import return_seed
from utils import read_json_file
th.backends.cudnn.deterministic = True
th.backends.cudnn.benchmark = True
warnings.filterwarnings("ignore")


def get_train_test(target_fn):
    train_lst = list()
    test_lst = list()
    with read_file(target_fn, mode="r") as fin:
        for indx, item in enumerate(fin):
            if item.split("\t")[1] in {"train", "training", "20news-bydate-train"}:
                train_lst.append(indx)
            else:
                test_lst.append(indx)

    return train_lst, test_lst


class CELossWithPSO(th.autograd.Function):
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
        grad_input = th.neg((sum_cr/eta) * (ctx.gbest - yy))
        return grad_input, grad_output, None, None, None


class CELoss:
    def __init__(self, y):
        self.y = y
        self.fitness = th.nn.CrossEntropyLoss()
    def evaluate(self, x):
        # print(x, self.y)
        return self.fitness(x, self.y)



class PrepareData:
    def __init__(self, args):
        # print("prepare data")
        self.graph_path = "data/graph"
        self.args = args

        # graph
        graph = nx.read_weighted_edgelist(f"{self.graph_path}/{args.dataset}.txt"
                                          , nodetype=int)
        # print_graph_detail(graph)
        adj = nx.to_scipy_sparse_matrix(graph,
                                        nodelist=list(range(graph.number_of_nodes())),
                                        weight='weight',
                                        dtype=float)

        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        self.adj = preprocess_adj(adj, is_sparse=True)

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # features
        self.nfeat_dim = graph.number_of_nodes()
        row = list(range(self.nfeat_dim))
        col = list(range(self.nfeat_dim))
        value = [1.] * self.nfeat_dim
        shape = (self.nfeat_dim, self.nfeat_dim)
        indices = th.from_numpy(
                np.vstack((row, col)).astype(np.int64))
        values = th.FloatTensor(value)
        shape = th.Size(shape)

        self.features = th.sparse.FloatTensor(indices, values, shape)

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # target

        target_fn = f"data/text_dataset/{self.args.dataset}.txt"
        target = np.array(pd.read_csv(target_fn,
                                      sep="\t",
                                      header=None)[2])
        target2id = {label: indx for indx, label in enumerate(set(target))}
        self.target = [target2id[label] for label in target]
        self.nclass = len(target2id)

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # train val test split

        self.train_lst, self.test_lst = get_train_test(target_fn)


class TextGCNTrainer:
    def __init__(self, args, model, pre_data):
        self.args = args
        self.model = model
        self.device = args.device

        self.max_epoch = self.args.max_epoch
        self.set_seed()

        self.dataset = args.dataset
        self.predata = pre_data
        self.earlystopping = EarlyStopping(args.early_stopping)

    def set_seed(self):
        th.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)

    def fit(self, optimizer):
        self.prepare_data()
        self.typeOfOptimizer = optimizer
        self.model = self.model(nfeat=self.nfeat_dim,
                                nhid=self.args.nhid,
                                nclass=self.nclass,
                                dropout=self.args.dropout)
        # print(self.model.parameters)
        self.model = self.model.to(self.device)
        self.criterion = th.nn.CrossEntropyLoss()
        self.model_param = sum(param.numel() for param in self.model.parameters())
        # print('# model parameters:', self.model_param)
        self.convert_tensor()
        if optimizer == "Adam":
            self.optimizer = th.optim.Adam(self.model.parameters(), lr=self.args.lr)
        elif optimizer == "SGD":
            self.optimizer = th.optim.SGD(self.model.parameters(), lr=self.args.lr)
        elif optimizer == "PSO":
            getData = read_json_file()
            psoParameters = getData["PSO"]
            self.optimizer = ParticleSwarmOptimizer(self.model.parameters(), inertial_weight=psoParameters["inertia_weight"], num_particles=psoParameters["swarm_size"],
                                           cognitive_coefficient=psoParameters["cognitive_parameter"], social_coefficient=psoParameters["social_parameter"], max_param_value=1, min_param_value=-1)
        elif optimizer == "AdaSwarm":
            self.optimizer = th.optim.Adam(self.model.parameters(), lr=self.args.lr)
            self.criterion = CELossWithPSO.apply
        elif optimizer == "GDPSO":
            getData = read_json_file()
            gdPsoParameters = getData["GDPSO"]
            iterations = gdPsoParameters["iterations_SGD"]
            listOfModelParameters = [self.nfeat_dim, self.args.nhid, self.nclass, self.args.dropout, self.features, self.adj, self.train_lst, self.target]
            self.optimizer = GDPSO.GDPSO(listOfModelParameters,self.model.parameters(), inertial_weight=gdPsoParameters["inertia_weight"], num_particles=gdPsoParameters["swarm_size"], cognitive_coefficient =gdPsoParameters["cognitive_parameter"], social_coefficient=gdPsoParameters["social_parameter"], max_param_value=1,min_param_value=-1)
        elif optimizer == "AdamPSO":
            getData = read_json_file()
            adamPSOParameters = getData["AdamPSO"]
            iterations = adamPSOParameters["iterations_SGD"]
            listOfModelParameters = [self.nfeat_dim, self.args.nhid, self.nclass, self.args.dropout, self.features, self.adj, self.train_lst, self.target]
            self.optimizer = AdamPSO.AdamPSO(listOfModelParameters,self.model.parameters(), inertial_weight=adamPSOParameters["inertia_weight"], num_particles=adamPSOParameters["swarm_size"], cognitive_coefficient =adamPSOParameters["cognitive_parameter"], social_coefficient=adamPSOParameters["social_parameter"], max_param_value=1,min_param_value=-1)

        s = ""
        if optimizer == "Adam":
            start = time()
            s =self.trainAdam()
        elif optimizer == "SGD":
            start = time()
            s = self.trainSGD()
        elif optimizer == "PSO":
            start = time()
            s = self.trainPSO()
        elif optimizer == "AdaSwarm":
            start = time()
            s = self.trainAdaSwarm()
        elif optimizer == "GDPSO":
            start = time()
            s = self.trainGDPSO()
        elif optimizer == "AdamPSO":
            start = time()
            s = self.trainAdamPSO()
        self.train_time = time() - start
        return s

    @classmethod
    def set_description(cls, desc):
        string = ""
        for key, value in desc.items():
            if isinstance(value, int):
                string += f"{key}:{value} "
            else:
                string += f"{key}:{value:.4f} "
        # print(string)
        string += "\n"
        return string

    def prepare_data(self):
        self.adj = self.predata.adj
        self.nfeat_dim = self.predata.nfeat_dim
        self.features = self.predata.features
        self.target = self.predata.target
        self.nclass = self.predata.nclass

        self.train_lst, self.val_lst = train_test_split(self.predata.train_lst,
                                                        test_size=self.args.val_ratio,
                                                        shuffle=True,
                                                        random_state=self.args.seed)
        self.test_lst = self.predata.test_lst

    def convert_tensor(self):
        self.model = self.model.to(self.device)
        self.adj = self.adj.to(self.device)
        self.features = self.features.to(self.device)
        self.target = th.tensor(self.target).long().to(self.device)
        self.train_lst = th.tensor(self.train_lst).long().to(self.device)
        self.val_lst = th.tensor(self.val_lst).long().to(self.device)

    def trainPSO(self):
        bigString = ""

        for epoch in range(self.max_epoch):
            self.model.train()
            b = ""
            def closure():
                nonlocal b
                # if th.is_grad_enabled():
                #     optimizer.zero_grad()  # Clear gradients.
                self.optimizer.zero_grad()

                logits = self.model.forward(self.features, self.adj)
                loss = self.criterion(logits[self.train_lst],
                                      self.target[self.train_lst])
                b = loss
                return loss
            loss = self.optimizer.step(closure)            
            val_desc = self.val(self.val_lst)
            # loss = b
            desc = dict(**{"epoch"     : epoch,
                           "train_loss": loss.item(),
                           "train_loss": b,
                           }, **val_desc)            
            bigString += self.set_description(desc)

            # if self.earlystopping(val_desc["val_loss"]):
            #     break
        return bigString

    def trainAdaSwarm(self):
        getData = read_json_file()
        adaswarmParameters = getData["AdaSwarm"]
        dimensions = len(self.train_lst)
        swarm_size = adaswarmParameters["swarm_size"]
        numberOfClasses = self.nclass
        numberOfIterations = adaswarmParameters["iterations"]
        cognitive_coefficient = adaswarmParameters["cognitive_parameter"]
        social_coefficient = adaswarmParameters["social_parameter"]
        momentum = adaswarmParameters["momentum_factor"]
        options = [cognitive_coefficient, social_coefficient, momentum, 100]
        bigString = ""
        for epoch in range(self.max_epoch):
            self.model.train()            
            y = self.target[self.train_lst]
            y.requires_grad = False
            

            p = RotatedEMParicleSwarmOptimizer(dimensions=dimensions, swarm_size=swarm_size, classes=numberOfClasses, true_y=y, options=options,  iterations=numberOfIterations)
            p.optimize(CELoss(y))


            logits = self.model.forward(self.features, self.adj)

            # for i in range(numberOfIterations):
            #     print(i)
            #     c1r1, c2r2, gbest = p.run_one_iter(verbosity=False)
            c1r1, c2r2, gbest = p.run(verbosity=False)

            loss = self.criterion(logits[self.train_lst],
                                  y,c1r1 + c2r2, 0.1, gbest)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # val_desc = self.val(self.val_lst)
            prefix = "val"
            acc = accuracy(logits[self.val_lst],
                           self.target[self.val_lst])
            # f1, precision, recall = macro_f1(logits[self.val_lst],
            #                                  self.target[self.val_lst],
            #                                  num_classes=self.nclass)
            val_desc = {
                f"{prefix}_loss": loss.item(),
                  "accuracy": acc,
                # "macro_f1": f1,
                # "precision": precision,
                # "recall": recall,
            }
            desc = dict(**{"epoch": epoch,
                           "train_loss": loss.item(),
                           }, **val_desc)
            
            bigString += self.set_description(desc)

            if self.earlystopping(val_desc["val_loss"]):
                break
        return bigString

    def trainGDPSO(self):
        bigString = ""
        for epoch in range(self.max_epoch):
            self.model.train()
            b = ""
            def closure():
                nonlocal b
                if th.is_grad_enabled():
                    self.optimizer.zero_grad()
                logits = self.model.forward(self.features, self.adj)
                loss = self.criterion(logits[self.train_lst],
                                      self.target[self.train_lst])
                b = loss
                return loss
            self.optimizer.step(closure)

            val_desc = self.val(self.val_lst)
            loss = b
            desc = dict(**{"epoch"     : epoch,
                           "train_loss": loss.item(),                           
                           }, **val_desc)

            bigString += self.set_description(desc)

            if self.earlystopping(val_desc["val_loss"]):
                break
        return bigString
    
    def trainAdamPSO(self):
        bigString = ""
        for epoch in range(self.max_epoch):
            self.model.train()
            b = ""
            def closure():
                nonlocal b
                if th.is_grad_enabled():
                    self.optimizer.zero_grad()
                logits = self.model.forward(self.features, self.adj)
                loss = self.criterion(logits[self.train_lst],
                                      self.target[self.train_lst])
                b = loss
                return loss
            self.optimizer.step(closure)

            val_desc = self.val(self.val_lst)
            loss = b
            desc = dict(**{"epoch"     : epoch,
                           "train_loss": loss.item(),
                           "train_loss": b,
                           }, **val_desc)

            bigString += self.set_description(desc)

            if self.earlystopping(val_desc["val_loss"]):
                break
        return bigString
    
    
    def trainAdam(self):
        bigString = ""
        for epoch in range(self.max_epoch):
            self.model.train()
            self.optimizer.zero_grad()

            logits = self.model.forward(self.features, self.adj)
            loss = self.criterion(logits[self.train_lst],
                                  self.target[self.train_lst])            
            loss.backward()
            self.optimizer.step()

            val_desc = self.val(self.val_lst)


            desc = dict(**{"epoch"     : epoch,
                           "train_loss": loss.item(),
                           }, **val_desc)

            bigString += self.set_description(desc)

            if self.earlystopping(val_desc["val_loss"]):
                break
        return bigString
    def trainSGD(self):
        bigString = ""
        for epoch in range(self.max_epoch):
            self.model.train()
            self.optimizer.zero_grad()  # Clear gradients.
            logits = self.model.forward(self.features, self.adj)
            loss = self.criterion(logits[self.train_lst],
                                  self.target[self.train_lst])
            loss.backward()  # Derive gradients.

            self.optimizer.step()  # Update parameters based on gradients.

            val_desc = self.val(self.val_lst)

            desc = dict(**{"epoch": epoch,
                           "train_loss": loss.item(),
                           }, **val_desc)

            bigString += self.set_description(desc)
        return bigString

    @th.no_grad()
    def val(self, x, prefix="val"):
        self.model.eval()
        #If the olpimizer is the adaswarm, change the criterion to the cross entrop
        if self.typeOfOptimizer == "AdaSwarm":
            self.criterion = th.nn.CrossEntropyLoss()
        with th.no_grad():
            logits = self.model.forward(self.features, self.adj)
            loss = self.criterion(logits[x],
                                  self.target[x])
            acc = accuracy(logits[x],
                           self.target[x])
            # f1, precision, recall = macro_f1(logits[x],
            #                                  self.target[x],
            #                                  num_classes=self.nclass)

            desc = {
                f"{prefix}_loss": loss.item(),
                "accuracy"           : acc,
                # "macro_f1"      : f1,
                # "precision"     : precision,
                # "recall"        : recall,
            }
        return desc

    @th.no_grad()
    def test(self):
        self.test_lst = th.tensor(self.test_lst).long().to(self.device)
        test_desc = self.val(self.test_lst, prefix="test")
        test_desc["train_time"] = strftime("%H:%M:%S", gmtime(self.train_time))
        # print(strftime("%H:%M:%S", gmtime(self.train_time)))
        test_desc["model_param"] = self.model_param
        return test_desc

