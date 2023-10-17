import torch
import trainer
from utils import parameter_parser
import torch as th
from layer import GCN
from utils import CudaUse
from utils import LogResult
import gc
import random
import multiprocessing
import torch.multiprocessing as mp
def addToFile(optimizerFile, add):
    file = optimizerFile + ".txt"
    f = open(file, "a")
    f.write(str(add))
    f.close()


def runCode():
    #dataset = "mr"  #Mr dataset
    dataset = "20ng"  # 20NG dataset
    args = parameter_parser()
    args.dataset = dataset

    args.device = th.device('cuda') if th.cuda.is_available() else th.device('cpu')
    args.nhid = 200
    args.max_epoch = 400
    args.dropout = 0.5
    args.val_ratio = 0.1
    args.early_stopping = 10
    args.lr = 0.02
    model = GCN

    predata = trainer.PrepareData(args)



    seed_lst = list()
    optimizerList = ["Adam", "SGD", "PSO", "GDPSO", "AdamPSO"]

    random.seed(2)

    listOfSeeds = random.sample(range(0, 100000), 10)
    strng = vars(args)

    for o in optimizerList:
        ind = 0
        addToFile(o, strng)
        for seed in listOfSeeds:
            print(seed)
            stringToReturn = ""
            stringToReturn += f"\n\n==> {ind}, seed:{seed}" + "\n"
            stringToReturn += "Optimizer : " + o + "\n"
            args.seed = seed
            seed_lst.append(seed)

            framework = trainer.TextGCNTrainer(model=model, args=args, pre_data=predata)

            s = framework.fit(o)
            stringToReturn += "======= Train =======\n"
            stringToReturn += str(s)
            t = framework.test()
            stringToReturn += "======= Test =======\n"
            stringToReturn += str(t) + "\n"

            del framework
            gc.collect()

            addToFile(o, stringToReturn)
            ind = ind + 1
        lastly = "==> seed set:\n"
        lastly += str(seed_lst)
        lastly += "\n"
        addToFile(o, lastly)

    # record.show_str()


if __name__ == '__main__':    
    runCode()