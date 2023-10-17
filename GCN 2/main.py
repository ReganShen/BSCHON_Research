import random
import torch
import train
if __name__ == '__main__':
    optimizerList = ["Adam", "SGD", "PSO","GDPSO", "AdamPSO"]
    random.seed(2)
    listOfSeeds = random.sample(range(0, 100000), 10)
    for o in optimizerList:
        for s in listOfSeeds:
            if o == "Adam":
                t = train.training(s)
                t.runAdam(s)
            elif o == "SGD":
                t = train.training(s)
                t.runSGD(s)
            elif o == "PSO":
                t = train.training(s)
                t.runPSO(s)
            elif o == "AdaSwarm":
                t = train.training(s)
                t.runAdaSwarm(s)
            elif o == "GDPSO":
                t = train.training(s)
                t.runGDPSO(s)
            elif o == "AdamPSO":
                t = train.training(s)
                t.runAdamPSO(s)



