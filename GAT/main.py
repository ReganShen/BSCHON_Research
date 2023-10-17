import random

import train


def addToFile(self, add):
    file = self.o + ".txt"
    f = open(file, "a")
    f.write(str(add + "\n"))
    f.close()




if __name__ == '__main__':
    random.seed(2)

    listOfSeeds = random.sample(range(0, 100000), 10)
    optimizerList = ["Adam", "SGD", "PSO","GDPSO","AdamPSO"]
    for o in optimizerList:
        for seeds in listOfSeeds:
            t = train.GATTrainer(o,seeds)
            t.startEverythiung(o)
