
def getTrainData():
    with open("data/train.csv",mode='r') as f:
        for line in f.readlines()[0:10]:
            print(line)


if(__name__=="__main__"):
    getTrainData()