import numpy as np
import os
import pandas as pd
from collections import Counter
from random import shuffle, sample
import time
import pickle

#This will balance a distributed dataset that is too big to load into RAM
#input: N .npy files, each around 200Mb.
#output: N randomized,balanced .npy files.

def removeData(FolderName, Dtype):

    if FolderName == "sortedData":
        for i in range(1,6):
            for root, dirs, files in os.walk("{}/{}/d{}".format(FolderName,Dtype,str(i))):
                for filename in files:
                    os.remove("{}/{}/d{}/{}".format(FolderName, Dtype, str(i), filename))
    elif FolderName == "collectedData":
        for root, dirs, files in os.walk("{}/{}".format(FolderName, Dtype)):
            for filename in files:
                os.remove("{}/{}/{}".format(FolderName, Dtype, filename))

def createUnbalancedTrainingData(Dtype):
    # Iterate through each motion for each file and construct the data
    # Smallest defines how many files we can have
    LastIndex = pickle.load(open("trainingData/Unbalanced_rgb_299/lastDataIndex_{}.p".format(Dtype), "rb"))
    dataIndex = pickle.load(open("collectedData/dataIndex_{}.p".format(Dtype), "rb")) # Get current trainingData data file number
    trainingIndex = pickle.load(open("trainingData/Unbalanced_rgb_299/dataIndex_{}.p".format(Dtype), "rb"))


    chunkIndex = 100

    for i in range(LastIndex, dataIndex):
        data = list(np.load("collectedData/{}/training_data_{}_{}.npy".format(Dtype, Dtype, str(i+1)))) # load a data file.
        shuffle(data)

        # Cuts into 5 parts of 100 samples
        for i in range(5):
            tmp = data[i*chunkIndex:((i+1)*chunkIndex)]
            print(trainingIndex)
            trainingIndex += 1
            np.save("trainingData/Unbalanced_rgb_299/{}/data_{}.npy".format(Dtype,str(trainingIndex)), tmp)

    pickle.dump(dataIndex,open("trainingData/Unbalanced_rgb_299/lastDataIndex_{}.p".format(Dtype), "wb")) # Update index
    pickle.dump(trainingIndex,open("trainingData/Unbalanced_rgb_299/dataIndex_{}.p".format(Dtype), "wb")) # Save final number

createUnbalancedTrainingData('both')
