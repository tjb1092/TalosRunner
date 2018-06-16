import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle
import cv2
import os

def open_concat_Data(Datafolder, Dtype):
    FullFolder =os.path.join(Datafolder,Dtype)
    data = []
    for x in os.walk(FullFolder):
        if x[0] == FullFolder:
            for y in x[2]:
                F = y.replace(FullFolder, "")
                f_file = os.path.join(x[0], F)
                print(f_file)
                n_file = np.load(f_file)
                if len(data) == 0:
                    data = n_file
                else:
                    data = np.concatenate((data, n_file))
    return data


def balanceData(Datafolder, Dtype):

    training_data = open_concat_Data(Datafolder, Dtype)

    df = pd.DataFrame(training_data)

    print(Counter(df[1].apply(str)))


    d1 = []
    d2 = []
    d3 = []
    d4 = []
    d5 = []
    for data in training_data:
        img = data[0]
        choice = data[1]

        if choice == [1,0,0,0,0]:
            d1.append([img, choice])
        elif choice == [0,1,0,0,0]:
            d2.append([img, choice])
        elif choice == [0,0,1,0,0]:
            d3.append([img, choice])
        elif choice == [0,0,0,1,0]:
            d4.append([img, choice])
        elif choice == [0,0,0,0,1]:
            d5.append([img, choice])

    if Dtype == 'body':
        indexLen = np.sort(np.array([len(d1), len(d2), len(d3), len(d4), len(d5)]))[1] # Pick longest len
    else:
        indexLen = np.sort(np.array([len(d1), len(d2), len(d3), len(d4), len(d5)]))[0] # Pick longest len

    #This ensures that all parts of the data have an opportunity to get into the training set
    shuffle(d1)
    shuffle(d2)
    shuffle(d3)
    shuffle(d4)
    shuffle(d5)

    d1 = d1[:indexLen]

    d2 = d2[:indexLen]
    d3 = d3[:indexLen]
    if Dtype != 'body':
        d4 = d4[:indexLen]
    d5 = d5[:indexLen]
    final_data = d1 + d2 + d3 + d4 + d5
    shuffle(final_data)
    print("Number of Training Samples: {}".format(len(final_data)))
    np.save(os.path.join('.','preprocessedTrainingData',Dtype,"training_data_"+Dtype), final_data)


balanceData('collectedData', 'body')
balanceData('collectedData', 'head')
