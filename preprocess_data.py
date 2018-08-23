import numpy as np
import os
import pandas as pd
from collections import Counter
from random import shuffle, sample
import time
import pickle

#This will balance a distributed dataset that is too big to load into RAM
# This is for Conv nets only and won't be useful for recurrent nets.
#input: N .npy files, each around 200Mb.
#output: N randomized,balanced .npy files.


BATCH_SIZE = 100 # Needs to be divisible by 5. 20/20/20/20

def prepare_Data(Datafolder, Dtype):
    FullFolder =os.path.join(Datafolder,Dtype)
    data = []
    counts = [0,0,0,0,0]
    d1Cnt = d5Cnt = d4Cnt = d3Cnt = d2Cnt = 0
    last_time = time.time()
    Indicies = [0,0,0,0,0]

    for x in os.walk(FullFolder):
        if x[0] == FullFolder:
            for y in x[2]:
                F = y.replace(FullFolder, "")
                f_file = os.path.join(x[0], F)
                print(f_file)

                #Step 1: Load a file
                n_file = np.load(f_file)

                #Step 2: sort the data into categories
                d1, d2, d3, d4, d5 = sort_Data_Labels(n_file)
                # I think each should be shuffled so data is random when put into folder
                shuffle(d1)
                shuffle(d2)
                shuffle(d3)
                shuffle(d4)
                shuffle(d5)
                Data = [d1, d2, d3, d4, d5] #These are hopefully small enough to not make this super slow
                # Load last data file. If not at max, fill it up before moving on.
                for i in range(5):
                    counts[i] += len(Data[i]) # Tally up the counts for later analysis

                    #Ensure each file adds up to 100 samples over 5 sample types.
                    if Dtype == 'body':
                        if i+1 == 4:
                            limit = 4
                        else:
                            limit = 24
                    else:
                        limit = 20
                    local_Index = 0
                    data_file = []
                    if Indicies[i] != 0:
                        data_file = list(np.load("sortedData/{}/d{}/data_{}.npy".format(Dtype, str(i+1), str(Indicies[i]-1)))) # load latest file.

                        fileLen = len(data_file)
                        if len(Data[i]) != 0: # Don't do this if there's no data to do it with!
                            for sample in range(limit-fileLen):
                                #Only executes if there is a < difference.

                                data_file.append(Data[i][local_Index])
                                local_Index += 1

                    for j in range(local_Index, len(Data[i])):
                        if len(data_file) == limit:
                            # Save and restart
                            np.save("sortedData/{}/d{}/data_{}.npy".format(Dtype, str(i+1), str(Indicies[i])), data_file)
                            Indicies[i] += 1 # Increment file count for new file
                            data_file = [] #Restart file
                            data_file.append(Data[i][j]) # Still append present sample
                        else:
                            data_file.append(Data[i][j])

                    if len(data_file) != 0:
                        # One more save for last, non-uniform sample
                        np.save("sortedData/{}/d{}/data_{}.npy".format(Dtype, str(i+1), str(Indicies[i])), data_file)
                print("Counts")
                print(counts)
                print("Indicies")
                print(Indicies)
    #Works pretty well, but there's something weird going on where I'm making too many files.
    #It is about 5Gb bigger than the original folder, so there is definitely some overlap somehow.
    print('process took {:0.3f} min'.format((time.time()-last_time)/60.))
    last_time = time.time()

    #Okay now we should have each file struct managed properly
    #Create the balanced training data set
    createBalancedTrainingData(Indicies, Dtype)
    print('process took {:0.3f} min'.format((time.time()-last_time)/60.))
    last_time = time.time()
    # Delete all data after to save space. It doesn't seem like I need it afterwards. I don't think this is super damaging to the hard drive.
    removeData("sortedData", Dtype)
    #removeData("collectedData", Dtype)


#making it its own function so I can test it out without making the data each time lol.
def createBalancedTrainingData(Indicies, Dtype):
    #Iterate through each motion for each file and construct the data
    # Smallest defines how many files we can have
    training_file_len = min(Indicies)
    trainingFileIndex = []
    dataIndex = pickle.load(open("trainingData/rgb_299/dataIndex_{}.p".format(Dtype), "rb")) # Get current trainingData data file number

    for i in range(len(Indicies)):
        choices = sample(range(Indicies[i]), training_file_len) # pull N random samples from the avaliable files
        trainingFileIndex.append(choices)

    for i in range(training_file_len):
        data_file = []
        for j in range(5):
            print("sortedData/{}/d{}/data_{}.npy".format(Dtype, str(j+1), str(trainingFileIndex[j][i])))
            d = list(np.load("sortedData/{}/d{}/data_{}.npy".format(Dtype, str(j+1), str(trainingFileIndex[j][i])))) # load a data file.
            data_file = data_file + d
        print(len(data_file))
        shuffle(data_file)
        dataIndex += 1
        np.save("trainingData/rgb_299/{}/data_{}.npy".format(Dtype,str(dataIndex)), data_file)

    pickle.dump(dataIndex,open("trainingData/rgb_299/dataIndex_{}.p".format(Dtype), "wb")) # Save final number

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


def sort_Data_Labels(data):

    df = pd.DataFrame(data)

    print(Counter(df[1].apply(str)))


    d1 = []
    d2 = []
    d3 = []
    d4 = []
    d5 = []
    for data in data:
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

    return d1, d2, d3, d4, d5

def createUnbalancedTrainingData(Dtype, isblur):
    # Iterate through each motion for each file and construct the data
    # Smallest defines how many files we can have
    LastIndex = pickle.load(open("trainingData/Unbalanced_rgb_299/lastDataIndex_{}.p".format(Dtype), "rb"))
    dataIndex = pickle.load(open("collectedData/dataIndex_{}.p".format(Dtype), "rb")) # Get current trainingData data file number
    trainingIndex = pickle.load(open("trainingData/Unbalanced_rgb_299/dataIndex_{}.p".format(Dtype), "rb"))

    if isblur:
        chunkIndex = 99
    else:
        chunkIndex = 100

    for i in range(LastIndex, dataIndex):
        data = list(np.load("collectedData/{}/training_data_{}_{}.npy".format(Dtype, Dtype, str(i+1)))) # load a data file.
        if isblur:
            data = blurData(data)
        shuffle(data)

        # Cuts into 5 parts of 100 samples
        for i in range(5):
            tmp = data[i*chunkIndex:((i+1)*chunkIndex)]
            print(trainingIndex)
            trainingIndex += 1
            np.save("trainingData/Unbalanced_rgb_299/{}/data_{}.npy".format(Dtype,str(trainingIndex)), tmp)

    pickle.dump(dataIndex,open("trainingData/Unbalanced_rgb_299/lastDataIndex_{}.p".format(Dtype), "wb")) # Update index
    pickle.dump(trainingIndex,open("trainingData/Unbalanced_rgb_299/dataIndex_{}.p".format(Dtype), "wb")) # Save final number

def blurData(data):
    blurredData = []
    for i in range(4,len(data)-1):
        D = np.round(0.2*data[i][0]+0.2*data[i-1][0]+0.2*data[i-2][0]+0.2*data[i-3][0]+0.2*data[i-4][0]).astype('uint8')
        blurredData.append([D,data[i][1]]) # Preceeding 5 frames and present frame's prediction.
    #print(len(blurredData))
    return blurredData

# Maybe these could be in a main function that allows me to choose? Is it worth it?
#prepare_Data('collectedData', 'body')
#prepare_Data('collectedData', 'head')
createUnbalancedTrainingData('both',False)
