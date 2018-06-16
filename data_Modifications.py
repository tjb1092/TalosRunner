import numpy as np
import pickle
import os
import cv2
from utils import viewData



def delete_data_file():
    # Remove file and re-adjust the file numbers
    file_number = input("enter the file number you wish to delete: ")
    Dtypes = ("body", "head")


    dataIndex = pickle.load(open("collectedData/dataIndex.p", "rb"))
    print(dataIndex)
    newIndex = dataIndex - 1

    for atype in Dtypes:
        os.remove("collectedData/{}/training_data_{}_{}.npy".format(atype, atype, str(file_number)))

        for i in range(int(file_number)+1,dataIndex+1):

            old_file_name = "collectedData/{}/training_data_{}_{}.npy".format(atype, atype, str(i))
            temp = np.load(old_file_name)
            os.remove(old_file_name)
            new_file_name = "collectedData/{}/training_data_{}_{}.npy".format(atype, atype, str(i-1))
            np.save(new_file_name, temp)

    print(newIndex)
    pickle.dump(newIndex,open("collectedData/dataIndex.p", "wb"))


"""
I want the option to look at data files to see if they are "good". There were a
couple of times I really messed up the puzzle, so they shouldn't really be in the
training data
"""

def view_data():
    dataIndex = pickle.load(open("collectedData/dataIndex.p","rb"))
    selectedFile = input("Pick a DF number from 1 to {}: ".format(dataIndex))
    data_type = input("body or head?\n")

    fname = "collectedData/{}/training_data_{}_{}.npy".format(data_type,data_type,selectedFile)
    viewData(fname)





view_data()
