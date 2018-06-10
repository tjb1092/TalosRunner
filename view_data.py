import cv2
import pickle
from utils import viewData
"""
I want the option to look at data files to see if they are "good". There were a
couple of times I really messed up the puzzle, so they shouldn't really be in the
training data
"""


dataIndex = pickle.load(open("collectedData/dataIndex.p","rb"))
selectedFile = input("Pick a DF number from 1 to {}: ".format(dataIndex))
data_type = input("body or head?\n")

fname = "collectedData/{}/training_data_{}_{}.npy".format(data_type,data_type,selectedFile)
viewData(fname)
