import time
import numpy as np
import cv2
#from ctypes import cdll

def countDown(cnt):
    #Delay before recording takes place
    for i in list(range(cnt))[::-1]:
        print(i+1)
        time.sleep(1)

def viewData(fname):
    training_data = np.load(fname)

    #Look at the data!
    for data in training_data:
        image = data[0]
        #image = cv2.resize(image,(1000,1000)) #That hurt my eyes lol
        cv2.imshow('test', image)
        print(data[1])
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
def setPickleIndex(pickle_fname,val):
    #update the stored index. Only for head right now.
    pickle.dump(val, open(pickle_fname, "wb"))

"""
#From DrMeers from ubuntu forums
#https://ubuntuforums.org/showthread.php?t=853369
dll = cdll.LoadLibrary('libX11.so') # Only load this once. No other ambiguity here.
def mouseMove(x,y):
    d = dll.XOpenDisplay(None)
    root = dll.XDefaultRootWindow(d)
    dll.XWarpPointer(d,None,root,0,0,0,0,x,y)
    dll.XCloseDisplay(d)
"""
