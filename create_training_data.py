import numpy as np
from screenGrab import grabscreen
import cv2
import time
import os
from Xlib import display, X
import pyxhook
import math
from utils import countDown
import pickle


def keys_to_output(keys):
    output = [0,0,0,0] # a, d, w, s

    if 'a' in keys:
        output[0] = 1
    elif 'd' in keys:
        output[1] = 1
    elif 'w' in keys:
        output[2] = 1
    elif 's' in keys:
        output[3] = 1

    return output

def mmove_to_output(root, prevX, prevY):
    output = [0,0,0,0] #up, down, left, right
    thresh = 5
    mouse_x, mouse_y = getMousePos(root)
    dx = prevX - mouse_x
    dy = prevY - mouse_y
    #print("{}, {}".format(dx, dy))
    if dy > thresh: #up
        output[0] = 1
    elif (-1.*dy) > thresh: #down
        output[1] = 1
    elif dx > thresh: #left
        output[2] = 1
    elif (-1.*dx) > thresh: #right
        output[3] = 1

    return output, mouse_x, mouse_y

# More complicated data saving methods
def open_npy_file(dataIndex, Dtype):

    if dataIndex == 0:
        training_data=[]
    else:
        training_data = list(np.load("collectedData/" + Dtype +"/training_data_" + Dtype +  "_" + str(dataIndex)+".npy")) # load latest file.

    return training_data

def save_npy_file(Dtype, dataIndex, training_data, pickle_fname, pWrite):
    # Assume the dataIndex corresponds with the current data file.
    if pWrite == 0:
        dataIndex += 1

    #Write to new datafile
    np.save("collectedData/"+Dtype+"/training_data_"+Dtype+"_"+str(dataIndex), training_data)
    if pWrite == 1:
        #update the stored index. Only for head right now.
        pickle.dump(dataIndex, open(pickle_fname, "wb"))

    return dataIndex

def open_dataIndex(pickle_fname):
    if os.path.isfile(pickle_fname):
        dataIndex = pickle.load(open(pickle_fname,"rb"))
    else:
        dataIndex = 0
    return dataIndex



#this function is called everytime a key is pressed.
def OnKeyPress(event):
    global keysPressed
    keysPressed.append(event.Key)  # Append key to global key press list

    if event.Ascii==96: #96 is the ascii value of the grave key (`)
        new_hook.cancel()

def getMousePos(root):
    mouse_x = root.query_pointer()._data["root_x"]
    mouse_y = root.query_pointer()._data["root_y"]
    return mouse_x, mouse_y

def main():
    global keysPressed
    pickle_fname = "collectedData/dataIndex.p"
    dataIndex = open_dataIndex(pickle_fname)
    training_data_body = open_npy_file(dataIndex, "body")
    training_data_head = open_npy_file(dataIndex, "head")

    countDown(5)

    # Setup for screenGrab
    last_time = time.time()
    W,H = 575,525
    # This beautiful solution comes from JHolta on:
    # https://stackoverflow.com/questions/69645/take-a-screenshot-via-a-python-script-linux
    # It can be improved to do C-based implementation, but it is basically real-time now.
    dsp = display.Display()
    root = dsp.screen().root
    prevX, prevY = getMousePos(root)

    while True:

        #for k in keysPressed:
            #print(k)
        image = grabscreen(root, W,H)
        image = cv2.resize(image,(100,100))
        output_body = keys_to_output(keysPressed)
        output_head, prevX, prevY = mmove_to_output(root, prevX, prevY)

        #print(output_body)
        #print(output_head)
        training_data_body.append([image, output_body])
        training_data_head.append([image, output_head])

        print('loop took {} seconds'.format(time.time()-last_time))
        last_time = time.time()
        keysPressed = [] # refresh it each time we loop to keep out past inputs
        cv2.imshow('window',image)
        time.sleep(0.01)  # Needs to be there for the async hook and this synced loop to keep up with each other
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

        if len(training_data_body) % 1000 == 0:
            print(len(training_data_body))
            dataIndex = save_npy_file("body", dataIndex, training_data_body, pickle_fname, 0)
            dataIndex = save_npy_file("head", dataIndex, training_data_head, pickle_fname, 1)
            # Restarts array for next batch
            training_data_body = []
            training_data_head = []


keysPressed = []
#instantiate HookManager class
new_hook=pyxhook.HookManager()
#listen to all keystrokes
new_hook.KeyDown=OnKeyPress
#hook the keyboard
new_hook.HookKeyboard()
#start the session
new_hook.start()
main()
