import numpy as np
from screenGrab import grabscreen
import cv2
import time
import os
from Xlib import display, X
import pyxhook
import math
from utils import countDown, save_npy_file, open_dataIndex
import pickle


def keys_to_output(keys):
    output = [0,0,0,0,0] # a, d, w, s, nothing
    #print(keys)
    if 'a' in keys or 'A' in keys:
        output[0] = 1
    elif 'd' in keys or 'D' in keys:
        output[1] = 1
    elif 'w' in keys or 'W' in keys:
        output[2] = 1
    elif 's' in keys or 'S' in keys:
        output[3] = 1
    elif len(keys) == 0:
        output[4] = 1
    print("Body:")
    print(output)
    return output

#this function is called everytime a key is pressed.
def OnKeyPress(event):
    global keysPressed
    keysPressed.append(event.Key)  # Append key to global key press list

    if event.Ascii==96: #96 is the ascii value of the grave key (`)
        new_hook.cancel()

def main():
    SaveData = False
    global keysPressed
    pickle_fname = "collectedData/dataIndex_body.p"
    dataIndex = open_dataIndex(pickle_fname)

    training_data = []

    countDown(5)

    # Setup for screenGrab
    last_time = time.time()
    W,H = 575,525
    # This beautiful solution comes from JHolta on:
    # https://stackoverflow.com/questions/69645/take-a-screenshot-via-a-python-script-linux
    # It can be improved to do C-based implementation, but it is basically real-time now.
    dsp = display.Display()
    root = dsp.screen().root

    while True:

        image = grabscreen(root, W,H)
        image = cv2.resize(image,(299,299))
        print(keysPressed)
        output_body = keys_to_output(keysPressed)
        #print(output_body)

        training_data.append([image, output_body])
        keysPressed = [] # refresh it each time we loop to keep out past inputs

        #cv2.imshow('window',image) # Significant framedrop to look at the captured image.

        time.sleep(0.01)  # Needs to be there for the async hook and this synced loop to keep up with each other
        print('loop took {:0.3f} seconds'.format(time.time()-last_time))
        last_time = time.time()
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

        if len(training_data) % 500 == 0:
            if SaveData:
                print(dataIndex)
                dataIndex = save_npy_file("body", dataIndex, training_data, pickle_fname)
            # Restarts array for next batch
            training_data = []


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
