import numpy as np
from screenGrab import grabscreen
import cv2
import time
import os
from Xlib import display, X
import pyxhook
import math
from utils import countDown


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



def openNPFile(file_name):

    if os.path.isfile(file_name):
        print('File Exists, loading previous data!')
        training_data = list(np.load(file_name))
    else:
        print('File does not exist, starting fresh!')
        training_data = []
    return training_data

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
    body_fname = 'training_data_body.npy'
    head_fname = 'training_data_head.npy'
    training_data_body = openNPFile(body_fname)
    training_data_head = openNPFile(head_fname)

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

        if len(training_data_body) % 500 == 0:
            print(len(training_data_body))
            np.save(body_fname, training_data_body)
            np.save(head_fname, training_data_head)


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
