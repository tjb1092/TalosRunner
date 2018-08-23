import numpy as np
from screenGrab import grabscreen
import cv2
import time
from Xlib import display, X
import pyxhook
import math
from utils import countDown, save_npy_file, open_dataIndex, move_body
import pickle
from keras.models import load_model
#import evdev

def h_keys_to_output(keys):
    output = [0,0,0,0,0] # a, d, w, s, nothing.
    if 'j' in keys or 'J' in keys:
        output[0] = 1
    elif 'l' in keys or 'L' in keys:
        output[1] = 1
    elif 'i' in keys or 'I' in keys:
        output[2] = 1
    elif 'k' in keys or 'K' in keys:
        output[3] = 1
    else: # Probably needs to be different b/c the body AI will be running as well. Maybe just "else?"
        output[4] = 1
    #print(keys)
    print(output, end="\r")
    return output

#this function is called everytime a key is pressed.
def OnKeyPress(event):
    global keysPressed
    #If not in list, add it.
    if event.Key.lower() not in keysPressed:
        keysPressed.append(event.Key.lower())  # Append key to global key press list

    if event.Ascii==96: #96 is the ascii value of the grave key (`)
        new_hook.cancel()

#This gets called everything a key is released
def OnKeyRelease(event):
    global keysPressed
    #If in list, remove it.
    if event.Key.lower() in keysPressed:
        keysPressed.remove(event.Key.lower())  # Append key to global key press list

    if event.Ascii==96: #96 is the ascii value of the grave key (`)
        new_hook.cancel()

def main():
    SaveData = True
    global keysPressed
    pickle_fname = "collectedData/dataIndex_head.p"
    dataIndex = open_dataIndex(pickle_fname)

    training_data= []

    #This chunk could probably be abstracted out?
    WIDTH = HEIGHT = 224
    LR = 1e-4
    EPOCHS_1 = 4
    DTYPE = 'body'
    OPTIMIZER = 'Adam'
    DATA_TYPE = "Unbalanced_rgb_299"
    ARCH = "VGG16"
    #FILENUM = pickle.load(open("trainingData/Unbalanced_rgb_299/dataIndex_{}.p".format(DTYPE), "rb"))
    FILENUM = 3335
    MODEL_NAME = 'pytalos_{}_{}_{}_{}_files_{}_epocs_{}_{}.h5'.format(DTYPE, ARCH, OPTIMIZER, FILENUM, EPOCHS_1, DATA_TYPE,LR)
    model_path = "models/{}/{}".format(DTYPE,MODEL_NAME)

    print(model_path)
    model = load_model(model_path)
    model.load_weights("models/{}/best_weights_{}".format(DTYPE,MODEL_NAME))

    # Setup for screenGrab
    last_time = time.time()
    W,H = 575,525
    dsp = display.Display()
    root = dsp.screen().root

    countDown(5)

    while True:
        #print(keysPressed)
        image = grabscreen(root, W,H)
        image = cv2.resize(image,(299,299))
        image_predict = cv2.resize(image,(WIDTH,HEIGHT))
        # Move the body.
        prediction = model.predict([image_predict.reshape(1,WIDTH,HEIGHT,3)])[0]
        moves = list(np.around(prediction))
        move_body(moves)

        # Classify the movement for the head.
        output_head = h_keys_to_output(keysPressed)
        training_data.append([image, output_head])

        #cv2.imshow('window',image)
        time.sleep(0.01)  # Needs to be there for the async hook and this synced loop to keep up with each other
        #print('loop took {:0.3f} seconds'.format(time.time()-last_time))
        last_time = time.time()

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

        if len(training_data) % 500 == 0:
            if SaveData:
                print(dataIndex)
                dataIndex = save_npy_file("head", dataIndex, training_data, pickle_fname)
            # Restarts array for next batch
            training_data = []


keysPressed = []
#instantiate HookManager class
new_hook=pyxhook.HookManager()
#listen to all keystrokes
new_hook.KeyDown=OnKeyPress
new_hook.KeyUp=OnKeyRelease


#hook the keyboard
new_hook.HookKeyboard()
#start the session
new_hook.start()

main()
