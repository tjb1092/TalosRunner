import numpy as np
import cv2
import time
from directInputs import SendKeyPress, SendKeyRelease
#from alexnet import alexnet, alexnet_color
from screenGrab import grabscreen
import os
from Xlib import display, X
from utils import countDown
import uinput


from keras.models import load_model
W = 575
H = 525
dy = 5 # pixels
dx = 5 # pixels

#need to add a timestamp to see how long it takes!
WIDTH = 200
HEIGHT = 200
LR = 1e-4
EPOCHS_1 = 40
EPOCHS_2 = 100
DTYPE = 'body'
OPTIMIZER = 'momentum'
DATA_TYPE = 'rgb_200'
ARCH = "VGG16"
FILENUM = 232 # Get this automatically in the future

MODEL_NAME = 'pytalos_{}_{}_{}_{}_files_{}_epocs_{}_{}.h5'.format(DTYPE, ARCH, OPTIMIZER, FILENUM, EPOCHS_1, DATA_TYPE,LR)
model_path = "models/{}/{}".format(DTYPE,MODEL_NAME)

print(model_path)
model = load_model(model_path)

def forwards():
    SendKeyPress('w')
    SendKeyRelease('s')
    SendKeyRelease('a')
    SendKeyRelease('d')

def backwards():
    SendKeyPress('s')
    SendKeyRelease('w')
    SendKeyRelease('a')
    SendKeyRelease('d')

def left():
    SendKeyPress('a')
    SendKeyRelease('s')
    SendKeyRelease('w')
    SendKeyRelease('d')

def right():
    SendKeyPress('d')
    SendKeyRelease('s')
    SendKeyRelease('a')
    SendKeyRelease('w')

def stop():
    SendKeyRelease('w')
    SendKeyRelease('s')
    SendKeyRelease('a')
    SendKeyRelease('d')

#Terrible I know, but whatever
def getMousePos(root):
    mouse_x = root.query_pointer()._data["root_x"]
    mouse_y = root.query_pointer()._data["root_y"]
    return mouse_x, mouse_y

#These can probably just be regular function calls in the main loop now. Maybe.
#It does look more intuitive with the names attached

def lookUp(d):
    d.emit(uinput.REL_Y,-1*dy)

def lookDown(d):
    d.emit(uinput.REL_Y,dy)

def lookLeft(d):
    d.emit(uinput.REL_X,-1*dx)

def lookRight(d):
    d.emit(uinput.REL_X,dx)

# Trying this out. It mimics what happens with the body function.
# It seemed like before, it was either too jerky or not fluid enough.
def look(d, direction):

    if direction == 1:
        d.emit(uinput.REL_Y,-1*dy)  # Up
    elif direction == 2:
        d.emit(uinput.REL_Y,dy)     # Down
    elif direction == 3:
        d.emit(uinput.REL_X,-1*dx)  # Left
    elif direction == 4:
        d.emit(uinput.REL_X,dx)     # Right
    # 5 will make nothing happen.


def main():

    countDown(5)
    # Setup for screenGrab
    last_time = time.time()
    W,H = 575,525
    dsp = display.Display()
    screen = dsp.screen()
    root = dsp.screen().root

    # This was such an obscure way to to this. X11 doesn't force the mouse in the game window
    # This device object does work, so that's nice that it works
    device = uinput.Device([
        uinput.BTN_LEFT,
        uinput.BTN_RIGHT,
        uinput.REL_X,
        uinput.REL_Y,
        ])

    choice = 5 #start not moving head

    while True:

        image = grabscreen(root, W,H)
        # These need to be put into a conditional for the different types?
        image = cv2.resize(image,(WIDTH,HEIGHT))
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        prediction = model.predict([image.reshape(1,WIDTH,HEIGHT,3)])[0]

        print(prediction)
        moves = list(np.around(prediction))
        print(moves)

        if moves == [1,0,0,0,0]:
            if DTYPE == 'body':
                left()
            elif DTYPE == 'head':
                choice = 1
                #lookUp(device)
        elif moves == [0,1,0,0,0]:
            if DTYPE == 'body':
                right()
            elif DTYPE == 'head':
                choice = 2
                #lookDown(device)
        elif moves == [0,0,1,0,0]:
            if DTYPE == 'body':
                forwards()
            elif DTYPE == 'head':
                choice = 3
                #lookLeft(device)
        elif moves == [0,0,0,1,0]:
            if DTYPE == 'body':
                backwards()
            elif DTYPE == 'head':
                choice = 4
                #lookRight(device)
        elif moves == [0,0,0,0,1]:
            if DTYPE == 'body':
                stop()
                #time.sleep(0.5) #Take a break
            elif DTYPE == 'head':
                choice = 5

        if DTYPE == 'head':
            look(device, choice)
        print('loop took {:0.3f} seconds'.format(time.time()-last_time))
        last_time = time.time()
        keysPressed = [] # refresh it each time we loop to keep out past inputs

        time.sleep(0.01)  # Needs to be there for the async hook and this synced loop to keep up with each other
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

main()
