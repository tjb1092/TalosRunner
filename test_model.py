import numpy as np
import cv2
import time
from directInputs import SendKeyPress, SendKeyRelease
from alexnet import alexnet
from screenGrab import grabscreen
import os
from Xlib import display, X
from utils import countDown
import uinput

W = 575
H = 525
dy = 1 # pixels
dx = 1 # pixels

WIDTH = 100
HEIGHT = 100
LR = 1e-4
EPOCHS = 100
Dtype = 'body'
model_path = 'models/{}/'.format(Dtype)
MODEL_NAME = 'pytalos-{}-{}-{}-epocs.model'.format(LR,'alexnet_{}_6kSamples'.format(Dtype),EPOCHS)
print(MODEL_NAME)

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

model = alexnet(WIDTH,HEIGHT,LR, MODEL_NAME)
model.load(model_path + MODEL_NAME)


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
        image = cv2.resize(image,(100,100))
        moves = list(np.around(model.predict([image.reshape(100,100,1)])[0]))
        print(moves)

        if moves == [1,0,0,0,0]:
            if Dtype == 'body':
                left()
            elif Dtype == 'head':
                choice = 1
                #lookUp(device)
        elif moves == [0,1,0,0,0]:
            if Dtype == 'body':
                right()
            elif Dtype == 'head':
                choice = 2
                #lookDown(device)
        elif moves == [0,0,1,0,0]:
            if Dtype == 'body':
                forwards()
            elif Dtype == 'head':
                choice = 3
                #lookLeft(device)
        elif moves == [0,0,0,1,0]:
            if Dtype == 'body':
                backwards()
            elif Dtype == 'head':
                choice = 4
                #lookRight(device)
        elif moves == [0,0,0,0,1]:
            if Dtype == 'body':
                stop()
                #time.sleep(0.5) #Take a break
            elif Dtype == 'head':
                choice = 5

        if Dtype == 'head':
            look(device, choice)
        print('loop took {} seconds'.format(time.time()-last_time))
        last_time = time.time()
        keysPressed = [] # refresh it each time we loop to keep out past inputs

        time.sleep(0.01)  # Needs to be there for the async hook and this synced loop to keep up with each other
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

main()
