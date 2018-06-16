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
    SendKeyRelease('d')
    SendKeyRelease('s')
    SendKeyRelease('a')
    SendKeyRelease('w')

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


model = alexnet(WIDTH,HEIGHT,LR, MODEL_NAME)
model.load(MODEL_NAME)


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

    while True:

        image = grabscreen(root, W,H)
        image = cv2.resize(image,(100,100))
        moves = list(np.around(model.predict([image.reshape(100,100,1)])[0]))
        print(moves)

        if moves == [1,0,0,0,0]:
            if Dtype == 'body':
                left()
            elif Dtype == 'head':
                lookUp(device)
        elif moves == [0,1,0,0,0]:
            if Dtype == 'body':
                right()
            elif Dtype == 'head':
                lookDown(device)
        elif moves == [0,0,1,0,0]:
            if Dtype == 'body':
                forwards()
            elif Dtype == 'head':
                lookLeft(device)
        elif moves == [0,0,0,1,0]:
            if Dtype == 'body':
                backwards()
            elif Dtype == 'head':
                lookRight(device)
        elif moves == [0,0,0,0,1]:
            stop()
            #time.sleep(0.5) #Take a break

        print('loop took {} seconds'.format(time.time()-last_time))
        last_time = time.time()
        keysPressed = [] # refresh it each time we loop to keep out past inputs

        time.sleep(0.01)  # Needs to be there for the async hook and this synced loop to keep up with each other
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

main()
