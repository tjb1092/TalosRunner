import numpy as np
import cv2
import time
from directInputs import SendKeyPress, SendKeyRelease
from alexnet import alexnet
from screenGrab import grabscreen
import os
from Xlib import display, X
from utils import countDown


WIDTH = 100
HEIGHT = 100
LR = 1e-3
EPOCHS = 10
MODEL_NAME = 'pytalos-{}-{}-{}-epocs.model'.format(LR,'alexnet_body',EPOCHS)


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



model = alexnet(WIDTH,HEIGHT,LR)
model.load(MODEL_NAME)

def main():

    countDown(5)

    # Setup for screenGrab
    last_time = time.time()
    W,H = 575,525
    dsp = display.Display()
    root = dsp.screen().root

    while True:

        image = grabscreen(root, W,H)
        image = cv2.resize(image,(100,100))
        moves = list(np.around(model.predict([image.reshape(100,100,1)])[0]))
        print(moves)

        if moves == [1,0,0,0]:
            left()
        elif moves == [0,1,0,0]:
            right()
        elif moves == [0,0,1,0]:
            forwards()
        elif moves == [0,0,0,1]:
            backwards()

        print('loop took {} seconds'.format(time.time()-last_time))
        last_time = time.time()
        keysPressed = [] # refresh it each time we loop to keep out past inputs

        time.sleep(0.01)  # Needs to be there for the async hook and this synced loop to keep up with each other
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

main()
