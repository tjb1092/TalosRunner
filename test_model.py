import numpy as np
import cv2
import time
from screenGrab import grabscreen
import os
from Xlib import display, X
from utils import countDown, move_body, move_head, get_Keras_model
import uinput
import pickle
from keras.models import load_model

# Load model
WIDTH = HEIGHT = 224 # I don't like how split up this is and I'll have to think of a better way.
FILENUM = pickle.load(open("trainingData/Unbalanced_rgb_299/dataIndex_{}.p".format('both'), "rb"))

model = get_Keras_model('both', WIDTH, HEIGHT, 10, "AdamReg_Jesus", 608)

def main():

    countDown(5)
    # Setup for screenGrab
    last_time = time.time()
    W,H = 575,525
    dsp = display.Display()
    screen = dsp.screen()
    root = dsp.screen().root

    while True:

        image = grabscreen(root, W,H)
        # These need to be put into a conditional for the different types?
        image = cv2.resize(image,(WIDTH,HEIGHT))

        if mode == 3:
            
        #Make Predictions
        body_moves = [0, 0, 0, 0, 0]
        head_moves = [0, 0, 0, 0, 0]
        p_body, p_head = model.predict([image.reshape(1, WIDTH, HEIGHT, 3)])
        # This method ensures that each frame, the agent will pick the most likely option.
        body_m_index = np.argmax(p_body[0])
        head_m_index = np.argmax(p_head[0])
        body_moves[body_m_index] = 1
        head_moves[head_m_index] = 1

        move_body(body_moves)
        move_head(head_moves)

        print('loop took {:0.3f} seconds'.format(time.time()-last_time))
        last_time = time.time()

        time.sleep(0.01)  # Needs to be there for the async hook and this synced loop to keep up with each other
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

main()
