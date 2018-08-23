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


# Mode == 1, Body only.
# Mode == 2, Body + Head
# Mode == 3, Both
mode = 3
isblur = False

# Load model
WIDTH = HEIGHT = 224 # I don't like how split up this is and I'll have to think of a better way.
FILENUM_body = pickle.load(open("trainingData/Unbalanced_rgb_299/dataIndex_{}.p".format('body'), "rb"))
FILENUM_head = pickle.load(open("trainingData/Unbalanced_rgb_299/dataIndex_{}.p".format('head'), "rb"))
FILENUM_both = pickle.load(open("trainingData/Unbalanced_rgb_299/dataIndex_{}.p".format('both'), "rb"))

if mode == 3:
    model = get_Keras_model('both', WIDTH, HEIGHT, 10, "AdamReg_Jesus", 608)
else:
    model_body = get_Keras_model('body', WIDTH, HEIGHT, 4, "Adam", FILENUM_body)
    if mode == 2:
        model_head = get_Keras_model('head', WIDTH, HEIGHT, 11, "AdamReg", FILENUM_head)



def blurImage(Imqueue):
    blurredImage = np.round(0.2*Imqueue[4]+0.2*Imqueue[3]+
                    0.2*Imqueue[2]+0.2*Imqueue[1]+
                    0.2*Imqueue[0]).astype('uint8')
    return blurredImage
    

def main():

    countDown(5)
    # Setup for screenGrab
    last_time = time.time()
    W,H = 575,525
    dsp = display.Display()
    screen = dsp.screen()
    root = dsp.screen().root

    if isblur:
        cnt = 0
        Imqueue = []

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

        else:
            prediction_body = model_body.predict([image.reshape(1,WIDTH,HEIGHT,3)])[0]
            moves_body = list(np.around(prediction_body))
            move_body(moves_body)
            print(moves_body)

            if mode == 2:

                if isblur:
                    if cnt < 5:
                        Imqueue.append(image)
                        cnt += 1
                    else:
                        #Keep a queue to cache the last 5 images.
                        Imqueue.reverse()
                        Imqueue.pop()
                        Imqueue.reverse()
                        Imqueue.append(image)
                        image = blurImage(Imqueue)
                        #print(len(Imqueue))
                        cv2.imshow('test', image)

                prediction_head = model_head.predict([image.reshape(1,WIDTH,HEIGHT,3)])[0]
                moves_head = list(np.around(prediction_head))
                move_head(moves_head)
                print(moves_head)

        print('loop took {:0.3f} seconds'.format(time.time()-last_time))
        last_time = time.time()

        time.sleep(0.01)  # Needs to be there for the async hook and this synced loop to keep up with each other
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

main()
