import os
import glob
import pickle
import shutil
import subprocess
import cv2
import time
from random import shuffle
import numpy as np
from Xlib import display, X

import keras
from keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam, SGD
from keras.models import load_model
from screenGrab import grabscreen

import pyxhook
from utils import countDown, move_body, move_head, get_Keras_model
from dataGenerator import DataGenerator


def SupervisorCheck(Image, body_moves, head_moves, trainingData):
    global keysPressed

    noop = [0,0,0,0,1]
    #saveTraining = False
    noop_b = noop_h = False
    if 'shift_r' in keysPressed:
        noop_b = True
    if 'p_home' in keysPressed:
        noop_h = True

    supervised_body = sb_keys_to_output(keysPressed)
    supervised_head = sh_keys_to_output(keysPressed)

    if supervised_body == noop and not noop_b:
        #If the supervisor isn't doing anything, proceed with agent's choice.
        output_body = body_moves
    else:
        if noop_b:
            output_body = noop
        else:
            output_body = supervised_body #else, use the supervisor's decision.

    if supervised_head == noop and not noop_h:
        #If the supervisor isn't doing anything, proceed with agent's choice.
        output_head = head_moves
    else:
        if noop_h:
            output_head = noop
        else:
            output_head = supervised_head

    move_body(output_body)
    move_head(output_head)

    trainingData.append([Image,output_body,output_head])

    return trainingData


def sb_keys_to_output(keys):
    # agent:    a, d, w, s, nothing
    # me:       left, right, up, down, nothing
    output = [0,0,0,0,0]
    #print(keys)
    if 'left' in keys:
        output[0] = 1
    elif 'right' in keys:
        output[1] = 1
    elif 'up' in keys:
        output[2] = 1
    elif 'down' in keys:
        output[3] = 1
    else:
        output[4] = 1

    print("Supervised Body:")
    print(output)
    return output

def sh_keys_to_output(keys):
    # agent:    j, l, i, k, nothing.
    # me:       p_left, p_right, p_up, p_begin, nothing
    output = [0,0,0,0,0]
    #print(keys)
    if 'p_left' in keys:
        output[0] = 1
    elif 'p_right' in keys:
        output[1] = 1
    elif 'p_up' in keys:
        output[2] = 1
    elif 'p_begin' in keys:
        output[3] = 1
    else:
        output[4] = 1
    print("Supervised Head:")
    print(output)
    return output

#this function is called everytime a key is pressed. Presently, this also can't be shoved its
#own file because of the hook and possibly due to the global variable.
def OnKeyPress(event):
    global keysPressed
    #If not in list, add it.
    if event.Key.lower() not in keysPressed:
        keysPressed.append(event.Key.lower())  # Append key to global key press list

    if event.Ascii==96: #96 is the ascii value of the grave key (`)
        new_hook.cancel()

#This gets called everything a key is released. Presently, this also can't be shoved its
#own file because of the hook and possibly due to the global variable.
def OnKeyRelease(event):
    global keysPressed
    #If in list, remove it.
    if event.Key.lower() in keysPressed:
        keysPressed.remove(event.Key.lower())  # Append key to global key press list

    if event.Ascii==96: #96 is the ascii value of the grave key (`)
        new_hook.cancel()


def main():
    ############################
    # Initialize
    global keysPressed
    EpisodicMemCap = 100
    dataIndex_episode = pickle.load(open("trainingData/episodeData/dataIndex_episode.p", "rb"))

    # Load model
    WIDTH = HEIGHT = 224 # I don't like how split up this is and I'll have to think of a better way.
    FILENUM_both = pickle.load(open("trainingData/Unbalanced_rgb_299/dataIndex_{}.p".format('both'), "rb"))
    #FILENUM_both = 458
    model = get_Keras_model('both', WIDTH, HEIGHT, 8, "AdamReg", FILENUM_both)

    # Setup for screenGrab
    last_time = time.time()
    W, H = 575, 525
    dsp = display.Display()
    screen = dsp.screen()
    root = dsp.screen().root

    countDown(5)
    dataIndex = 0
    cnter = 0
    while True:
        #Reinforcement Loop
        isSolved = False
        trainingData = []


        while not isSolved:
            #During the puzzle, this loop will play till it gets determined that the puzzle has been solved.

            ############################
            #Get Image
            image = grabscreen(root, W, H)
            image_rs = cv2.resize(image,(WIDTH, HEIGHT))

            ############################
            #Make Predictions
            body_moves = [0, 0, 0, 0, 0]
            head_moves = [0, 0, 0, 0, 0]
            p_body, p_head = model.predict([image_rs.reshape(1, WIDTH, HEIGHT, 3)])
            # This method ensures that each frame, the agent will pick the most likely option.
            body_m_index = np.argmax(p_body[0])
            head_m_index = np.argmax(p_head[0])

            body_moves[body_m_index] = 1
            head_moves[head_m_index] = 1

            #body_moves = list(np.around(p_body[0]).astype(int))
            #head_moves = list(np.around(p_head[0]).astype(int))
            if np.all(np.asarray(body_moves) == 0) or np.all(np.asarray(head_moves) == 0):
                cnter += 1
            print(body_moves)
            print(head_moves)
            print("Corrupted Predictions: {}".format(cnter))
            ############################
            #See if Supervisor overrides
            trainingData = SupervisorCheck(cv2.resize(image,(299,299)), body_moves, head_moves, trainingData)

            print('loop took {:0.3f} seconds'.format(time.time()-last_time))
            last_time = time.time()

            #Store episodic memory to file. Sets of 100 so I don't need to preprocess.
            if len(trainingData) == 100:

                print(dataIndex)
                """
                for k in range(dataIndex*100):
                    cv2.imshow('test',trainingData[k][0])
                    print("predictions (body/head):")
                    print(trainingData[k][1])
                    print(trainingData[k][2])
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        cv2.destroyAllWindows()
                        break
                    input("pause")
                """
                shuffle(trainingData) # Just randomize it now.
                dataIndex += 1

                #Write to new datafile. Formatted to fit with current dataGenerator code.
                np.save("trainingData/episodeData/episodeData/data_{}".format(dataIndex), trainingData)

                # Restarts array for next batch
                trainingData = []


            if 'm' in keysPressed or 'M' in keysPressed:
                # My manual way to break the loop
                isSolved = True

            time.sleep(0.01)  # Needs to be there for the async hook and this synced loop to keep up with each other
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

        print("Number of frames in Episodic Memory: {}".format(dataIndex))

        #Stop the agent
        move_body([0,0,0,0,1])
        move_head([0,0,0,0,1])

        if dataIndex > EpisodicMemCap:


            for i, layer in enumerate(model.layers):
                print(i,layer.name)
            #input("pause: I need to update dis once")


            Layer_End = 19 # Only train the last dense layers that I personally added.
            for layer in model.layers[:Layer_End]:
                layer.trainable = False
            for layer in model.layers[Layer_End:]:
                layer.trainable = True

            ############################
            # Update the Policy
            EPOCHS = 4
            DTYPE = DATA_TYPE = 'episodeData'

            TrainLen = dataIndex - 5
            Indicies = np.array(range(1, dataIndex+1))
            shuffle(Indicies)
            TrainIndex = Indicies[:TrainLen]
            ValidIndex = Indicies[TrainLen:]

            params = {'WIDTH': WIDTH, 'HEIGHT': HEIGHT, 'DTYPE': DTYPE,
                        'DATA_TYPE': DATA_TYPE, 'isConcat': True,
                        'batch_size': 1, 'shuffle': True}

            training_generator = DataGenerator(Indicies, **params)
            validation_generator = DataGenerator(ValidIndex, **params)

            # Defining my callbacks:
            filepath="models/SupervisedReinforcement/best_weights.h5"
            checkpoint = ModelCheckpoint(filepath, monitor='val_dense_5_acc', verbose=1, save_best_only=True, mode='max')
            callbacks_list = [checkpoint]

            # compile the model (should be done *after* setting layers to non-trainable)
            model.compile(optimizer=SGD(lr=1e-4, momentum=0.9), loss='categorical_crossentropy', metrics=['acc'])
            #model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['acc'])

            # Train the model on the new data for a few epochs
            model.fit_generator(generator=training_generator,
                                validation_data=validation_generator,
                                callbacks=callbacks_list,
                                use_multiprocessing=True,
                                workers=6, epochs = EPOCHS, steps_per_epoch=dataIndex)

            #Reload the best weights from that update session
            model.load_weights("models/SupervisedReinforcement/best_weights.h5")

            # Save data rather than delete because it is like 10k data samples

            dataIndex_episode += 1
            shutil.move("trainingData/episodeData/episodeData","trainingData/episodeData/episodeDataArchived/E{}".format(dataIndex_episode))
            subprocess.call(['mkdir','trainingData/episodeData/episodeData'])
            pickle.dump(dataIndex_episode, open("trainingData/episodeData/dataIndex_episode.p", "wb"))

            dataIndex = 0

        else:
            # wait till I tell it to start testing again.
            testStart = False

            while not testStart:
                if 'n' in keysPressed or 'N' in keysPressed:
                    testStart = True


# Initialize the hooks!
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
