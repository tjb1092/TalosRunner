import time
import numpy as np
import cv2
from directInputs import forwards_body, backwards_body, left_body, right_body, stop_body, up_head, down_head, left_head, right_head, stop_head
import os
import pickle
from keras.models import load_model


def countDown(cnt):
    #Delay before recording takes place
    for i in list(range(cnt))[::-1]:
        print(i+1)
        time.sleep(1)

def viewData(fname):
    training_data = np.load(fname)

    """
    cv2.imshow('test',D)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
    input("blended Pause")
    """
    #Look at the data!
    cnter = 0

    for i in range(len(training_data)):
        A = training_data[i][1]

        if np.all(np.asarray(training_data[i][1]) == 0) or np.all(np.asarray(training_data[i][2]) == 0):
            cnter += 1
            D = training_data[i][0]
            """
            cv2.imshow('test', D )

            print("Predicitions (body/head)")
            print(training_data[i][1])
            print(training_data[i][2])
            #input("pause")
            """
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

    print(cnter)
    return cnter

def setPickleIndex(pickle_fname,val):
    #update the stored index. Only for head right now.
    pickle.dump(val, open(pickle_fname, "wb"))

def save_npy_file(Dtype, dataIndex, training_data, pickle_fname):
    # Assume the dataIndex corresponds with the current data file.
    dataIndex += 1

    #Write to new datafile
    np.save("collectedData/"+Dtype+"/training_data_"+Dtype+"_"+str(dataIndex), training_data)
    pickle.dump(dataIndex, open(pickle_fname, "wb"))

    return dataIndex

def open_dataIndex(pickle_fname):
    if os.path.isfile(pickle_fname):
        dataIndex = pickle.load(open(pickle_fname,"rb"))
    else:
        dataIndex = 0
    return dataIndex

def move_body(moves):
    if moves == [1,0,0,0,0]:
        left_body()
    elif moves == [0,1,0,0,0]:
        right_body()
    elif moves == [0,0,1,0,0]:
        forwards_body()
    elif moves == [0,0,0,1,0]:
        backwards_body()
    elif moves == [0,0,0,0,1]:
        stop_body()

def move_head(moves):
    if moves == [1,0,0,0,0]:
        left_head()
    elif moves == [0,1,0,0,0]:
        right_head()
    elif moves == [0,0,1,0,0]:
        up_head()
    elif moves == [0,0,0,1,0]:
        down_head()
    elif moves == [0,0,0,0,1]:
        stop_head()

def get_Keras_model(DTYPE, WIDTH, HEIGHT, EPOCHS_1, OPTIMIZER, FILENUM):
    # Load model

    LR = 1e-4
    DATA_TYPE = "Unbalanced_rgb_299"
    ARCH = "VGG16"
    MODEL_NAME = 'pytalos_{}_{}_{}_{}_files_{}_epocs_{}_{}.h5'.format(DTYPE, ARCH, OPTIMIZER, FILENUM, EPOCHS_1, DATA_TYPE,LR)
    model_path = "models/{}/{}".format(DTYPE,MODEL_NAME)

    print(model_path)
    model = load_model(model_path)
    model.load_weights("models/{}/best_weights_{}".format(DTYPE,MODEL_NAME))
    return model
