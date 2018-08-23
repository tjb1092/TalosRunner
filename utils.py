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

    #def blendImages(*images): return
    #D = np.round(0.2*training_data[0][0]+0.2*training_data[1][0]+0.2*training_data[2][0]+0.2*training_data[3][0]+0.2*training_data[4][0]).astype('uint8')

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

        #D = np.round(0.2*training_data[i][0]+0.2*training_data[i-1][0]+0.2*training_data[i-2][0]+0.2*training_data[i-3][0]+0.2*training_data[i-4][0]).astype('uint8')
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
    """
    for data in training_data:
        image = data[0]
        #image = cv2.resize(image,(1000,1000)) #That hurt my eyes lol
        cv2.imshow('test', image)
        print(data[1])

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        input("pause")
    """
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

    #This can be stored as something like a pickle config that was used for passing between training and testing.
    LR = 1e-4
    #OPTIMIZER = 'Adam'
    #DATA_TYPE = "rgb_{}".format(WIDTH)
    DATA_TYPE = "Unbalanced_rgb_299"
    ARCH = "VGG16"
    MODEL_NAME = 'pytalos_{}_{}_{}_{}_files_{}_epocs_{}_{}.h5'.format(DTYPE, ARCH, OPTIMIZER, FILENUM, EPOCHS_1, DATA_TYPE,LR)
    model_path = "models/{}/{}".format(DTYPE,MODEL_NAME)

    print(model_path)
    model = load_model(model_path)
    model.load_weights("models/{}/best_weights_{}".format(DTYPE,MODEL_NAME))
    return model


"""
# Code for the mouse if I ever need it again...

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

# This was such an obscure way to to this. X11 doesn't force the mouse in the game window
# This device object does work, so that's nice that it works
device = uinput.Device([
    uinput.BTN_LEFT,
    uinput.BTN_RIGHT,
    uinput.REL_X,
    uinput.REL_Y,
    ])

choice = 5 #start not moving head

"""
