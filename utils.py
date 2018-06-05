import time
import numpy as np
import cv2


def countDown(cnt):
    #Delay before recording takes place
    for i in list(range(cnt))[::-1]:
        print(i+1)
        time.sleep(1)

def viewData(fname):
        training_data = np.load(fname)

        #Look at the data!

        for data in training_data:
            cv2.imshow('test', data[0])
            print(data[1])
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
