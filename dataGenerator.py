import keras
import numpy as np
import cv2
# Modified from:
#https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html
#Had to conform with my pre-balance data approach I'm taking thus far.
class DataGenerator(keras.utils.Sequence):
    #Generates data for Keras
    def __init__(self, list_IDs, WIDTH, HEIGHT, DTYPE, DATA_TYPE, batch_size=1, shuffle=True):

        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT
        self.DTYPE = DTYPE
        self.DATA_TYPE = DATA_TYPE
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        # Denotes the number of batches per epoch
        # Might need to change it
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        # Generate one batch of data
        indicies = self.indicies[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        list_IDs_temp = int(self.list_IDs[indicies])
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y


    def on_epoch_end(self):
        # Updates indicies after each epoch
        self.indicies = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indicies)


    def __data_generation(self, ID):

        tmp = np.load('trainingData/{}/{}/data_{}.npy'.format(self.DATA_TYPE, self.DTYPE, ID))
        X= np.array([cv2.resize(i[0],(self.WIDTH,self.HEIGHT)) for i in tmp]).reshape(-1,self.WIDTH,self.HEIGHT,3)
        y = np.array([i[1] for i in tmp])

        return X, y
