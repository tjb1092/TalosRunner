#from alexnet import alexnet, alexnet_color
from random import shuffle
#from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.models import Sequential
from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint, TensorBoard
import keras
import numpy as np
from dataGenerator import DataGenerator

#need to add a timestamp to see how long it takes!
WIDTH = 200
HEIGHT = 200
LR = 1e-4
EPOCHS_1 = 20
EPOCHS_2 = 40
DTYPE = 'body'
OPTIMIZER = 'momentum'
DATA_TYPE = 'rgb_200'
ARCH = "VGG16"
FILENUM = 232 # Get this automatically in the future


# I Should be able to define how many I want in my validation set.
#Then make the splits based off of that.
TrainIndex = np.array(range(1, FILENUM))
ValidIndex = np.array([FILENUM])


params = {'DTYPE': DTYPE, 'DATA_TYPE': DATA_TYPE, 'batch_size': 1, 'shuffle': True}

MODEL_NAME = 'pytalos_{}_{}_{}_{}_files_{}_epocs_{}_{}.h5'.format(DTYPE, ARCH, OPTIMIZER, FILENUM, EPOCHS_2, DATA_TYPE,LR)
model_path = "models/{}/{}".format(DTYPE,MODEL_NAME)

training_generator = DataGenerator(TrainIndex, **params)
validation_generator = DataGenerator(ValidIndex, **params)

# Define the model w/ Keras from their documentation on applications
#base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299,299,3))
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(200,200,3))

# Add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# Add a fully connected layer
x = Dense(1024, activation='relu')(x)

# and a logistic layer -- let's say we have 5 classes
predictions = Dense(5, activation='softmax')(x)

#This is the model we will train
model = Model(inputs=base_model.input, outputs = predictions)

#First: train only the top layers (which were randomly initialized)
#i.e. freeze all conv. InceptionV3 layers

for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['acc'])

# Train the model on the new data for a few epochs
model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                    workers=6, epochs = EPOCHS_1, steps_per_epoch=FILENUM)

print("Saving Model!")
model.save(model_path)
# At this point, the top layers are well trained and we can start fine-tuning conv. layers from InceptionV3.
# Freeze the bottom N layers and train the remaining top layers

# Let's visualize layer names and layer indicies to see how many layers we should freeze:

for i, layer in enumerate(base_model.layers):
    print(i,layer.name)

#Probs what I want to do anyways
#Choosing to train the top 2 inception blocks. i.e. freeze the first 249 layers and unfreeze the rest:
for layer in model.layers:
    layer.trainable = True
#for layer in model.layers[249:]:
    #layer.trainable = True
"""
# Apparently for now, freezing any layer fucks with the batch normalization... of course...
for layer in base_model.layers:
    layer.trainable = True
"""

# Recompile the model for these modifications to take effect
# Using SGD with a low learning rate.
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=LR, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model again (this time fine-tuning the top 2 inception blocks and the Dense layers)
model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                    workers=6, epochs = EPOCHS_2, steps_per_epoch=FILENUM)

print("Saving Model!")
model.save(model_path)


# tensorboard --logdir=~/store2/Code/TalosRunner/log
