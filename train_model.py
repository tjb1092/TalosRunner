import numpy as np
from alexnet import alexnet

WIDTH = 100
HEIGHT = 100
LR = 1e-4
EPOCHS = 100
Dtype = 'body'
OPTIMIZER = 'momentum'
MODEL_NAME = 'pytalos-{}-{}-{}-epocs-{}.model'.format(LR,'alexnet_{}_6kSamples'.format(Dtype),EPOCHS, OPTIMIZER)

model = alexnet(WIDTH,HEIGHT,LR, MODEL_NAME)

train_data = np.load('preprocessedTrainingData/{}/training_data_{}.npy'.format(Dtype, Dtype))
#Automate this to 10% of data bruh
train = train_data[:-600]
test = train_data[-600:]

X = np.array([i[0] for i in train]).reshape(-1,WIDTH,HEIGHT,1)
Y = np.array([i[1] for i in train])

test_x = np.array([i[0] for i in test]).reshape(-1,WIDTH,HEIGHT,1)
test_y = np.array([i[1] for i in test])

model.fit({'input': X}, {'targets': Y}, n_epoch=EPOCHS, validation_set=({'input': test_x}, {'targets': test_y}),
    snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

# tensorboard --logdir=~/store/Code/TalosRunner/log
model.save("models/{}/{}".format(Dtype,MODEL_NAME)) #change this argument to make it store in a file path
