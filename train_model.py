import numpy as np
from alexnet import alexnet

WIDTH = 100
HEIGHT = 100
LR = 1e-3
EPOCHS = 20
MODEL_NAME = 'pytalos-{}-{}-{}-epocs.model'.format(LR,'alexnet_body',EPOCHS)

model = alexnet(WIDTH,HEIGHT,LR)
"""
train_data = np.load('training_data_body_v2.npy')
train = train_data[:-20]
test = train_data[-20:]

X = np.array([i[0] for i in train]).reshape(-1,WIDTH,HEIGHT,1)
Y = np.array([i[1] for i in train])

testX = np.array([i[0] for i in test]).reshape(-1,WIDTH,HEIGHT,1)
testY = np.array([i[1] for i in test])

model.fit({'input': X}, {'targets', Y}, n_epoch=EPOCS, validation_set=({'input': testX}, {'targets': testY}), snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
"""

train_data = np.load('training_data_body_v2.npy')

train = train_data[:-20]
test = train_data[-20:]

X = np.array([i[0] for i in train]).reshape(-1,WIDTH,HEIGHT,1)
Y = np.array([i[1] for i in train])

test_x = np.array([i[0] for i in test]).reshape(-1,WIDTH,HEIGHT,1)
test_y = np.array([i[1] for i in test])

model.fit({'input': X}, {'targets': Y}, n_epoch=EPOCHS, validation_set=({'input': test_x}, {'targets': test_y}),
    snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

# tensorboard --logdir=~/store/Code/TalosRunner/log
model.save(MODEL_NAME)
