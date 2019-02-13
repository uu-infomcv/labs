'''Trains a CNN on the Dogs vs Cats dataset.
Gets to 80.17% test accuracy after 30 epochs
(there is *a lot* of margin for parameter tuning).
1.102 seconds per epoch on a RTX 2018 Ti GPU.
'''

import numpy as np
import cv2, os
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, Activation, Dropout, MaxPool2D, AveragePooling2D, Conv2D, GlobalAvgPool2D, Flatten, BatchNormalization
from keras.layers.advanced_activations import ReLU
from keras.models import Model
np.random.seed(1337) # for reproducibility
from keras.optimizers import SGD
from layers import ShakeShake
from lr_callbacks import Step_decay, InternalStateHistory

batch_size = 64
nb_classes = 2
nb_epoch = 30
bn_axis = 3

def paths_list_from_directory(directory):

    # Code goes here


def load_image(filename):

    # Code goes here

def DataGenerator(img_addrs, img_labels, batch_size, num_classes):

    # Code goes here



paths = paths_list_from_directory('./PetImages')

# train_test_split

X_train = []
Y_train = []

for p in train:
    tpl = load_image(p)
    X_train.append(tpl[0])
    Y_train.append(np.eye(nb_classes)[tpl[1]])

X_val = []
Y_val = []

for p in val:
    tpl = load_image(p)
    X_val.append(tpl[0])
    Y_val.append(np.eye(nb_classes)[tpl[1]])

lt = len(X_train)
lv = len(X_val)

X_train = np.asarray(X_train)
Y_train = np.asarray(Y_train)
X_val = np.asarray(X_val)
Y_val = np.asarray(Y_val)


# Should be changed to the right size
inputs = Input(shape=(2304,))

x = inputs
x = Conv2D(filters=32, kernel_size=(3, 3), name='cnn_lab_conv1')(x)
x = BatchNormalization(axis=bn_axis)(x)
x = Activation('relu')(x)

# Add MaxPool here

# Second block
x = Conv2D(filters=64, kernel_size=(3, 3), name='cnn_lab_conv2')(x)
x = BatchNormalization(axis=bn_axis)(x)
x = Activation('relu')(x)

# Add MaxPool here

# Third block
x = Conv2D(filters=128, kernel_size=(3, 3), name='cnn_lab_conv3')(x)
x = BatchNormalization(axis=bn_axis)(x)
x = Activation('relu')(x)

# x = ShakeShake()(x)

x = Flatten()(x)

predictions = Dense(nb_classes, activation='softmax')(x)
model = Model(input=inputs, output=predictions)

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#model.summary()

# Declare the Step_decay and InternalStateHistory Callbacks here
# Add them to a list

history = model.fit(X_train, Y_train,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_data=(X_val, Y_val))
#history = model.fit_generator(DataGenerator(...)...)
score = model.evaluate(X_val, Y_val, verbose=0)
#history = model.evaluate_generator(DataGenerator(...)...)

print('Test score:', score[0])
print('Test accuracy:', score[1])
