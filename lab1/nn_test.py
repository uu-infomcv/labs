'''Trains a simple deep NN on the Dogs vs Cats dataset.
Gets to 60.2% test accuracy after 30 epochs
(there is *a lot* of margin for parameter tuning).
1.33 seconds per epoch on a RTX 2080 Ti GPU.
'''

import numpy as np
import cv2, os
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, Activation, Dropout
from keras.layers.advanced_activations import ReLU
from keras.models import Model
np.random.seed(1337) # for reproducibility
from keras.optimizers import SGD

batch_size = 64
nb_classes = 2
nb_epoch = 30

def paths_list_from_directory(directory):

    # Loading full image directories

    return subdirs


def load_image(filename):


    # [1] Get the file category and make the conversion. If 'dog' assign it integer 1, if 'cat' assign it integer 0.

    # [2] Load the image in greyscale with opencv.

    height, width = image.shape[:2]
    crop_dim = None

    # [3] Find the dimension that is the smallest between the height and the width and assign it to the crop_drim variable.

    # [4] Crop the centre of the image based on the crop_dim dimension for both the height and width.

    # [5] Resize the image to 48 x 48 and divide it with 255.0 to normalise it to floating point format.

    return (image,label)


def DataGenerator(img_addrs, img_labels, batch_size, num_classes):
  while 1:
    # Ensure randomisation per epoch
    addrs_labels = list(zip(img_addrs,img_labels))
    shuffle(addrs_labels)
    img_addrs, img_labels = zip(*addrs_labels)

    X = []
    Y = []

    count = 0

    for i in range(len(img_addrs)):

      # [1] Call the load_images function and append the image in X.

      # [2] Create a one-hot encoding with np.eye and append the one-hot vector to Y.

      count += 1

      # [3] Commpare the count and batch_size (hint: modulo operation) and if so:
      #   - Use yield to return X,Y as numpy arrays with types 'float32' and 'uint8' respectively
      #   - delete X,Y
      #   - set X,Y to []
      #   - use python garbage collector

paths = paths_list_from_directory('./PetImages')

# Use train test split

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

X_train = X_train.reshape(lt, 2304)
X_val = X_val.reshape(lv, 2304)


inputs = Input(shape=(2304,))
x = inputs

i = 0

while i<3:

	x = Dense(254)(x)
	x = ReLU()(x)#Non-linearily
	x = Dropout(0.5)(x)
	i = i +1


predictions = Dense(nb_classes, activation='softmax')(x)
model = Model(input=inputs, output=predictions)

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# model.summary()



history = model.fit(X_train, Y_train,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_data=(X_val, Y_val))
#history = model.fit_generator(DataGenerator(...)...)
score = model.evaluate(X_val, Y_val, verbose=0)
#history = model.evaluate_generator(DataGenerator(...)...)

print('Test score:', score[0])
print('Test accuracy:', score[1])
