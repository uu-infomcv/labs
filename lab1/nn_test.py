'''Trains a simple deep NN on the Dogs vs Cats dataset.
Gets to 60.2% test accuracy after 30 epochs
(there is *a lot* of margin for parameter tuning).
1.33 seconds per epoch on a RTX 2018 Ti GPU.
'''

import numpy as np
import cv2, os
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, Activation, Dropout
from keras.layers.advanced_activations import ReLU
from keras.models import Model
np.random.seed(1337) # for reproducibility

batch_size = 64
nb_classes = 2
nb_epoch = 30

def paths_list_from_directory(directory):

    subdirs = [os.path.join(d, x)for d, dirs, files in os.walk(directory) for x in files]
    return subdirs


def load_image(filename):
    label = filename.split('/')[-2]
    if (label == 'Cat'):
        label = 0
    else:
        label = 1
    image = cv2.imread(filename, 0)
    height, width = image.shape[:2]
    crop_dim = None

    if height>width:
        crop_dim = width
    else:
        crop_dim = height

    image = image[height//2-crop_dim//2:height//2+crop_dim//2, width//2-crop_dim//2:width//2+crop_dim//2]

    image = cv2.resize(image, (48, 48))

    image = np.true_divide(image,255.0)

    return (image, label)

paths = paths_list_from_directory('./PetImages')

train, val = train_test_split(paths, test_size=0.2, random_state=42)

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
model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()



history = model.fit(X_train, Y_train,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_data=(X_val, Y_val))
score = model.evaluate(X_val, Y_val, verbose=0)

print('Test score:', score[0])
print('Test accuracy:', score[1])
