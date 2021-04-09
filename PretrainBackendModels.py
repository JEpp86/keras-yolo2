# following https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Activation, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import SGD
from keras.layers import Dense
from keras.utils import np_utils

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

from keras.applications.mobilenet import MobileNet
from keras.models import Model
from keras.layers import Input
from keras import backend as K
from keras import optimizers

from backend_models import full_darknet, tiny_darknet, squeezenet, tiniest_yolo
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)

import numpy as np
import argparse
import cv2
import os, os.path
import random
import argparse

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import matplotlib
import matplotlib.pyplot as plt
#matplotlib.use('TkAgg')

argparser = argparse.ArgumentParser()
argparser.add_argument('-m','--mobile', action='store_true', help='MobileNet backend')
argparser.add_argument('-d','--darknet', action='store_true', help='DarkNet19 backend')
argparser.add_argument('-t','--tinydark', action='store_true', help='Tiny DarkNet backend')
argparser.add_argument('-s','--squeeze', action='store_true', help='SqueezeNet backend')
argparser.add_argument('-ts','--tiniest', action='store_true', help='Tiniest backend')
args = argparser.parse_args()


# ---- load data ----
# path to training images
dock_data = '../data/Dock/'
not_data = '../data/NotDock/'

# images to be resized to (image_dim) x (image_dim)
image_dim = 224
val_split = 0.2

#select model backend
print('Selecting model')
input_image = Input(shape=(image_dim, image_dim, 3))

if args.mobile:
    base_model = MobileNet(input_shape=(image_dim,image_dim,3), include_top=False)
    x = base_model(input_image)
    backend = 'MobileNet'
elif args.darknet:
    x = full_darknet(input_image)
    backend = 'DN19'
elif args.tinydark:
    x = tiny_darknet(input_image)
    backend = 'TinyDN'
elif args.squeeze:
    x = squeezenet(input_image)
    backend = 'SqueezeNet'
elif args.tiniest:
    x = tiniest_yolo(input_image)
    backend = 'Tiniest'
else:
    print('No backend selected, exiting')
    exit()

x_train = []
y_train = []
x_valid = []
y_valid = []

train_files = []
valid_files = []

print("Getting Training Files")

for filename in os.listdir(dock_data):
    train_files.append(''.join([dock_data, filename]))
for filename in os.listdir(not_data):
    train_files.append(''.join([not_data, filename]))

print('Prepareing validation split')
random.shuffle(train_files)
split = (int)(len(train_files)*val_split)
valid_files = train_files[:split]
train_files = train_files[split:]

print('preparing training data')
for file in train_files:
    image = cv2.imread(file)
    x_train.append(cv2.resize(image, (image_dim, image_dim)))
    label = 1 if file.split("/")[1] == 'Dock' else 0
    y_train.append(label)
print('preparing validation data')
for file in valid_files:
    image = cv2.imread(file)
    x_valid.append(cv2.resize(image, (image_dim, image_dim)))
    label = 1 if file.split("/")[1] == 'Dock' else 0
    y_valid.append(label)



# convert data to NumPy array of floats
x_train = np.array(x_train, np.float32)
x_valid = np.array(x_valid, np.float32)

print(x_train.shape)

# ---- define data generator ----
datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=None,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='reflect') 

datagen.fit(x_train)
datagen_valid = ImageDataGenerator()
datagen_valid.fit(x_valid)


print('preparing model')

# add detection layers
det = Flatten()(x)
d1 = Dense(64, activation='relu')(det)
d1 = Dropout(0.5)(d1)
predictions = Dense(1, activation='sigmoid')(d1)
model = Model(inputs=input_image, outputs=predictions) # final model

opt  = optimizers.SGD(lr=0.0001)

model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])


model.summary()


# ---- train the model ----
batch_size = 16
num_epochs = 10

history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                    steps_per_epoch=len(x_train) / batch_size, epochs=num_epochs,
                    validation_data=datagen.flow(x_valid, y_valid, batch_size=batch_size),
                    validation_steps = len(x_valid) / batch_size)



# ---- save the model and the weights ----
model.save(backend+'_backend.h5')
model.save_weights(backend+'_backend_weights.h5')
print('model saved')



# ---- display history ----
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('train_test_accuracy_mobilenet_dock.png')
plt.clf() # clear figure
# summarize history for loss (binary cross-entropy)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('binary cross-entropy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('train_test_loss_mobilenet_dock.png')
plt.clf()
