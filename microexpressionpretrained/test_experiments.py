import os
import cv2
import numpy
import imageio
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.convolutional import Convolution3D, MaxPooling3D
from keras.optimizers import SGD, RMSprop
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils, generic_utils
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras import Input
import tensorflow as tf
import sys

# MicroExpSTCNN Model
model = Sequential()
model.add(Input(shape=(1, 64, 64, 18)))  # 250x250 RGB images
model.add(Convolution3D(32, (3, 3, 15), input_shape=(1, 64, 64, 18), activation='relu',
data_format='channels_first'))

model.add(MaxPooling3D(pool_size=(3, 3, 3)))
model.add(Dropout(0.5))
model.add(Flatten(data_format='channels_first'))
model.add(tf.keras.layers.Reshape([16000]))
model.add(Dense(128, kernel_initializer='random_uniform', activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, kernel_initializer='random_uniform'))
model.add(Activation('softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'SGD', metrics = ['accuracy'])
model.summary()

# Load pre-trained weights

model.load_weights('weights-improvement-26-0.69.hdf5')


# Load validation set from numpy array

validation_images = numpy.load('microexpstcnn_val_images.npy')
validation_labels = numpy.load('microexpstcnn_val_labels.npy')



# Finding Confusion Matrix using pretrained weights

predictions = model.predict(validation_images)
predictions_labels = numpy.argmax(predictions, axis=1)
validation_labels = numpy.argmax(validation_labels, axis=1)
cfm = confusion_matrix(validation_labels, predictions_labels)
print (cfm)

