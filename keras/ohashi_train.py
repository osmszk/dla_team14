# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 18:44:23 2018

@author: s-ohashi
"""

from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras import optimizers

import szk_input
import sys
import numpy as np
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# -f floydhubで実行する場合は、`-f`オプションをつける。
on_floydhub = True if "-f" in sys.argv else False
# -m ももくろで実行する場合は、`-m`オプションをつける。
on_momokuro = True if "-m" in sys.argv else False

batch_size = 128
nb_classes = 5 if on_momokuro else 3
nb_epoch = 50
data_augmentation = False

output_path = '/output/model.h5' if on_floydhub else './model.h5'
graph_path = '/output/result.png' if on_floydhub else './train_result.png'
train_data_path = '/data/train/data.txt' if on_floydhub else './data/train/data.txt'
test_data_path = '/data/test/data.txt' if on_floydhub else './data/test/data.txt'

img_rows, img_cols = 112, 112
img_channels = 3

print('train_data_path:',train_data_path)
print('test_data_path:',test_data_path)
(X_train, y_train)= szk_input.read_data(train_data_path)
(X_test, y_test)= szk_input.read_data(test_data_path)

print('X_train shape:', X_train.shape)
print('y_train shape:', y_train.shape)
print('input shape:',X_train.shape[1:])
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
print(X_train.shape[1:])

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same', input_shape=X_train.shape[1:] ,name='conv1'))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3),padding='same',name='conv2'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same',name='conv3'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3),padding='same', name='conv4'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

pre_trained_filename='./pretrain_model/pretrain_weights.87-0.00-0.00.h5'
model.load_weights(pre_trained_filename, by_name=True)



model.add(Flatten())
model.add(Dense(512,name='dense1'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(nb_classes,name='dense2'))
model.add(Activation('softmax'))

#pre_trained_filename='./pretrain_model/pretrain_weights.87-0.00-0.00.h5'
#model.load_weights(pre_trained_filename, by_name=True)

model.summary()

opt_adam = optimizers.Adam(lr=0.00001)
model.compile(loss='categorical_crossentropy',
              optimizer=opt_adam,
              metrics=['accuracy'])

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
csv_logger = CSVLogger('log.csv', append=True, separator=';')

fpath = './train_model/weights.{epoch:02d}-{loss:.2f}-{acc:.2f}-{val_loss:.2f}-{val_acc:.2f}.h5'
cp_cb = ModelCheckpoint(fpath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

if not data_augmentation:
    print('Not using data augmentation.')
    fit = model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_data=(X_test, Y_test),
              shuffle=True,
              callbacks=[csv_logger, cp_cb, stopping])
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(X_train)

    # Fit the model on the batches generated by datagen.flow().
    fit = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                        epochs=nb_epoch,
                        validation_data=(X_test, Y_test),
                        callbacks=[csv_logger])

model.save(output_path)


# ----------------------------------------------
# Some plots
# ----------------------------------------------

fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10,4))


# loss
def plot_history_loss(fit):
    # Plot the loss in the history
    axL.plot(fit.history['loss'],label="loss for training")
    axL.plot(fit.history['val_loss'],label="loss for validation")
    axL.set_title('model loss')
    axL.set_xlabel('epoch')
    axL.set_ylabel('loss')
    axL.legend(loc='upper right')

# acc
def plot_history_acc(fit):
    # Plot the loss in the history
    axR.plot(fit.history['acc'],label="accuracy for training")
    axR.plot(fit.history['val_acc'],label="accuracy for validation")
    axR.set_title('model accuracy')
    axR.set_xlabel('epoch')
    axR.set_ylabel('accuracy')
    axR.legend(loc='upper right')

plot_history_loss(fit)
plot_history_acc(fit)
fig.savefig(graph_path)
plt.close()

# @MBP touchbar (3.3GHz Core i7, memory 16GB)
# 1 epoch -> ETA 3300 (55min.?)

# @Google Cloud Platform(vCPU x 8, memory 300GB, Ubuntu 16.04)
# 1 epoch -> ETA 31:30
