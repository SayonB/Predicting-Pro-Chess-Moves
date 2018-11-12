import os
from time import time

import h5py
import keras
import numpy as np
from keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import (Activation, Conv2D, Dense, Dropout, Flatten,
                          LeakyReLU, MaxPooling2D)
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, load_model
from keras.optimizers import SGD, Adadelta, Adam, RMSprop

import pandas as pd
from inputboard import convert_position_prediction, make_clean_board, move_from


def generator(inputs, outputs, batch_size, training_examples=0, validation_examples=0):
    'Generator that chunks the data, and processes.'
    training_steps = training_examples//batch_size
    validation_steps = validation_examples//batch_size
    start_val = inputs.shape[0] - validation_examples

    if validation_examples == 0:
        while 1:
            for i in range(training_steps):
                batch_x = inputs[i*batch_size:(i+1)*batch_size]
                batch_y = outputs[i*batch_size:(i+1)*batch_size]

                batch_x = batch_x.reshape(batch_x.shape[0], 8, 8, 12)
                yield batch_x, batch_y
    else:
        while 1:
            for i in range(validation_steps):
                batch_x = inputs[(i*batch_size)+start_val:((i+1)*batch_size)+start_val]
                batch_y = outputs[(i*batch_size)+start_val:((i+1)*batch_size)+start_val]

                batch_x = batch_x.reshape(batch_x.shape[0], 8, 8, 12)
                yield batch_x, batch_y

# ----------The network----------
stime = time()
batch_size = 128
num_classes = 64
epochs = 30

stime = time()

# input image dimensions
img_rows, img_cols = 8, 8
input_shape = (img_rows, img_cols, 12)

model = Sequential()
model.add(Conv2D(128, kernel_size=(2, 2),
                 input_shape=input_shape))
model.add(Activation('relu'))

model.add(Conv2D(128, kernel_size=(2, 2)))
model.add(Activation('relu'))

model.add(Flatten())

model.add(Dense(1024))
model.add(Activation('relu'))

model.add(Dense(1024))
model.add(Activation('relu'))

model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=Adam(),
              metrics=['accuracy'])

tensorboard = TensorBoard(log_dir='/tmp/keras_logs/ChessAI/moved_from')
filepath = 'moved_from_model.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1,
                             save_best_only=True, mode='max')
callbacks = [tensorboard, checkpoint]

h5f = h5py.File('FF2048_Data.h5', 'r')
X = h5f['input_position']
Y = h5f['moved_from']

num_training_examples = 6000000
num_val_examples = 1200000

training_batch_gen = generator(X, Y, batch_size, training_examples=num_training_examples)
validation_batch_gen = generator(X, Y, batch_size, validation_examples=num_val_examples)

if __name__ == '__main__':
    model.fit_generator(generator=training_batch_gen,
                        steps_per_epoch=(num_training_examples // batch_size),
                        epochs=epochs,
                        shuffle=True,
                        verbose=1,
                        callbacks=callbacks,
                        validation_data=validation_batch_gen,
                        validation_steps=(num_val_examples // batch_size),
                        workers=16,
                        max_queue_size=32)

    score = model.evaluate(X_val, Y_val, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print('Time taken:', time() - stime)


def test():
    model = load_model(r'moved_from_model.h5')

    input_board_one_hot = make_clean_board()
    input_board_one_hot = np.array(input_board_one_hot).reshape(1, 8, 8, 12)
    for i in range(1000):
        t1 = time()
        prediction = model.predict(input_board_one_hot)
        print(time() - t1)

    move = sorted(prediction[0])

    output_board_one_hot = []
    for i in prediction[0]:
        if i == move[-1]:
            output_board_one_hot.append(1)
        elif i == move[-2]:
            output_board_one_hot.append(2)
        elif i == move[-3]:
            output_board_one_hot.append(3)
        elif i == move[-4]:
            output_board_one_hot.append(4)
        elif i == move[-5]:
            output_board_one_hot.append(5)
        elif i == move[-6]:
            output_board_one_hot.append(6)
        elif i == move[-7]:
            output_board_one_hot.append(7)
        else:
            output_board_one_hot.append(0)

    print(move_from(output_board_one_hot))
