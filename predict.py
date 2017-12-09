# -*- coding: utf-8 -*-
import pandas as pd
from train import load,load2d
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Activation, Dropout, MaxPooling2D, Flatten, GlobalMaxPooling2D, BatchNormalization
from keras.optimizers import SGD, Adam
import matplotlib.pyplot as plt

def plot_sample(x, y, axis):
    img = x.reshape(96, 96)
    axis.imshow(img, cmap='gray')
    axis.scatter(y[0::2]*48+48, y[1::2]*48+48, marker='x', s=10)

model = Sequential()

model.add(BatchNormalization(input_shape=(96,96,1)))

model.add(Conv2D(32, kernel_size=(3, 3),padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128,kernel_size=(3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, kernel_size=(3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(GlobalMaxPooling2D())

model.add(Dense(512)) #512
model.add(Activation('relu'))

model.add(Dense(30))

model.load_weights(filepath='weights.75-0.74206.hdf5')
#predict test images
X_test, _ = load2d(test=True)
y_test = model.predict(X_test)

fig = plt.figure(figsize=(6,6))
for i in range(25):
    axis = fig.add_subplot(5,5,i+1,xticks=[],yticks=[])
    plot_sample(X_test[i], y_test[i], axis)
plt.show()
fig.savefig('predicted.png')   
