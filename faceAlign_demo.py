# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from keypoints_kernel import load,load2d
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Activation, Dropout, MaxPooling2D, Flatten, GlobalMaxPooling2D, BatchNormalization
from keras.optimizers import SGD, Adam
import matplotlib.pyplot as plt
from keras.preprocessing import image
import time

def cnn():
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
    return model

model = cnn()
model.load_weights(filepath='weights.75-0.74206.hdf5')

# 加载图像
image_path='lena.jpg'
img = image.load_img(image_path,grayscale=True, target_size=(96, 96))

# 图像预处理
x = image.img_to_array(img)
x /= 255
x = np.expand_dims(x, axis=0)

#测试一张人脸的时间
t1 = time.time()
pred = model.predict(x)
t2 = time.time()
t = t2 - t1
print('{:0.5f}s'.format(t))

points = pred[0]*48+48
points = points.clip(0, 96)
print(points)

fig = plt.figure()  
axis = fig.add_subplot(111)  
#画人脸
axis.imshow(img, cmap='gray')
#画散点图  
axis.scatter(points[0::2],points[1::2],c = 'r',marker = 'o')  
#显示所画的图  
plt.show()  

