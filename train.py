import os
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Activation, Dropout, MaxPooling2D, Flatten, GlobalMaxPooling2D, BatchNormalization
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator

FTRAIN = 'C:/DeepLearning/faceID/faceAlign/data/training/training.csv'
FTEST = 'C:/DeepLearning/faceID/faceAlign/data/test/test.csv'

def load(test=False, cols=None):
    
    fname = FTEST if test else FTRAIN
    df = pd.read_csv(os.path.expanduser(fname))
    
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))
    
    if cols:
        df = df[list(cols)+['Image']]
    
    print (df.count())
    df = df.dropna()
    
    X = np.vstack(df['Image'].values)/255
    X = X.astype(np.float32)
    
    if not test:
        y = df[df.columns[:-1]].values
        y = (y-48)/48
        X, y = shuffle(X, y, random_state=42)
        y = y.astype(np.float32)
    else:
        y = None
    
    return X, y
    
def load2d(test=False, cols=None):
    
    X, y = load(test, cols)
    X = X.reshape(-1,96,96,1)
    
    return X, y
    
def plot_sample(x, y, axis):
    
    img = x.reshape(96, 96)
    axis.imshow(img, cmap='gray')
    axis.scatter(y[0::2]*48+48, y[1::2]*48+48, marker='x', s=10)

if __name__ == "__main__":
    
    X, y = load2d(test=False)
    
    print ("X.shape", X.shape)
    print ("y.shape", y.shape)        
    
    x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=10)
    print ("x_train.shape", x_train.shape)
    print ("y_train.shape", y_train.shape)
    datagen = ImageDataGenerator()
    
#    #neural net
#    model = Sequential()
#    model.add(Conv2D(32, 3, 3, input_shape=(96, 96, 1)))    
#    model.add(Activation('relu'))
#    model.add(MaxPooling2D(pool_size=(2,2)))
#    
#    model.add(Conv2D(64, 2, 2))
#    model.add(Activation('relu'))
#    model.add(MaxPooling2D(pool_size=(2,2)))    
#
#    model.add(Conv2D(128, 2, 2))
#    model.add(Activation('relu'))
#    model.add(MaxPooling2D(pool_size=(2,2)))    
#    
#    model.add(Flatten())
#    model.add(Dense(500))
#    model.add(Activation('relu'))
#    model.add(Dense(500))
#    model.add(Activation('relu'))
#    model.add(Dense(30))    
                
    # my model(bn)
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
    
#    sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])
    
    check = ModelCheckpoint("weights.{epoch:02d}-{val_acc:.5f}.hdf5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='auto')
    early = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=1, mode='max')
    hist = model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),steps_per_epoch=len(x_train)/32,epochs=100,callbacks=[check,early],validation_data=(x_test,y_test))

    #display loss 
    f=plt.figure()
    plt.plot(hist.history['loss'], linewidth=3, label='train')
    plt.plot(hist.history['val_loss'], linewidth=3, label='valid')
    plt.grid()
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.yscale('log')
    plt.show()
    f.savefig('loss.png')
    
    #predict test images
    X_test, _ = load2d(test=True)
    y_test = model.predict(X_test)
    
    fig = plt.figure(figsize=(6,6))
    for i in range(16):
        axis = fig.add_subplot(4,4,i+1,xticks=[],yticks=[])
        plot_sample(X_test[i], y_test[i], axis)
    plt.show()
    fig.savefig('C:/DeepLearning/faceID/faceAlign/results/predicted.png')        
    
    #write out submission
#    submission = pd.DataFrame(index=pd.RangeIndex(start=1, stop=27124, step=1), columns=['Location'])
#    submission['Location'] = y_test.reshape(-1,1)
#    submission.index.name = 'RowId'
#    submission.to_csv('keypoints_pred.csv', index=True, header=True)
    
    
    
    
    