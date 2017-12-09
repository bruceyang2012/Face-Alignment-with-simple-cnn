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
fig.savefig('C:/DeepLearning/faceID/faceAlign/results/predicted.png')   

FLOOKUP='IdLookupTable.csv'

columns=(
    'left_eye_center_x', 'left_eye_center_y',
    'right_eye_center_x', 'right_eye_center_y',
    'nose_tip_x', 'nose_tip_y',
    'mouth_left_corner_x', 'mouth_left_corner_y',
    'mouth_right_corner_x', 'mouth_right_corner_y',
    'mouth_center_top_lip_x', 'mouth_center_top_lip_y',
    'mouth_center_bottom_lip_x', 'mouth_center_bottom_lip_y',
    'left_eye_inner_corner_x', 'left_eye_inner_corner_y',
    'right_eye_inner_corner_x', 'right_eye_inner_corner_y',
    'left_eye_outer_corner_x', 'left_eye_outer_corner_y',
    'right_eye_outer_corner_x', 'right_eye_outer_corner_y',
    'left_eyebrow_inner_end_x', 'left_eyebrow_inner_end_y',
    'right_eyebrow_inner_end_x', 'right_eyebrow_inner_end_y',
    'left_eyebrow_outer_end_x', 'left_eyebrow_outer_end_y',
    'right_eyebrow_outer_end_x', 'right_eyebrow_outer_end_y',
)

def create_submission(predicted_labels, columns):
    predicted_labels = (predicted_labels + 1) * 48.0
    predicted_labels = predicted_labels.clip(0, 96)
    df = pd.DataFrame(predicted_labels, columns=columns)
    lookup_table = pd.read_csv(FLOOKUP)
    values = []

    for index, row in lookup_table.iterrows():
        values.append((
            row['RowId'],
            df.ix[row.ImageId - 1][row.FeatureName],
        ))

    submission = pd.DataFrame(values, columns=('RowId', 'Location'))
    submission.to_csv("keypoints_pred.csv", index=False)
    
create_submission(y_test,columns)

#write out submission
#submission = pd.DataFrame(index=pd.RangeIndex(start=1, stop=27125, step=1), columns=['Location'])
#y_test = y_test.reshape(-1,1)
#print(y_test.shape)
#submission['Location'] = y_test.reshape(-1,1)
#submission.index.name = 'RowId'
#submission.to_csv('keypoints_pred.csv', index=True, header=True)