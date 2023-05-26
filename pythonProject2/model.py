import cv2
import numpy as np
import os
from PIL import Image


data = []
label = []

for j in range (1,6):
    for i in range (1,21):
        filename = 'dataset/User' + str(j) + '.' + str(i) +'.jpg'
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(src = img, dsize = (100,100))
        img = np.array(img)
        data.append(img)
        label.append(j-1)
data1 = np.array(data)
label = np.array(label)
data1 = data1.reshape((100,100,100,1))
X_train = data1/255

from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()
trainY = lb.fit_transform(label)
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout,Activation,Input,concatenate
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD

model = Sequential()
input_shape = (100,100,1)
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape,padding = 'same'))
model.add(Activation("relu"))
model.add(Conv2D(32, (3, 3), padding='same'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size= (2, 2)))
model.add(Conv2D(64,(3,3), padding ='same'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation("relu"))
model.add(Dense(5))
model.add(Activation('softmax'))
model.summary()
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

print("start training")
model.fit(X_train,trainY,batch_size =5,epochs=10)
model.save("khuonmat.h5")
