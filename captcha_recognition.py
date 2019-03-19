import cv2 as cv 
import os
import numpy as np 
import string
from keras.layers import Input, Dense, Conv2D, Flatten, MaxPooling2D, Dropout, GlobalAveragePooling2D
from keras.models import Model
from keras.applications import VGG16

symbols = string.ascii_lowercase + "0123456789"
num_symbols = len(symbols)
img_shape = (50, 200, 3)
vgg = True

def preprocess_data():
    n_samples = len(os.listdir('./samples/'))
    X = np.zeros((n_samples, 50, 200, 3)) #1070*50*200
    y = np.zeros((5, n_samples, num_symbols)) #5*1070*36

    for i, pic in enumerate(os.listdir('./samples/')):
        # Read image as grayscale
        img = cv.imread(os.path.join('./samples/', pic))
        print(img.shape)
        pic_target = pic[:-4]
        if len(pic_target) < 6:
            # Scale and reshape image
            img = img / 255.0
            img = np.reshape(img, img_shape)
            # Define targets and code them using OneHotEncoding
            targs = np.zeros((5, num_symbols))
            for j, l in enumerate(pic_target):
                ind = symbols.find(l)
                targs[j, ind] = 1
            X[i] = img
            y[:, i] = targs
    
    # Return final data
    return X, y

X, y = preprocess_data()
X_train, y_train = X[:970], y[:, :970]
X_test, y_test = X[970:],y[:, 970:]


if not vgg:
    img = Input(shape = img_shape)
    conv1 = Conv2D(64, (3, 3), padding='same', activation='relu')(img)
    mp1 = MaxPooling2D(pool_size=(2,2))(conv1)
    conv2 = Conv2D(32, (3, 3), padding='same', activation='relu')(mp1)
    mp2 = MaxPooling2D(pool_size=(2,2))(conv2)
    conv3 = Conv2D(16, (3, 3), padding='same', activation='relu')(mp2)
    mp3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    top_model = Flatten()(mp3)
else:
    vgg16 = VGG16(weights= None,
                        include_top = False, # don't include FC layers
                        input_shape = img_shape)
    top_model = GlobalAveragePooling2D()(vgg16.output)


outs = []
for _ in range(5):
    dense1 = Dense(64, activation='relu')(top_model)
    drop = Dropout(0.5)(dense1)
    dense2 = Dense(num_symbols, activation='softmax')(drop)
    outs.append(dense2)

model = Model(vgg16.inputs, outs)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
hist = model.fit(X_train, [y_train[0], y_train[1], y_train[2], y_train[3], y_train[4]], batch_size=32, epochs=30,verbose=1, validation_split=0.2)