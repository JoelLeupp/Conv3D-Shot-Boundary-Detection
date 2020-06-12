#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
implementation of 3D Conv model class
"""
from data_extractor import data_extractor
import numpy as np
from tqdm import tqdm
import pandas as pd

import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Conv3D,InputLayer,Dense,Activation,MaxPool3D,Flatten
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.metrics import categorical_crossentropy,categorical_accuracy
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping,TensorBoard
from tensorflow.keras.utils import to_categorical

class Conv3D_model():
    def __init__(self,lr=0.1):
        self.model_3d=self.model(lr)
        self.cuts = None
        self.images = None
        print(self.model_3d.summary())
    
    # Model with 10 input frames and 1 prediction. 
    def model(self,lr=0.1):
        input_layer=(64,64,10,3)
        kernel_conv1=(5,5,3)
        kernel_conv2=(3,3,3)
        kernel_conv3=(6,6,1)
        kernal_softmax=(1,1,4)

        model = Sequential()
        model.add(InputLayer(input_layer))
        model.add(Conv3D(kernel_size=kernel_conv1, filters=16, strides=(2, 2, 1), padding='valid', activation='relu'))
        model.add(Conv3D(kernel_size=kernel_conv2, filters=24, strides=(2, 2, 1), padding='valid', activation='relu'))
        model.add(Conv3D(kernel_size=kernel_conv2, filters=32, strides=(2, 2, 1), padding='valid', activation='relu'))
        model.add(Conv3D(kernel_size=kernel_conv3, filters=12, strides=(2, 2, 1), padding='valid', activation='relu'))
        model.add(MaxPool3D(kernal_softmax))
        model.add(Flatten())
        model.add(Dense(2,activation='softmax'))
        opt = SGD(learning_rate=lr)
        model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=[categorical_crossentropy,categorical_accuracy])

        return(model)
    
    def FCN(self,n=None):
        """
        fully convolutional neural network (doesn't work as intended when scaling up input dimension')
        """
        input_layer=(64,64,n,3)
        kernel_conv1=(5,5,3)
        kernel_conv2=(3,3,3)
        kernel_conv3=(6,6,1)
        kernal_softmax=(1,1,4)
        
        model = Sequential()
        model.add(InputLayer(input_layer))
        model.add(Conv3D(kernel_size=kernel_conv1, filters=16, strides=(2, 2, 1), padding="valid", activation='relu'))
        model.add(Conv3D(kernel_size=kernel_conv2, filters=24, strides=(2, 2, 1), padding="valid", activation='relu'))
        model.add(Conv3D(kernel_size=kernel_conv2, filters=32, strides=(2, 2, 1), padding='valid', activation='relu'))
        model.add(Conv3D(kernel_size=kernel_conv3, filters=12, strides=(2, 2, 1), padding='valid', activation='relu'))
        model.add(MaxPool3D((1,1,4),strides=(1,1,1)))
        model.add(Conv3D(kernel_size=(1,1,1), filters=2, strides=(1, 1, 1),padding='valid', activation='softmax'))
        model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=[categorical_crossentropy,categorical_accuracy])
        model.summary()
        return(model)
    
    def train(self,  img_data, cut, epoch=100, batch_size=32, out_weight='modelWeights.h5'):
        try:
            print("loading Data")
            image_data=np.load(img_data, allow_pickle=True)
            cut=np.load(cut, allow_pickle=True)
            print('#'*50)
            print("data has been loaded")
        
        except:
            print('create training data first')
        
        image_train,image_test=image_data[:int(.8*len(cut))],image_data[int(.8*len(cut)):]
        cut_train,cut_test=cut[:int(.8*len(cut))],cut[int(.8*len(cut)):]
        cut_train, cut_test = to_categorical(cut_train), to_categorical(cut_test)
        self.model_3d.fit(image_train,cut_train, batch_size =batch_size, epochs=epoch , validation_data=(image_test,cut_test))
        self.model_3d.save_weights(out_weight)
        del(image_data)
        del(cut)
        print('training finished')
        
    def predict_shots(self,video_file,csv_file, weights):
        gen=data_extractor(no_frames=10,video_file=video_file,csv_file=csv_file)
        self.model_3d.load_weights(weights)
        predict = []
        for image_64,_ in tqdm(gen.data_extrac(),total=gen.len):
            img = image_64.reshape((1,64, 64, 10, 3))
            predict.append(np.argmax((self.model_3d.predict(img))))
        return predict
