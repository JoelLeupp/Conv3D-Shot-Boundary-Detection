#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 14:32:54 2020

@author: joel
"""
import glob
from model import Conv3D_model
import os
import numpy as np
from tensorflow.keras.utils import to_categorical
import evaluation as eva
from datagen import datagen
import json
import cv2

#wd at /home/GitHub/Conv3D-Shot-Boundary-Detection
video = 'train_data/final.mp4'
csv = 'train_data/final.csv'

img_set = glob.glob('train_data/numpy_data/test_im_data*.npy')
img_set.sort()
cut_set = glob.glob('train_data/numpy_data/test_cut*.npy')
cut_set.sort()

model3D = Conv3D_model()
#model= Conv3D_model().model_3d
weights = 'modelWeights2.h5'

#train model with all 
epoch = 5
for i in range(10):
    for img, cut in zip(img_set,cut_set):
        print(img)
        print(cut)
        try:
            model3D.model_3d.load_weights(weights)
            model3D.train(img,cut, epoch=epoch, batch_size=32, out_weight=weights)
        except:
            model3D.train(img,cut, epoch=epoch, batch_size=32, out_weight=weights)
    
   
        
        