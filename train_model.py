#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 14:32:54 2020

@author: joel
"""
import glob
from model import Conv3D_model

#wd at /home/GitHub/Conv3D-Shot-Boundary-Detection
video = 'train_data/train_big.mp4'
csv = 'train_data/train_big.csv'

img_set = glob.glob('train_data/numpy_data/train_im_data*.npy')
img_set.sort()
cut_set = glob.glob('train_data/numpy_data/train_cut*.npy')
cut_set.sort()

model3D = Conv3D_model(lr=0.1)
#model= Conv3D_model().model_3d
weights = 'modelWeights.h5'

#train model with all 
epoch = 20

for img, cut in zip(img_set[2:],cut_set[2:]):
    print(img)
    print(cut)
    try:
        model3D.model_3d.load_weights(weights)
        model3D.train(img,cut, epoch=epoch, batch_size=32, out_weight=weights)
    except:
        model3D.train(img,cut, epoch=5*epoch, batch_size=32, out_weight=weights)
        model3D = Conv3D_model(lr=0.01)
        
        
        