#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate model Predictions and compare with Cineast
"""
from model import Conv3D_model
import os
import numpy as np
from tensorflow.keras.utils import to_categorical
import evaluation as eva
from gen_training_data import create_dataset_small
import json
import cv2

#wd at /home/GitHub/Conv3D-Shot-Boundary-Detection
video = 'test_data/final.mp4'
csv = 'test_data/final.csv'

model3D = Conv3D_model()
#model3D.train(epoch=100, batch_size=32, out_weight='newWeights.h5')

model= Conv3D_model().model_3d
model_weight='modelWeights.h5' #Weights.h5
model.load_weights(model_weight)

#prepare data
#create_dataset_small(video, csv,'test_data/eval_')

try:
    print("loading Data")
    image_data=np.load('test_data/eval_im_data.npy', allow_pickle=True)
    cut=np.load('test_data/eval_cut.npy', allow_pickle=True)
    cut = to_categorical(cut)
    print('#'*70)
    print("data has been loaded")
    #predict cuts with prepared data
    p=model.predict(image_data)

except:
    print('No prepared data yet! loop through whole video')
    p = model3D.predict_shots(video,csv, 'modelWeights.h5')
    
#predict cuts while running through video
#p = model3D.predict_shots(video,csv, 'Weights.h5')

res = [np.argmax(y, axis=None, out=None) for y in p]
#cut=[np.argmax(y, axis=None, out=None) for y in cut]

frame_nr =eva.get_frame_nr(video)
eva.save_csv(res[5:],'test_data/predict.csv')

sol_cut = eva.read_cuts(csv, frame_nr)
predict_cut = eva.read_cuts('test_data/predict.csv', frame_nr)
cineast_cut = eva.read_cuts('test_data/cineast.csv', frame_nr)

res_pre = eva.evaluate_SBD(prediction=predict_cut,solution=sol_cut)
res_cin = eva.evaluate_SBD(prediction=cineast_cut,solution=sol_cut)
test_result = dict(conv3D_model = res_pre, cineast = res_cin )

with open('test_data/eval_result.json','w') as f:
    json.dump(test_result, f, indent = 4)
    
eva.create_table('test_data/eval_result.json','results')
