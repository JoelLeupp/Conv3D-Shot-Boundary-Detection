#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 10 16:54:48 2020

@author: joel
"""
############################ VISUALISATION OF PREDICTION #####################
from datagen import datagen

video = 'test_data/final.mp4'
csv = 'test_data/final.csv'

#compare prediction from the Conv3D model and cineast
comp = datagen(10,video, csv)
comp.compare_prediction('test_data/predict.csv','test_data/cineast.csv')

#prediction from model
conv = datagen(10,video, 'test_data/predict.csv')
conv.eval_csv()

#prediction from cineast
cin = datagen(10,video, 'test_data/cineast.csv')
cin.eval_csv()

#true cuts as reference
ref = datagen(10,video, csv)
ref.eval_csv()



