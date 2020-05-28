#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 11:49:57 2020

@author: joel
"""
from datagen import datagen
import numpy as np
from tqdm import tqdm
import pandas as pd
import glob
video = 'train_data/final.mp4'
csv = 'train_data/final.csv'
output = 'train_data/test_'

def create_dataset(video_file,csv_file,name):
     gen=datagen(no_frames=10,video_file=video_file,csv_file=csv_file)
     image_data,cut=[],[]
     count=0
     for image_64,prediction in tqdm(gen.data_extrac(),total=gen.len):
         image_data.append(image_64.reshape((64, 64, 10, 3)))
         cut.append(prediction)
         
         if len(cut) == 5000:
            print("\n saving data")
            count+=1
            np.save(name+'im_data'+"{:02d}".format(count)+'.npy',image_data)
            np.save(name+'cut'+"{:02d}".format(count)+'.npy',cut)
            image_data,cut=[],[]
                 
     np.save(name+'im_data'+"{:02d}".format(count)+'.npy',image_data)
     np.save(name+'cut'+"{:02d}".format(count)+'.npy',cut)
     print('saving data final')
     
create_dataset(video,csv,output)

img_set = glob.glob('train_data/test_im_data*.npy')
img_set.sort()
image_data=[]
for i in img_set:
    image_data.append(np.load(i, allow_pickle=True))
    print(i)

