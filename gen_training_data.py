#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create datasets for training, testing or prepare for prediction
"""
from data_extractor import data_extractor
import numpy as np
from tqdm import tqdm
import pandas as pd
import glob
video = 'train_data/train_big.mp4'
csv = 'train_data/train_big.csv'
output = 'train_data/numpy_data/train_'

def create_dataset(video_file,csv_file,name):
     """
     creats numpy arrays im_data.npy with the shape (10'000',64,64,10,3) and 
     a numpy array of cut.npy of the shape (10'000',2) which indicates if 
     middle frames are of the same shot or not.
     """
     gen=data_extractor(no_frames=10,video_file=video_file,csv_file=csv_file)
     image_data,cut=[],[]
     count=0
     for image_64,prediction in tqdm(gen.data_extrac(),total=gen.len):
         image_data.append(image_64.reshape((64, 64, 10, 3)))
         cut.append(prediction)
         
         if len(cut) == 10000:
            print("\n saving data")
            count+=1
            np.save(name+'im_data'+"{:02d}".format(count)+'.npy',image_data)
            np.save(name+'cut'+"{:02d}".format(count)+'.npy',cut)
            image_data,cut=[],[]
                 
     np.save(name+'im_data'+"{:02d}".format(count)+'.npy',image_data)
     np.save(name+'cut'+"{:02d}".format(count)+'.npy',cut)
     print('saving data final')
     
def create_dataset_small(video_file,csv_file,name):
    """
    use this function to create one big numpy dataset but it works only if the
    video is not too long because numpy stores data in memory
    """
    gen=data_extractor(no_frames=10,video_file=video_file,csv_file=csv_file)
    image_data,cut=[],[]
    for image_64,prediction in tqdm(gen.data_extrac(),total=gen.len):
        image_data.append(image_64.reshape((64, 64, 10, 3)))
        cut.append(prediction)
        
        if len(cut) == 5000:
            try:
                image_loaded=list(np.load(name+'im_data.npy', allow_pickle=True))
                cut_loaded=list(np.load(name+'cut.npy', allow_pickle=True))
                print("data has been loaded")
                image_loaded.extend(image_data)
                cut_loaded.extend(cut)
                image_loaded=np.array(image_loaded)
                cut_loaded=np.array(cut_loaded)
                np.save(name+'im_data.npy',image_loaded)
                np.save(name+'cut.npy',cut_loaded)
                print('saving data')
                image_data,cut=[],[]
                
            except:
                print("\n saving data")
                np.save(name+'im_data.npy',image_data)
                np.save(name+'cut.npy',cut)
                image_data,cut=[],[]
                
    image_loaded=list(np.load(name+'im_data.npy', allow_pickle=True))
    cut_loaded=list(np.load(name+'cut.npy', allow_pickle=True))
    print("data has been loaded")
    image_loaded.extend(image_data)
    cut_loaded.extend(cut)
    image_loaded=np.array(image_loaded)
    cut_loaded=np.array(cut_loaded)
    np.save(name+'im_data.npy',image_loaded)
    np.save(name+'cut.npy',cut_loaded)
    print('saving data final')


    


