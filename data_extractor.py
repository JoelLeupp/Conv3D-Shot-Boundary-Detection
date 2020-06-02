#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import argparse
import numpy as np
from math import ceil,floor
import pandas as pd 

'''
Datagenerator from the video file and CSV 
 
'''
class data_extractor():
    def __init__(self,no_frames,video_file=None,csv_file=None):
        if video_file==None:
            input_video=input()
        self.extra_frames=ceil(no_frames/2.0)
        self.csv_file=csv_file
        self.cap=cv2.VideoCapture(video_file)
        self.len=int(self.cap.get(7))
        channel=3
        self.image_pipe=np.zeros((no_frames,64,64,channel))
        self.prediction=0,0

        # internal funtion to insert images into queue
    def _image_insert(self,frame_64):
        self.image_pipe=np.append(self.image_pipe[1:],[frame_64],axis=0)
        return()
    
    # Generator for the extracting the image and the corresponding labels
    def data_extrac(self):
        _,init_image=self.cap.read()
        frame_64=cv2.resize(init_image,(64,64),cv2.INTER_LINEAR).astype(np.float32)
        frame_64/=255

        self.image_pipe=np.tile(frame_64,(10,1,1,1))

        #csv data
        scene_cut=pd.read_csv(self.csv_file,index_col=0)
        frame_nos=scene_cut['frame_no']
        ##cut_frames=frame_nos.as_matrix()
        cut_frames=np.array(frame_nos)

        count=-self.extra_frames  
        while(self.cap.isOpened()):
            ret, frame = self.cap.read()
            if ret==True:
                count+=1
                frame_64=cv2.resize(frame,(64,64),cv2.INTER_LINEAR).astype(np.float32)
                frame_64/=255.
                self._image_insert(frame_64)
                #csv data retrival
                if count in cut_frames:
                    prediction = 1
                else:
                    prediction = 0
                yield self.image_pipe, prediction
            else:
                break


