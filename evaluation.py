#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 15:28:36 2020

@author: joel
"""
import cv2
import numpy as np
import pandas as pd

def get_frame_nr(video_file):
    cap=cv2.VideoCapture(video_file)
    _,_=cap.read()
    frames = 1
    while(cap.isOpened()):
        ret,_=cap.read()
        if ret==True:
            frames+=1
        else:
            break
    return frames
    

def read_cuts(file, nr_frames):
    scene_cut= pd.read_csv(file)
    frame_nos=scene_cut['frame_no']
    cut_frames=np.array(frame_nos)
    cut_bool = []
    for frame in range(nr_frames):
        if frame+1 in cut_frames:
            cut_bool.append(1)
        else:
            cut_bool.append(0)
    return np.array(cut_bool)

def save_csv(data, out_name):
    csv_data=pd.DataFrame(columns=['frame_no'])
    frames = 0
    for i in data:
        frames+=1
        if i == 1:
             csv_thread=pd.Series(data=frames,index=['frame_no'])
             csv_data=csv_data.append(csv_thread,ignore_index=True)
    csv_data.to_csv(out_name)

def evaluate_SBD(prediction, solution):
    
    fp=0
    fn=0
    tp=0
    tn=0
    pos = np.count_nonzero(np.array(prediction) == 1)
    
    for i in range(len(solution)):
        if prediction[i] == 0 and solution[i]==0:
            tn+=1
            
        if prediction[i] == 0 and solution[i]==1:
            fn+=1
            
        if prediction[i] == 1 and solution[i]==1:
            tp+=1
            
        if prediction[i] == 1 and solution[i]==0:
            fp+=1

    accuracy = (fp+tn)/(tp+fp+fn+tn)
    precision = tp/pos
    recall = tp/(tp+fn)
    f1_score = 2*(recall * precision) / (recall + precision)
    
    results=dict(accuracy=accuracy, 
                 precision=precision, 
                 recall=recall, 
                 f1_score=f1_score,
                 fp=fp,
                 fn=fn,
                 tp=tp,
                 tn=tn
                 )
    
    return results
    

