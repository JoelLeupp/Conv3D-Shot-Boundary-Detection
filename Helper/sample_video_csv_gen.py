'''
Code for generating the augmented dataset resulting a .mp4 file and .csv file. 
'''

import glob
import numpy as np 
from math import ceil
from augmentation_helper import speed,shift_channel,shift_hue,bw,blur,artifical_flash
from moviepy.editor import VideoFileClip,concatenate_videoclips
import pandas as pd
from math import ceil
from dataset_generator import video_generator
from dataset_generator import video_generator2
import os
import cv2

#gathering the video samples to be augmented and generated
root="directory to video clips"
sample_vid_set=glob.glob(os.path.join(root,'output tv2007d/*.mp4'))

out_folder = "outfolder ot test/train data"
out_temp = os.path.join(out_folder,'temp')

#using the generator to augment clips
for i,aug_clip in video_generator(sample_vid_set,samples=1000):
   aug_clip.write_videofile(os.path.join(out_temp,"temp_"+str(i)+".mp4"))
       
       
videofiles = glob.glob(out_temp+ '/*.mp4')
videofiles.sort()

video_index = 0
cap = cv2.VideoCapture(videofiles[0])

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output = os.path.join(out_folder, 'final.mp4')
out = cv2.VideoWriter(output, fourcc, 15, (320, 240), 1)

csv_data=pd.DataFrame(columns=['frame_no'])

frame_count=0
while(cap.isOpened()):
    ret, frame = cap.read()
    if frame is None:
        print ("end of video " + str(video_index))
        video_index += 1  
        #write to csv
        csv_thread=pd.Series(data=[frame_count-1],index=['frame_no'])
        csv_data=csv_data.append(csv_thread,ignore_index=True)
        
        if video_index >= len(videofiles):
            break
        cap = cv2.VideoCapture(videofiles[ video_index ])
        #ret, frame = cap.read()
        #frame_count+=1
    else:
        frame_count+=1
        out.write(frame)

cap.release()
out.release()
csv_data.to_csv(os.path.join(out_folder,"aug_data_eval.csv"))

