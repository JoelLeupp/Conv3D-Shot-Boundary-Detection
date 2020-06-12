import glob
import numpy as np 
import pandas as pd
from clip_generator import clip_generator
import os
import cv2

#gathering the video samples to be augmented and generated
sample_vid_set=glob.glob('../output tv2007d/*.mp4')
out_folder = "../train_data"
out_temp = os.path.join(out_folder,'temp')
out_name = 'training_set'

#creating a temporary folder for the augmented videos
os.mkdir(out_temp)

#temporary store sampled augmented shots
for i,aug_clip in clip_generator(sample_vid_set,samples=10, is_rand_sample=False):
   aug_clip.write_videofile(os.path.join(out_temp,"temp_"+str(i)+".mp4"))
       
#go through all clips and create an mp4 and a csv with all cuts
videofiles = glob.glob(out_temp+ '/*.mp4')
videofiles.sort()

video_index = 0
cap = cv2.VideoCapture(videofiles[0])

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output = os.path.join(out_folder, out_name+'.mp4')
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
csv_data.to_csv(os.path.join(out_folder,out_name+".csv"))




