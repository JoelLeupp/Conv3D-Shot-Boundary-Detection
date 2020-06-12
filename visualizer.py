#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import argparse
import numpy as np
from math import ceil,floor
import pandas as pd 

'''
Helper class to visualize the shot prediction
 
'''
class visualizer():
    def __init__(self,no_frames,video_file=None,csv_file=None):
        if video_file==None:
            input_video=input()
        self.extra_frames=ceil(no_frames/2.0)
        self.csv_file=csv_file
        self.cap=cv2.VideoCapture(video_file)
        self.len=int(self.cap.get(7))
        channel=3
        self.panel_pipe=np.zeros((no_frames,128,128,channel))
        self.prediction=0,0

        # internal funtion to insert images into queue
    def _image_insert(self,frame_128):
        self.panel_pipe=np.append(self.panel_pipe[1:],[frame_128],axis=0)
        return()

        # creating the pannel for the visualization
    def _create_pannel(self):
        panel=np.hstack(self.panel_pipe)
        h,w,_=panel.shape
        panel_image=cv2.line(panel,(int(w/2),0),(int(w/2),h),(255,255,255),4)
        panel_text=np.ones((20,w,3), dtype=np.uint8)        
        panel_text=cv2.rectangle(panel_text,(1,1),(w,20),(255,255,255),thickness=cv2.FILLED)
        panel_text=cv2.putText(panel_text, 'Prediction val:{}'.format(self.prediction[0],self.prediction[1]), (int(w/2)-10, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), lineType=cv2.LINE_AA)
        panel_final=np.vstack([panel_image,panel_text]) 
        return(panel_final)
    

    #visual evaluation of the predictions given by the csv
    def eval_csv(self):

        _,init_image=self.cap.read()
        frame_128=cv2.resize(init_image,(128,128),cv2.INTER_LINEAR)

        self.panel_pipe=np.tile(frame_128,(10,1,1,1))

        
        frame_cuts = list(pd.read_csv(self.csv_file)['frame_no'])
        
        count=-self.extra_frames
        step=False
        while(self.cap.isOpened()):
#           for i in range(self.extra_frames):
            ret, frame = self.cap.read()
            if ret==True:
                count+=1
        
                frame_128=cv2.resize(frame,(128,128),cv2.INTER_LINEAR)
                self._image_insert(frame_128)
                self.prediction = (1,0) if count in frame_cuts else (0,0)
                
                panel=self._create_pannel()
                cv2.imshow('panel',panel)
                                    
                if step:
                    key = cv2.waitKey(0)
                    while key not in [ord('q'), ord('n'),ord('k')]:
                        key = cv2.waitKey(0)
                    # Quit when 'q' is pressed
                    if key == ord('q'):
                        break
                    if key==ord('k'):
                        step=False
                else:
                    key = cv2.waitKey(100) & 0xFF
                    # if the `q` key was pressed, break from the loop
                    if key == ord("q"):
                        break
                    if key == ord("k"):
                        step=True
                    if key == ord("b"):
                        key = cv2.waitKey(0) & 0xFF 
                 
            else:
                break
        self.cap.release()
        cv2.destroyAllWindows()
        return('Done')
    
    def compare_prediction(self, csv_1,csv_2):

        _,init_image=self.cap.read()
  
        frame_128=cv2.resize(init_image,(128,128),cv2.INTER_LINEAR)

        self.panel_pipe=np.tile(frame_128,(10,1,1,1))

        frame_cuts1 = list(pd.read_csv(csv_1)['frame_no'])
        frame_cuts2 = list(pd.read_csv(csv_2)['frame_no'])
        
        count=-self.extra_frames
        step=False
      
        while(self.cap.isOpened()):
            ret, frame = self.cap.read()
            if ret==True:
                count+=1

                frame_128=cv2.resize(frame,(128,128),cv2.INTER_LINEAR)
                self._image_insert(frame_128)
                prediction1 = (1,0) if count in frame_cuts1 else (0,0)
                prediction2 = (1,0) if count in frame_cuts2 else (0,0)
                
                panel=np.hstack(self.panel_pipe)
                h,w,_=panel.shape
                panel_image=cv2.line(panel,(int(w/2),0),(int(w/2),h),(255,255,255),4)
                panel_text=np.ones((20,w,3), dtype=np.uint8)        
                panel_text=cv2.rectangle(panel_text,(1,1),(w,20),(255,255,255),thickness=cv2.FILLED)
                panel_text=cv2.putText(panel_text, '(Prediction) Conv3D:{}  Cineast:{}'.format(prediction1[0],prediction2[0]), 
                                       (int(w/2)-220, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), lineType=cv2.LINE_AA)
                panel=np.vstack([panel_image,panel_text]) 
                cv2.imshow('panel',panel)
                                    
                if step:
                    key = cv2.waitKey(0)
                    while key not in [ord('q'), ord('n'),ord('k')]:
                        key = cv2.waitKey(0)
                    # Quit when 'q' is pressed
                    if key == ord('q'):
                        break
                    if key==ord('k'):
                        step=False
                else:
                    key = cv2.waitKey(100) & 0xFF
                    # if the `q` key was pressed, break from the loop
                    if key == ord("q"):
                        break
                    if key == ord("k"):
                        step=True
                    if key == ord("b"):
                        key = cv2.waitKey(0) & 0xFF 
                #out.write(panel)

            else:
                break
        self.cap.release()
       # out.release()
        cv2.destroyAllWindows()
        return('Done')

