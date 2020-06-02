# Conv3D-Shot-Boundary-Detection
The automatically generated labeled shot boundary dataset and the used convolutional model is inspired by the paper 'Ridiculously Fast Shot Boundary Detection with Fully Convolutional Neural Networks' which can be found here: https://arxiv.org/abs/1705.08214

## Data Collection
1. Download this big dataset for shot boundary detection:  https://nsl.cs.sfu.ca/projects/DeepSBD/dataset/DeepSBD.tar.gz
2. In the file Helper/deepSBD_video.py change the variable root_deepSBD to the direcory of the downloaded deepSBD folder
3. run the script Helper/deepSBD_video.py, which creates two folders:  "output tv2007d" and "output tv2007t" with 5'228 and 3'125 video-clips which are 11-15 frame long shots.

## Create Train/Test dataset
Create a labeled mp4 video out of the shots from "output tv2007t/d" with the script Helper/labeled_vide_gen.py

specify the folder from which you want to take the shots from and how many random samples you want to draw
(For the training dataset I took 10'000 samples from tv200d)

What does it do exactly?
From the specifyed folder it takes a random shot and with a given probability it will modify the shot with a random process from the Helper/clip_editor.py such as fade-in/out, artificial flash, random hue changes, color channel shuffle, random bw frames and random blur.

Repeat this process for how many samples where given and in the end add all these clips together to create the mp4 video, while adding the clips it count the frames and creates a CSV file which stores the number of all frames with a transition.

## 3D Convolutional Model
Input Shape: (None,64,64,10,3)


![](https://github.com/JoelLeupp/Conv3D-Shot-Boundary-Detection/blob/master/model.png)

Special thanks to https://github.com/abramjos for inspiring the data preparation process as well as the data visualisation








