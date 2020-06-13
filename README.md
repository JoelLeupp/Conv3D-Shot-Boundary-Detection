# Conv3D-Shot-Boundary-Detection
The automatically generated labeled shot boundary dataset and the used convolutional model is inspired by the paper 'Ridiculously Fast Shot Boundary Detection with Fully Convolutional Neural Networks' which can be found here: https://arxiv.org/abs/1705.08214

## Data Collection
1. Download this big dataset for shot boundary detection:  https://nsl.cs.sfu.ca/projects/DeepSBD/dataset/DeepSBD.tar.gz
2. In the file 'Helper/deepSBD_video.py' change the variable root_deepSBD to the direcory of the downloaded deepSBD folder
3. run the script 'Helper/deepSBD_video.py', which creates two folders:  "output tv2007d" and "output tv2007t" with 5'228 and 3'125 video-clips which are 11-15 frame long shots.

## Create Train/Test dataset
Create a labeled mp4 video out of the shots from "output tv2007t/d" with the script 'Helper/labeled_vide_gen.py'

specify the folder from which you want to take the shots from and how many random samples you want to draw
(For the training dataset I took 10'000 samples from tv200d)

What does it do exactly?
From the specifyed folder it takes a random shot and with a given probability it will modify the shot with a random process from the 'Helper/clip_editor.py' such as fade-in/out, artificial flash, random hue changes, color channel shuffle, random bw frames and random blur.

Repeat this process for how many samples where given and in the end add all these clips together to create the mp4 video, while adding the clips it count the frames and creates a CSV file which stores the number of all frames with a transition.

To prepare the data for training by creating and saving numpy arrays of the shape (10'000,64,64,10,3) out of the labeled training video and csv file the skript 'gen_training_data.py' is used. 

## 3D Convolutional Model
The model class is defined in 'model.py' and trained with 'train_model.py'

Detailed model architecture below

Input Shape: (None,64,64,10,3)

![](https://github.com/JoelLeupp/Conv3D-Shot-Boundary-Detection/blob/master/model.png)

## Evaluation
The model prediction followed by an evaluation is done with 'model_evaluation.py' which uses helper functions from 'evaluation.py' and the resuts where written to a json file. The results of the prediction on the final test/evaluation video where saved in the file 'test_data/eval_results.json'

The prediction can also be visualized in a 10 frame panel video. For this the file 'visual_evaluation.py' can be run which uses helper functions from the file 'visualizer.py'.

Following keyboard keys can be used to navigate through the visualizer:
    k: stop or resume panel video
    n: next frame 
    q: quit video 

Example of the prediction visualizer with a positive predicted sharp cut:

![](https://github.com/JoelLeupp/Conv3D-Shot-Boundary-Detection/blob/master/panel_prediction.png)

Example of a true negative prediction with heavy frame to frame changes (edited shot):

![](https://github.com/JoelLeupp/Conv3D-Shot-Boundary-Detection/blob/master/panel_no_prediction.png)

## References 
Special thanks to https://github.com/melgharib/DSBD for linking to the deepSBD dataset, which was used to generate the automatically labeled training dataset and to https://github.com/abramjos for inspiring the data preparation process as well as the data visualisation.








