# Computer Vision Nanodegree
This repository contains my exercises and projects for the Computer Vision Nanodegree.
## Project 1: Facial Keypoint Detection
In this project, I defined and trained a convolutional neural network to perform facial keypoint detection. The complete pipeline includes: 
1. Detect all the faces in an image using a Haar Cascade detector.
2. Pre-process those face images so that they are grayscale, and transformed to a Tensor of the expected input size.
3. Use the trained model to detect facial keypoints on the image.
### Training and Testing Data
This facial keypoints dataset consists of 5770 color images, which has been extracted from the [YouTube Faces Dataset](https://www.cs.tau.ac.il/~wolf/ytfaces/) and includes videos of people in YouTube videos. These videos have been fed through some processing steps and turned into sets of image frames containing one face and the associated keypoints.
* 3462 of these images are training images, for you to use as you create a model to predict keypoints.
* 2308 are test images to be used to test the accuracy of your model.
### Results
![image](https://github.com/jshangguan/CVND/blob/master/P1_Facial_Keypoints/images/obamas1_keypoints.png)
![image](https://github.com/jshangguan/CVND/blob/master/P1_Facial_Keypoints/images/obamas2_keypoints.png)

## Project 2: Image Captioning Project
In this project, I designed and trained a CNN-RNN model for automatically generating image captions. The training dataset is [Microsoft Common Objects in COntext (MS COCO) dataset](http://cocodataset.org/#home), which is a large-scale dataset for scene understanding. The dataset is commonly used to train and benchmark object detection, segmentation, and captioning algorithms. 
The modal uses a CNN as an image “encoder”, by first pre-training it for an image classification task and using the last hidden layer as an input to the RNN decoder that generates sentences. It generates complete sentences in natural language from an input image, as shown on the example below.

![image](https://github.com/jshangguan/Computer_Vision_ND/blob/master/P2_Image_Captioning/images/encoder-decoder.png)

### Results
Images with relatively accurate captions:

![image](https://github.com/jshangguan/Computer_Vision_ND/blob/master/P2_Image_Captioning/images/accurate_caption_1.png)
![image](https://github.com/jshangguan/Computer_Vision_ND/blob/master/P2_Image_Captioning/images/accurate_caption_2.png)

Images with relatively inaccurate captions:

![image](https://github.com/jshangguan/Computer_Vision_ND/blob/master/P2_Image_Captioning/images/inaccurate_caption_1.png)
![image](https://github.com/jshangguan/Computer_Vision_ND/blob/master/P2_Image_Captioning/images/inaccurate_caption_2.png)

## Project 3: Landmark Detection & Robot Tracking (SLAM)

### Project Overview

In this project, I implemented SLAM (Simultaneous Localization and Mapping) for a 2 dimensional world. I combined robot sensor measurements and movement to create a map of an environment from only sensor and motion data gathered by a robot, over time. SLAM gives a way to track the location of a robot in the world in real-time and identify the locations of landmarks such as buildings, trees, rocks, and other world features. This is an active area of research in the fields of robotics and autonomous systems. 

*Below is an example of a 2D robot world with landmarks (purple x's) and the robot (a red 'o') located and found using *only* sensor and motion data collected by that robot. This is just one example for a 50x50 grid world.*

<p align="center">
  <img src="https://github.com/jshangguan/Computer_Vision_ND/blob/master/P3_Implement_SLAM/images/robot_world.png" width=50% height=50% />
</p>

## Results
The final position of the robot is (24.66580405687, 82.86809855648303). 

*Below is the 2D robot world with landmarks (purple x's) and the robot (a red 'o') located and found using *only* sensor and motion data collected by that robot.*
<p align="center">
  <img src="https://github.com/jshangguan/Computer_Vision_ND/blob/master/P3_Implement_SLAM/images/final_pose.png" width=50% height=50% />
</p>
