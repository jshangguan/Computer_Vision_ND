# Project 1: Facial Keypoint Detection
In this project, I defined and trained a convolutional neural network to perform facial keypoint detection. The complete pipeline includes: 
1. Detect all the faces in an image using a Haar Cascade detector.
2. Pre-process those face images so that they are grayscale, and transformed to a Tensor of the expected input size.
3. Use the trained model to detect facial keypoints on the image.
## Training and Testing Data
This facial keypoints dataset consists of 5770 color images, which has been extracted from the [YouTube Faces Dataset](https://www.cs.tau.ac.il/~wolf/ytfaces/) and includes videos of people in YouTube videos. These videos have been fed through some processing steps and turned into sets of image frames containing one face and the associated keypoints.
* 3462 of these images are training images, for you to use as you create a model to predict keypoints.
* 2308 are test images to be used to test the accuracy of your model.
## Results
