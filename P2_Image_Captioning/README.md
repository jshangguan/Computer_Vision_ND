# Project 2: Image Captioning Project
In this project, I designed and trained a CNN-RNN model for automatically generating image captions. The training dataset is [Microsoft Common Objects in COntext (MS COCO) dataset](http://cocodataset.org/#home), which is a large-scale dataset for scene understanding. The dataset is commonly used to train and benchmark object detection, segmentation, and captioning algorithms. The modal uses a CNN as an image “encoder”, by first pre-training it for an image classification task and using the last hidden layer as an input to the RNN decoder that generates sentences. It generates complete sentences in natural language from an input image, as shown on the example below.

![image](https://github.com/jshangguan/Computer_Vision_ND/blob/master/P2_Image_Captioning/images/encoder-decoder.png)

## Results
Images with relatively accurate captions:

![image](https://github.com/jshangguan/Computer_Vision_ND/blob/master/P2_Image_Captioning/images/accurate_caption_1.png)
![image](https://github.com/jshangguan/Computer_Vision_ND/blob/master/P2_Image_Captioning/images/accurate_caption_2.png)

Images with relatively inaccurate captions:

![image](https://github.com/jshangguan/Computer_Vision_ND/blob/master/P2_Image_Captioning/images/inaccurate_caption_1.png)
![image](https://github.com/jshangguan/Computer_Vision_ND/blob/master/P2_Image_Captioning/images/inaccurate_caption_2.png)
