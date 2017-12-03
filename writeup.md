# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* [**model.py:**](https://github.com/jacquestkirk/Car_ND_BehavioralCloning/blob/master/model.py) The script to create and train the model.
* [**imageGenerator.py:**](https://github.com/jacquestkirk/Car_ND_BehavioralCloning/blob/master/imageGenerator.py) A generator function that loads images into python in batches
* [**drive.py:**](https://github.com/jacquestkirk/Car_ND_BehavioralCloning/blob/master/drive.py) for driving the car in autonomous mode. I didn't make any changes here.
* [**model.h5:**](https://github.com/jacquestkirk/Car_ND_BehavioralCloning/blob/master/model.h5) The trained convolutional neural network. 
* [**writeup_report.md:**](https://github.com/jacquestkirk/Car_ND_BehavioralCloning/blob/master/writeup.md) This file. It's a summary of the results
* [**video.mp4**](https://github.com/jacquestkirk/Car_ND_BehavioralCloning/blob/master/video.mp4) Video of the car driving autonomously along the track

#### 2. Submission includes functional code

I used pycharm as the IDE for this project, so running this code is from a pycharm perspective. Also, note that this code was run on a windows machine, so the slashes in the path might be wrong if you are using mac or linux

Training data is stored in the TrainingData folder. The log is in driving_log.csv. The image data was too large and exceeded Github's limit, so it is not included in the repo. If you'd like I can send you a zip of the data. 

To train the network run model.py. This generates model.h5.

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The pipeline I used for training and validating the model are in the ModelBuilder function. Magic numbers are stored near the top of the file for easy access and tweaking. 

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is based off of the Lenet architecture. It can be seen in the ModelBuilder function. Like Lenet, It uses convolutional layers followed by two dense layers and an output layer. Pooling is used between convolutional layers and relus are used as the non-linearities between the layers. 

The changes to Lenet are:
- Added processing steps including: 
	- 2D cropping to remove the sky and car bumper so as to not confuse the neural network
	- Input conditioning to prevent activations from blowing up or collapsing.
- 3rd convolutaion layer:
	- I accidentally addeded this, but it worked well, so I kept it. 
- Added dropout layers: 
	- My validation loss was much higher than my testing loss, so I added some dropout layers to avoid overfitting to the training data
- Output layer:
	- Reduced output layer to be of length 1 since we are only generating one output. And since we are not doing classification, I removed the softmax non-linearity



#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers before each dense layer in order to reduce overfitting (model.py lines 49 and 51). 

20% of the data was kept as a validation set and the model was trained and validated on these different data sets to ensure that the model was not overfitting (code line 106). 

Data was collected driving backwards through the track in order to give it more varied training data. 

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 59). 

Dropouts were tuned so that validation and testing losses were reasonably close. 

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the center of the road. I recorded one lap in the forward direction because that's what I want the car to do. then I drove one in the reverse direction in order to add some varied inputs. Finally, intermediate models had trouble getting off of the bridge, so I recorded an extra pass there. 

Training data was agumented with flips and the left and right cameras. This can be seen in lines 93-102. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My initial data set was taken by driving the car one lap forwards, and one lap backwards. 

Pre-processing is relatively cheap computationally, so I started by throwing all of the pre-processing I could think of at it. 

As for the network itsself, I decided to start with the relatively simple Lenet architecture and see how it would perform. I was plesantly surprised when the model actually did pretty well. It seemed like a good starting point, so I decided to see how far I could take this model. 

My training and validation losses were very similar, which implies overfitting both the training and validation sets. To avoid this overfitting, I decided to add more training data. An easy way to do that was to include left and right camera data. So I added the extra data in and performance was better. 

I realized that it was having some problems getting over the bridge. The end of the bridge is not aligned with the center of the road. My network was having problems with this, so I recorded some extra data making this transition. 

![The bridge that caused problems, notice that the center of the bridge is not aligned with the center of the road](/BridgeToRoad.jpg)

When I retrained my network, I realized that my car was drifting to the side. I hypothesized that it was because I added extra right turns, but didn't drive the bridge backwards, so I did not have an equal amount of left turns. To combat this I augmented the training data with flips. I only flipped images where the steering angle was greater than some threshold since straignt driving will not change much between flips. I also wanted to represent straight driving less so that the car wouldn't have a bias towards driving straight. 

I took a look at my training and validation losses again. This time there was a wide discrepancy between training and validation. I added some dropouts and tweaked the dropout percentages optimizing the validation loss. 

When I trained this model, the car behaved well, so that's what I stuck with. 

#### 2. Final Model Architecture

The final model architecture is defined in the Model Builder function in model.py starting at line 24. It consists of the following stages. 

1. **Cropping:** Remove the top and bottom of the image. These contain features that are not useful in determining how to steer and may confuse the neural network. 
2. **Input Scaling:** I scaled the inputs to prevent activations from blowing up or collapsing. I scaled the inputs by x / 127.5 - 1.
3. **5x5x6 Convolution:** First convolution layer, Relu activations
4. **Max Pooling:** 2x2 max pooling to reduce image size. 
5. **5x5x6 Convolution:** Second convolution layer, Relu activations
6. **Max Pooling:** 2x2 max pooling to reduce image size. 
7. **5x5x6 Convolution:** Third convolution layer, Relu activations
8. **Flatten:** Flatten activations to a vector to use with fully connected layers
9. **Dropout (50%):** Dropout to add regularization
10. **Dense:** 120 long, relu activation
11. **Dropout (20%):** Dropout to add regularization
12. **Dense:** 84 long, relu activation
13. **Output:** 1 long 



#### 3. Creation of the Training Set & Training Process

Training and Validation sets consisted of the following: 

1) Drive a lap of the track forwards
2) Drive a lap of the track backwards
3) An extra forward drive through the bridge
4) Left, right, and center camera data. 
5) Flipped images for left, right, and center camera when the absolute value of steering angle is greater than 0.75.


Refer back to question 1 of "Model Architecture and Training Strategy" for the reasons behind these datasets. 

I preprocessed the data by doing the following.
1) Cropping
2) Input scaling

Refer back to question 2 of "Model Architecture and Training Strategy" for the reasons behing these datasets. 


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting and if I had too small of a dataset. I used mean square error as my loss function since the steering angle is a continuous output. 

I used 10 epochs. It did not take too long to run so I used it as a starting point. I did not get to tweak this number before I got a working network. 

I used an adam optimizer so that manually training the learning rate wasn't necessary.

The results of the image are shown in the image below. 
![](https://github.com/jacquestkirk/Car_ND_BehavioralCloning/blob/master/figure_1.jpeg)

Allowing the simulator to run acts like the test data. But I would have liked to have had another similar test track, that the neural network has never seen before to act more like test data. Since I only have one track (the second is VERY different) I can't use an actual test set. 


