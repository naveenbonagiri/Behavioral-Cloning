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

[image6]: ./ImageRef/OrigImage.jpg "Normal Image"
[image7]: ./ImageRef/flippedImage.jpg "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 350 and 400 (model.py lines 122-130) 

The model includes RELU layers to introduce nonlinearity (code lines 122 to 130), and the data is normalized in the model using a Keras lambda layer (code line 114). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 135, 138, 141). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 154). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 169).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to the NVIDIA CNN. I thought this model might be appropriate because it automatically learns detecting useful road features without explicitly decomposing the problem and give steering angle as final output.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model by adding dropout layers to address overfitting issue.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases I was needed to adjust number of epochs, parameters to fit_generator.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 110-144) consisted of a convolution neural network with the following layers and layer attributes. 

input: 160x320x3 with Lambda Normalization
Cropping: crop unwated portions of the input like sky, top and bottom portion of the image
convolution: 24depth, 5x5 kernel with 2x2 stride, RELU
convolution: 36depth, 5x5 kernel with 2x2 stride, RELU
convolution: 48depth, 5x5 kernel with 2x2 stride, RELU
convolution: 64depth, 3x3 kernel, RELU
convolution: 64depth, 3x3 kernel, RELU
flatten layer
fully connected: 100 depth
dropout layer with 0.2 probability
fully connected: 50 depth
dropout layer with 0.2 probability
fully connected: 10 depth
dropout layer with 0.2 probability
fully connected: 1 depth
Output

#### 3. Creation of the Training Set & Training Process

I collected the data using center lane driving, by going to the side of the lanes and recovering. 
But I did not use that for below reasons. 
1. I used the data from workspace and learning was good enough to drive in autonomus mode.
2. I ran into troubles getting the data into workspace from simulator. 

To augment the data sat, I also flipped images and angles thinking that this would help the model learn efficiently. 

![alt text][image6]
![alt text][image7]

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by loss, rmse. I used an adam optimizer so that manually training the learning rate wasn't necessary.
