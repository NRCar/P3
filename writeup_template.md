**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* **model.py** containing the script to create and train the model
* **drive.py** for driving the car in autonomous mode
* **model.h5** containing a trained convolution neural network 
* **writeup_report.md** Summarizing the results
* **video.mp4** containing a recorded capture on track 1

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I first tried the same model i used in the Traffic sign  classifier project [a modified version of **LeNet** ], but the model had terrible performance and the car went off the road after some amount of bad driving.
After this, I tried the [NVidia self driving Model](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) model as suggested in the udacity lesson, and after collecting data and training it on the AWS

I modified the model to add a lambda layer for normalization and then crop the top (removing the background) and the bottom (removing the car hood) to allow the model to work only on the road information. At each layer the activation used for "relu"

I also added a dense layer (dense_5) with an output of 1 and a "tanh" activation to only output steering between -1 and +1.

To prevent overfitting i used one dropout layer. ( additional dropout did not seem to help much)

A model summary is as follows:
```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
lambda_1 (Lambda)                (None, 160, 320, 3)   0           lambda_input_1[0][0]             
____________________________________________________________________________________________________
cropping2d_1 (Cropping2D)        (None, 65, 320, 3)    0           lambda_1[0][0]                   
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 31, 158, 24)   1824        cropping2d_1[0][0]               
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 14, 77, 36)    21636       convolution2d_1[0][0]            
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 5, 37, 48)     43248       convolution2d_2[0][0]            
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 5, 37, 48)     0           convolution2d_3[0][0]            
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 3, 35, 64)     27712       dropout_1[0][0]                  
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 1, 33, 64)     36928       convolution2d_4[0][0]            
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 2112)          0           convolution2d_5[0][0]            
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 1164)          2459532     flatten_1[0][0]                  
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 100)           116500      dense_1[0][0]                    
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 50)            5050        dense_2[0][0]                    
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 10)            510         dense_3[0][0]                    
____________________________________________________________________________________________________
dense_5 (Dense)                  (None, 1)             11          dense_4[0][0]                    
====================================================================================================
Total params: 2,712,951
Trainable params: 2,712,951
Non-trainable params: 0
```

<figure>
    <img src="https://devblogs.nvidia.com/parallelforall/wp-content/uploads/2016/08/cnn-architecture.png" height="600" width="400" />
    <figcaption text-align: center>Nvidia Model</figcaption>
</figure>



#### 2. Attempts to reduce overfitting in the model

Dropout and Maxpooling seem to be counter intutive to the model and seem to not give as a good performance as the model directly as described. I only used one dropout layer since i wanted to prevent overfitting. I added a **relu** activation to each layer except the last. 

For the last layer the activation used was **tanh** to ensure the steering was between -1 and +1

I trained the model for 20 epochs and then used the keras checkpointing to save only the best version of the weights, optimizing for the validation loss. I then picked the best model generated and used that to drive the car.



#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.
But i did use a initial training rate of 0.0001

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

I drove the car mutiple times and was able to collect the data into mutiple folders each with different driving behavior, 
ie
* **data1**: where i drove slowly and in the center
* **data2**: I drove fast as possible with earatic behavior and recovering
* **data3**: drove on average at 10mph and did a mix of both one and two above

This generated enough data and i was able to run the model as:


```sh
python model.py -d "data1;data2;data3"

```

this collated the data from the folders data1, data2 and data3 into one massive data set.


For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to try image classification model that i had used earlier in the traffic sign classification project but when that did not work as expected I decided to try the mode powerful Nvidia model as per the suggestion the project lesson.

As described in the previous section I collected data into 3 folders each with a different driving behavior.


In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set to about 20 percent of the total data i had collected.

The final step was to run the simulator to see how well the car was driving around track one. But the model wasn't working as expected and right at the end the car ended up driving off the track.

Training for lower number of epochs seemed to do better but not always.

#### 2. Final Model Architecture

The final model architecture is  what was decribed in a prior section. But using check pointing to only get the weights that corresponded to the best validation loss helped pick the best model even when running for large number of epochs.


#### 3. Creation of the Training Set & Training Process

For each line in a csv file I used all three images and corrected the left and right images by adjusting the steering by 0.2 as suggested in the lessons.
Original Data:

<figure>
    <figcaption text-align: center>Center Image</figcaption>
    <img src="https://github.com/NRCar/P3/blob/master/examples/center.jpg"/>
</figure>


<figure>
    <figcaption text-align: center>Left Image</figcaption>
    <img src="https://github.com/NRCar/P3/blob/master/examples/left.jpg"/>
</figure>


<figure>
    <figcaption text-align: center>Right Image</figcaption>
    <img src="https://github.com/NRCar/P3/blob/master/examples/right.jpg"/>
</figure>


To augment the data sat, I also flipped images and angles. I also randomized the brightness of the images to overcome shadows.
I also used image skewing to move the iamge and to simulate curves and the cases where i drove off the road and fixed up the steering with a factor of 0.004 for every pixel i moved.

Since i used a generator to augument data, I had an almost infinite pool of training and validation data.

NOTE: i only augumented the validation data using brightness and flipping but not by skewing.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 13 since the 13/20 epoch was captured by the checkpoint capture.


Augumented Data:

<figure>  
    <figcaption text-align: center>Augmented Image 1 (Brightness and skewed) </figcaption>
    <img src="https://github.com/NRCar/P3/blob/master/examples/augumented_1.jpg"/>    
</figure>


<figure>
    <figcaption text-align: center>Augmented Image 2 (Brightness and skewed)</figcaption>
    <img src="https://github.com/NRCar/P3/blob/master/examples/augumented_2.jpg"/>
</figure>
