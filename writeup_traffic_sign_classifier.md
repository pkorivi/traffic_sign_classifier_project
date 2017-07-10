**Traffic Sign Recognition** 


**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./traffic_data.png "traffic_data"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./web_img/rsz_01.jpg "Traffic Sign 1"
[image5]: ./web_img/rsz_02.jpg "Traffic Sign 2"
[image6]: ./web_img/rsz_03.jpg "Traffic Sign 3"
[image7]: ./web_img/rsz_04.jpg "Traffic Sign 4"
[image8]: ./web_img/rsz_05.jpg "Traffic Sign 5"

[image9]: ./web_img/25.png "Traffic Sign 1"
[image10]: ./web_img/30.png "Traffic Sign 2"
[image11]: ./web_img/8.png "Traffic Sign 3"
[image12]: ./web_img/27.png "Traffic Sign 4"
[image13]: ./web_img/40.png "Traffic Sign 5"

---
**Detailed summary**

### 1. Here is a link to my [project code](https://github.com/pkorivi/traffic_sign_classifier_project/blob/master/Traffic_Sign_Classifier.ipynb)

### 2. Data Set Summary & Exploration

Summary Statistics of the traffic sign data used are as follows:
* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

Here is an exploratory visualization of the training data set.

![alt text][image1]

**Design and Test a Model Architecture**

### 3. Preprocessing:

As a first step, I have normlized the taffic sign images, as normalized data because the training results are better with normalized data by converging better. 

Grayscaling the images did not fetch any better results. It can be because the traffic signs are of different color and it can help in finding results more accurately. 


### 4. Architecture


# 1. Pipeline.

The Pipeline is divided into following steps
* Extract the training, vladidation and testing datasets from the pickled files
* Normalize the input(i.e image data)
* Cross check for the image dimensions, number of images in each set etc. 
* Shuffle the training data
* Train the system with Lenet Architecture designed for this task. 
* Validate it with Validation set.
* Test the results with the trained parameters. 

# 2 Neural Network Architecture:

This network is combination of convolutional and fully connected neural networks inspired from Lenet Model. 
* It has 3 steps of convolution in below 3 levels. 
  * Conv 1 one reducing the 32x32x3 data to 28x28x12 
  * relu activation function and 2x2 average filter added.
  * Conv 2 is same as above with 5x5 kernel and 2x2 avg filter modifying the next input to 5x5x32.
  * Conv 3 is 1x1 convolution added to create non linear affect for the network. 
* Then the data is flattened out and passed through 5 levels of fully connected networks reducing the dimensions from 800->500->250->120->84->43 
* Relu activation function is added.
* Dropout is added for regularization and to avoid overfitting at all levels of fully connected network.


### 5. Training the Model

Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Lenet architecture Deep neural network model. Cross entropy is calculated and adam optimizer is used to minimise the loss. 

I have tried various batchsizes, learning rates etc and the above values tended to give me best results for me.

batch size = 128.
learning rate = 0.001.
zero mean and 0.1 standard deviation is used in weight calculation. 

Early termination is implemented to avoid over fitting and around 30~35 Epochs seems to be a good training time. Increasing or decreasing the learning rate tended to destabilize the model.  

### 6.Results and Architecture decision summary 
My final model results were:
* training set accuracy of 99.5
* validation set accuracy of 94.5-95.6 
* test set accuracy of 93.5

* The architecture is inspired from Lenet model with few modifications. This is inspired from the results of using the model in identifying the nmist data. This model is able to extract various features from the images and the traffic sign classification is a similar problem. 
* There were no big initial issues, the validation accuracies were less and some modifications were needed to do that.
* I have learnt that depper the network better are the possibilities to extract features and I have followed this approach to add few more layers to the original Lenet architecture. 
* Mechanism to avoid over fitting is implemented through early termination.
* Not many parameters were tuned as the standard ones provided the best results for me. 
* As there are various types of features that could be extracted in terms of color, shape a convolutional model appeared to be a good choice.
 

### 7. Test a Model on New Images

# 1.Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The images 3 and 5 should be tough to classify as the 3rd image has a different background color(yellow) for the text compared to standard white and 5th image has some watermarks on it which can be similar to dust or reflections. The other images should be fairly possible to estimate. 

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Road Work      		| Road Work  									| 
| Snow     			| Snow										|
| 120Kmph					| 70kmph											|
| Pedestrians	      		| Snow					 				|
| Roundabout			| Round about      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

The various predictions for each image are listed in the following charts

![alt text][image9] ![alt text][image10] ![alt text][image11] 
![alt text][image12] ![alt text][image13]

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
