# **Finding Lane Lines on the Road**

---

**Finding Lane Lines on the Road**

This document explains about the steps involved in Traffic sign classification using Deep Learning.



### Reflection

### 1. Pipeline.

The Pipeline is divided into following steps
* Extract the training, vladidation and testing datasets from the pickled files
* Normalize the input(i.e image data)
* Cross check for the image dimensions, number of images in each set etc. 
* Shuffle the training data
* Train the system with Lenet Architecture designed for this task. 
* Validate it with Validation set.
* Test the results with the trained parameters. 

### 1.1 Neural Network Architecture:

This network is combination of convolutional and fully connected neural networks. 
* It has 3 steps of convolution in below 3 levels. 
  * Convl one reducing the 32x32x3 data to 28x28x12 
  * relu activation function and 2x2 average filter added.
  * Conv 2 is same as above modifying the next input to 5x5x32
  * Conv 3 is 1x1 convolution added to create non linear affect for the network. 
* Then the data is flattened out and passed through 4 levels of fully connected networks reducing the dimensions from 800->500->250->120->84->43 
* Relu activation function is added.
* Dropout is added for regularization at all levels except last one. 


### 2. Identify potential shortcomings with your current pipeline
* The hyper parametrs are tough to choose and after multiple trails also I couldnt go much further. 
* I assume this performance could be achieved with a smaller network and better parameters. 


### 3. Suggest possible improvements to your pipeline

* Find better ways to tune parameters and ways to arrange the network. 
