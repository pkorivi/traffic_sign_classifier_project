# **Finding Lane Lines on the Road**

---

**Finding Lane Lines on the Road**

This document explains about the steps involved in Lane detection.


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

The Pipeline is divided into following steps
* Conversion of Image to Gray Scale.
* Blurring image using Gaussian Blur function.
* Canny Edge detection
* Extracting a region of interest where there is high possibility of finding lane. Luckily for this project it is ok to look at the road region which will be in bottom half of the pixels.
* Finding Hough lines in the region of interest.
* Extracting the information from the Houh lines and mark the lanes in solid Red color.


In order to draw a single line on the left and right lanes, I modified the draw_lines() function in below ways

* For every hough line check the slope, if the slope is positive collect these points into a global list for left lane else into a list for right lane.
* Polyfit function is used with all the points extracted above to get a liner equation of 1st order.
* Find the lower and top intercetps for x using this equation and as y coordinate s fixed (image dimensions)
* Draw the line with these two points onto the image for lane detection.
* To counter false positives etc, a simple check mechanism is introduced in which the current slope is compared against the mean slope and if it varies for more than 20% then it is rejected and old line is drawn on the image.


Here is an image after the final lane_detection

[image1]: ./examples/final.jpg "Final Image"


### 2. Identify potential shortcomings with your current pipeline
* The feedback mechanism to counter false positives is not really robust, it works here in simple situations to counter rapid chnages in lane detection.
* The lane differentiation mechanism is especially not good. It is a problem when the lanes are curved.

### 3. Suggest possible improvements to your pipeline

* The check mechanism need to be improved such that it only considers a limited set of previous images based on vehicle speed, curvatures of the lane etc.

* There is a problem that if the lane is really curved the slope differentiation function will not work and something better need to be devised. I tried few approaches but nothing worked.
