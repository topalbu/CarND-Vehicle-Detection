# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


In this project, your goal is to write a software pipeline to detect vehicles in a video (start with the test_video.mp4 and later implement on full project_video.mp4), but the main output or product we want you to create is a detailed writeup of the project.  Check out the [writeup template](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup.  

Creating a great writeup:
---
A great writeup should include the rubric points as well as your description of how you addressed each point.  You should include a detailed description of the code used in each step (with line-number references and code snippets where necessary), and links to other supporting documents or external references.  You should include images in your writeup to demonstrate how your code works with examples.  

All that said, please be concise!  We're not looking for you to write a book here, just a brief description of how you passed each rubric point, and references to the relevant code :). 

You can submit your writeup in markdown or use another method and submit a pdf instead.

# Feature Extraction and Selection. 
---
To train a good classifier fot a given task first  we need to extract feature from the data. For this project we use histograms of colors, spatial binning of color and Histogram of Oriented Gradient (HOG)
 to extract the feature.



1.) Spatial Binning of Color
Firstly, we can use raw pixel values in our feature vector to search vehicles. However to include raw pixel values
from all 3 channel will put a huge burden on the system. Therefore we decrease the resolution and we see that it is still 
a good information source Therefore **cv2.resize()** function is used to decrease the resolution to 16x16 then we applied the **ravel()** function on resized image to get one dimensional feature vector. 


2. ) Histograms of Color 
Color can be a good indicator to identify vehicles and non vehicles images. For that purpose first the image is converted 
to **YCrCb** and we calculate histograms of pixel intensity of all 3 channels with histogram bin size 32. 

3. ) Histogram of Oriented Gradients (HOG) 
Histogram of Oriented Gradients is commonly used feature descriptor in image processing for object detection. To extract the HOG features 
hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3), block_norm='L1', visualise=False, transform_sqrt=False, feature_vector=True, normalise=None)
method is used form skimage.feature package. 


