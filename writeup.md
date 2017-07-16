##Writeup Template

###Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_car_example.png
[image3]: ./examples/HOG_not_car_example.png
[image4]: ./examples/color_spaces.png
[image5]: ./examples/HOG_normalization.png
[image6]: ./examples/scale_pyramid.png
[image7]: ./examples/sliding_windows.png
[image8]: ./examples/sliding_window.png
[image9]: ./examples/bboxes_and_heat.png
[image10]: ./examples/labels_map.png
[image11]: ./examples/output_bboxes.png
[video12]: ./project_video.mp4

###Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/513/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how you extracted HOG features from the training images.

The code for this step is contained in the module `feature_extraction.py`. The module contains one class for each feature extraction method:
* SpatialFeatureExtractor
* HistogramFeatureExtractor
* HogFeatureExtractor

Each class instance represents an extractor with a given parameter set, which is used for feature extraction during training as well as during prediction. Different extractor instances could be also chained to create a composite feature from multiple extractors. 

The classifier training code is located in the main section of the main module `vehicle_detection.py`.
For the training, I started by reading in all the `vehicle` and `non-vehicle` images from the project resources [vehicles.zip](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicles.zip](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip).  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![Car and Not-Car example][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YUV` color space and HOG parameters of `orientations=11`, `pixels_per_cell=(16, 16)` and `cells_per_block=(2, 2)`.

##### Car HOG

![YUV HOG for car][image2]

##### Not-Car HOG

![YUV HOG for not car][image3]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and compared the validation accuracy of the classifier as well as the visual performance on the test image set. 

#####Color channels
I did not make any good experiences with adding plain color histogram or spatial bin features into the equation - both features seem to be vastly inferior to HOGs. I did however notice a significant improve in detection performance if the HOGs are calculated over all images channels. The decision to use HOG over all channels also makes the choice of the color space less relevant. If chosen only one channel, the Y channel of YUV space delivers the best detection performance with 98.5% in precision and only sparse false positives in a few video frames. It makes sense, when looking at sample pictures in 3D color space:
 
 For YUV ,the pixel cloud is strongly aligned to the Y axis, while in RGB space, the pixel cloud is less directed and diagonal to the axes.

![YUV HOG for not car][image4]

#####Orientation bin counts
The orientation bin counts from 9 to 11 both performed very well (~99,2% accuracy on test set and no false positives on test images). With 8 bins or fewer, the accuracy of detecting begins to fall as the car edge directions became not distinctive enough. For orientation bin sizes larger than 11, the detection performance stay constant till about 15 and then slightly decreases as bin size grows larger. The reason for the quite large bin size window is probably the capability of the hog implementation which also takes weighted directions from neighbouring bins into account.

#####OpenCV parameters
The OpenCV HOG implementation is one of the worst documented interfaces I have ever encountered. It is e.g. important to set parameter `_winSigma` to `-1` or else OpenCv will apply a gaussian filter to the image before calculating the HOG which will have a significant negative impact in the detection performance. Also `_gamma_correction` has to be set to `true` unless the image has been normalized manually beforehand. `_blockStride` should be set to half of block size. Without these adjustments, the OpenCV HOG descriptor will deliver worse detection performance than the according skimage hog. Regardless of HOG implementation used, the best block normalization method is 'L2Hys' as is also supported in the [video by Navneet Dalal](https://www.youtube.com/watch?v=7S5qXET179I):

![HOG normalization][image5]


#####Block size & cell size
I was unable to test the parameters (6x6 cell size, 3x3 block size) that Navneet Dalal used for his person recognition implementation, because this did exceed the available RAM on my machine. Regardless, I think, these parameters would have not worked well. He relates the test results with cell size to the width of limbs/arms of persons, but that unit is quite uninteresting for our car detection. I settled for a much larger cell size (16x16) and accordingly for a smaller block size (2x2). Some OpenCV documentation also seems to claim that 16x16 is would be the only supported cell size, but this is not true (I tried different sizes successfully). 16x16 seems to perform better than 8x8, although details like the outline of the license plate are lost. The explanation is, that the gradients still cover the outline of the car and allow for better generalization. The block size does not seem to play such an important role for our vehicles, as the algorithm reaches about the same detection precision even with just one global block (4x4).

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The classifier training code is located in the main section of the main module `vehicle_detection.py`. I used `sklearn.model_selection.GridSearchCV` to find the optimal parameters for training the SVM and it turned out the output of the rbf kernel was slightly better than the linear kernel. I also tried to find out the best parameters for `C`, `gamma` and `max_iter` parameters using the grid search, but it turned out that the default parameters were just fine and tweaks to them just led to "improvements" just within normal precision variance (~0.2%). 

As suggested in the project, I used `sklearn.preprocessing.StandardScaler` to normalize the feature vectors. Before training, I performed the standard actions:
 * Shuffling the feature sets
 * Truncated feature sets to the same size (so there would be no bias towards either class) 
 * Split feature sets into 80/20 training and test sets

What I did not do:
* Augmenting the feature sets (many images were already augmented by e.g. flipping)
* Devising a train/test split that avoids having nearly identical images (too much effort)

After the training, I persisted the SVM as well as the scale class, so new images and videos could be predicted without going through another long training process. 


###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The code for the sliding window search is located in the module `window_search.py` within the functions
* `get_car_search_windows()`
* `get_search_windows()`

Basically all search windows have a constant size of 64x64, but are calculated over differently scaled versions of the original image (as proposed in the [video by Navneet Dalal](https://www.youtube.com/watch?v=7S5qXET179I)):

![Slace pyramid for window search][image6]


This has the advantage, that the HOGs only need to be calculated once for every scale. This is performance-critical, especially as I  have chose quite a large window overlap of 75% in x and y direction. 
 
The windows of the different scales are aligned in such a way, that only the area with a high probability of car occurance is covered. Furthermore, the perspective is taken into account in such a way that windows near the bottom of the image cover more space on the camera plane than the windows near the vanishing point.

In total, there a six scales with windows. Translated to 1.0 scale, the windows look like following:
![Sliding windows][image7]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on five scales using all YUV channel HOG features, which provided a very nice result. But as my classifier detected the white car only with view windows, I decided to add 9 white car samples of varying sizes and offsets from the test images into the training data set. 

Here are some example images with the classifier that was trained on that slightly extended training data set:
![alt text][image8]

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_videos/project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.


I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heat map and then thresholded that map to identify vehicle positions.  

I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heat map.  I then assumed each blob corresponded to a vehicle. The according code is located within the module `window_search.py` inside the class `HeatMap`. 

I have chosen the following parameters for the heat map:
* 15% heat for each detected window
* 20% threshold after global heat normalization
* Integration of heat map over 10 frames

Here's an example result showing a heat map frame of the [test video](.test_videos/test_video.mp4). I've also recorded the full heat map sequence into the following [heat map video](./output_videos/test_video_heat.mp4).
![Heatmap example][image9]


Also here's the output of `scipy.ndimage.measurements.label()` output from the integrated heat map over 10 frames:
 ![Labels example][image10]
 
And last but not least here are the bounding boxes overlaid on the last frame of video (together with lane recognition):

![Final output][image11]

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

#####Implementation issues
There main issue during implementation was the performance of the algorithm. Even though I made compromises regarding the number of scales and the window overlap and even though I used `cv2.HogDescriptor` to calculate the HOGs only once for each scale, the algorithm is still taking about 1s per frame which makes it unsuited for real time applications. Maybe using single channel HOG and further reducing the window count and overlap would have helped.

#####Weak points
Situations in which the algorithm will struggle:
* Partially visible/covered vehicle (e.g. parking vehicles)
* Vehicles in unlearnt angles (front view, side view)
* Unlearnt vehicle shapes (e.g. trucks, beatles)

Right now, there are a lot of images in the data set which refer to the very same car occurance. In order to improve the situation, the data set could be extended by more different car models, car colors and car angles.

One missing point in the solution is also the capability to track individual cars. E.g. in the current implementation, the labels flip when the black and white car pass each other. An extended solution could calculate something like a feature signature from the HOG inside the matching windows of each individually detected car. These signatures could be tracked and updated over successive frames and be stored inside a "surrounding car" short-term memory. Given also the trajectory for each vehicle, this would allow to even track vehicles while hidden behind obstacles.

Another weak point is the missing depth perception and the incapability to generate tight 3-dimensional bounding boxes. I have no idea how to improve this though other than using additional input devices.

Other another note, this method of vehicle detection seems to have some similarity to an inception type neural network:
* The search windows on different scales remind me of parallel convolutions with different kernel sizes
* The calculation of the gradient directions could be seen as a special type of non-linearity
* The strongest orientation histogram binning is similar to max pooling
* The SVM would represent the classifier layer
 
 