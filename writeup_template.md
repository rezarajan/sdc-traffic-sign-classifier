# **Traffic Sign Recognition** 

<!-- ## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

--- -->

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./report_files/images/dataset_composition.png "Dataset Composition"
[image2]: ./report_files/images/pre_processed.png "Pre-Processing"
[image3]: ./report_files/images/augmented.png  "Augmented Image"
[image4]: ./report_files/images/model_c_loss.png "Model C Loss"
[image5]: ./report_files/images/model_c_acc.png "Model C Accuracy"
[image6]: ./report_files/images/model_a_loss.png "Model A Loss"
[image7]: ./report_files/images/model_a_acc.png "Model A Accuracy"
[image8]: ./report_files/images/model_b_loss.png "Model B Loss"
[image9]: ./report_files/images/model_b_acc.png "Model B Accuracy"
[image10]: ./report_files/images/model_d_loss.png "Model D Loss"
[image11]: ./report_files/images/model_d_acc.png "Model D Accuracy"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/rezarajan/sdc-traffic-sign-classifier)

---
## Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

Below are some summary statistics from the training dataset. They are used to gauge how well-suited the dataset is for training, as well as some areas in which the dataset may be improved.

* Number of training examples: 34799
* Number of testing examples: 4410
* Number of testing examples: 12630
* Image data shape: (32, 32, 3)
* Number of unique classes: 43

#### 2. Include an exploratory visualization of the dataset.

Shown below is a visualization of the dataset, grouped by sign type:

![Dataset Composition][image1]

The dataset is observed to have inequitable distributions of sign types, which may introduce biases in the trained model. However, this is only a hypothesis and should be tested.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

### Pre-Processing
Initially the model is constructed without any image modifications, to establish a baseline for performance. Note that this **does not** include testing on the test set, but rather only comparison to the validation set.

It has been observed that the baseline model (LeNet-5 replicated architecture) did not perform to specifications of 93%, and thus other methods have been implemented as follows:

1. CLAHE (Contrast Limited Adaptive Histogram Equalization)
2. Grayscaling
3. Normalization

**CLAHE:**
This method helps correct for any lighting differences between the images in the dataset. This is done to improve consistency when training.

**Grayscaling:**
Initially it had been thought that the model's prediction capability would benefit from color images, since signs differ by color. However, grayscaled images proved to increase the model's accuracy, at least in the LeNet-5-based architecture.

**Normalization:**
To improve model training the grayscaled images are normalized by subtracting each pixel from the dataset's mean, and dividing by the dataset's standard deviation. Note that the mean and standard deviation are taken from all images in the dataset, not per-image.


Below is an example of the pre-processing step:

![Image Pre-Processing][image2]

### Image Augmentation
To test how model accuracy is affected, a second training dataset is created (to be tested on a separate, identical model), in which the images include augmented additions. This is done by applying _affine_ transformations for 10000 images on the original training dataset, which include (between two to four of the following per image):
* Cropping
* Scaling
* Translation
* Rotation
* Shearing 

![Augmentation][image3]

Though the differences are subtle, they should be sufficient to help the the model fit to the true features of the signs. In so doing, the model should become more robust to variations in images. Again, this is simply a hypothesis, and must be tested.

---
## Model Architecture

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The general model used in this project is similar to the LeNet-5 model, but uses:
* 3 convolution layers instead of two (with max pooling)
* 3 fully connected layers
* includes dropout for both convolution and fully connected layers

| Layer         		|     Description	        					| 
|:----------------------|:----------------------------------------------| 
| Input         		| 32x32x1 Grayscale image   					| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 30x30x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 15x15x32				    |
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 13x13x64   |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 6x6x64				    |
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 4x4x6128   |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 2x2x128				    |
| Flatten       		| outputs 512  									|
| Dropout       		| 0.25 rate  									|
| Fully connected		| outputs 120  									|
| RELU					|												|
| Fully connected		| outputs 84  									|
| RELU					|												|
| Dropout       		| 0.5 rate  									|
| Fully connected		| outputs 43  									|
| Softmax				| outputs 43   									|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

A `Model()` class is created, which defines the general stucture of the model, as outlined [above](#2-describe-what-your-final-model-architecture-looks-like-including-model-type-layers-layer-sizes-connectivity-etc-consider-including-a-diagram-andor-table-describing-the-final-model). To train the model, as well as executre any other tasks involving each `Model` class, a `ModelExecutor()` class is created. In this class the following parameters may be defined:
* Convolution Kernel
* Pooling Kernel
* Dropout Rates
* Training Rate
* Epochs

This way, the different models may easily be tested by tuning the mentioned parameters.

The following models have been tested:

| Model Name |Convolution Kernel|Pooling Kernel|Convolution Layer Dropout|Fully Connected Layer Dropout|Training Rate|Epochs|Training Set|Optimizer| 
|:---------- |:-----------------|--------------|-------------------------|-----------------------------|-------------|------|------------|---------| 
|A           |5x5               |2x2           |0.00                     |0.00                         |0.0001       |100   |Modified    |Adam     | 
|B           |5x5               |2x2           |0.25                     |0.50                         |0.0001       |100   |Modified    |Adam     |
|C           |3x3               |2x2           |0.25                     |0.50                         |0.0001       |100   |Modified    |Adam     | 
|D           |3x3               |2x2           |0.25                     |0.50                         |0.0003       |200   |Extended    |Adam     | 

---
## Model Results
#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

After training, the model results are plotted with comparisons of the training and validation accuracies. All models have been tested, as defined in the above table, and results are compared to select the best performer. In the end, ***Model C produced the best results, with the lowest losses and highest accuracies, and it showed robustness against overfitting.*** The results are as follows:

**Model C** performance:

- **Training Accuracy**: 99.99%
- **Validation Accuracy**: 98.40%
- **Test Accuracy**: 97.57%
- **Test Loss:** 0.1290

![Model C Loss][image4]

![Model C Accuracy][image5]


The test accuracy of Model C is 4.57% above the 93% required for this project, and approximately 0.83% below the validation test accuracy. The loss graph also shows how well the model does to reduce overfitting, and this is attributed to the dropout layers. **Thus Model C is accepted as the final model.**

For comparison purposes, the results of the other models are discussed below:
**Model A** performance:

- **Training Accuracy**: 99.99%
- **Validation Accuracy**: 95.70%

![Model A Loss][image6]

![Model A Accuracy][image7]

**Model B** performance:

- **Training Accuracy**: 99.00%
- **Validation Accuracy**: 97.00%

![Model B Loss][image8]

![Model B Accuracy][image9]

**Model D** performance:

- **Training Accuracy**: 99.99%
- **Validation Accuracy**: 97.90%
- **Test Accuracy**: 97.19%
- **Test Loss:** 0.1666

![Model D Loss][image10]

![Model D Accuracy][image11]

**Note:** Tests are only performed _once_ for each model, after each model is trained. This is done to reduce bias when tuning each model.

### General Model Remarks
It is clear from the model plots that introducing dropout significantly boosts the performance of the model. As compared to Model A, which has no dropout, the other models do not appear to overfit the training set, even with large epochs. Most notbale is that Model A does very much overfit the training set quite early in the training process, and therefore dropout is a necessity for improving model performance.

Furthermore, Model C shows the best performance, at 98.4% validation accuracy, whereas Model D shows the second best performance at 97.9% accuracy. There difference in these two models primarily lies in the training dataset used, where Model D uses an extended dataset with augmented images. The learning rate for Model D is also higher, at 0.0003 rather than the 0.0001 used for Model C. Further investigation into the performance of these two models may be necessary, since the performances are similar, but Model D has exposure to different data and may therefore, be more general.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


