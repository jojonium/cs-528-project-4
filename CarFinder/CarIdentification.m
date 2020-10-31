%% Real-time Car Identification Using Image Data
% Image classification involves determining if an image contains some 
% specific object, feature, or activity. The goal of this example is to
% provide a strategy to construct a classifier that can automatically 
% detect which car we are looking at using streaming images from a webcam 
% feed.
% Copyright (c) 2015, MathWorks, Inc.

%% Capture Training Images
% Use |captureTrainingImages| function to capture training images for
% objects around you. Here is an example of how you would capture images of
% four different objects:  CarType1, CarType2, CarType3 ,CarType4 and save 
% it in a folder called trainingImages
captureTrainingImages('trainingImages', 'CarType1')
captureTrainingImages('trainingImages', 'CarType2')
captureTrainingImages('trainingImages', 'CarType3')
captureTrainingImages('trainingImages', 'CarType4')

%% Load image data
imset = imageSet('trainingImages','recursive'); 

%% Pre-process Training Data: *Feature Extraction*
% Requires: Computer Vision System Toolbox

% Create a bag-of-features from the Car image database
bag = bagOfFeatures(imset,'VocabularySize',200,...
    'PointSelection','Detector');

% Encode the images as new features
imagefeatures = encode(bag,imset);

%% Create a Table using the encoded features
CarData         = array2table(imagefeatures);
CarData.carType = getImageLabels(imset);

%% Use the new features to train a model and assess its performance
classificationLearner

%% Test Trained Model
CarFinderLive(trainedClassifier,bag)

%% Compare multiple classifiers
CarFinderLiveCompare(trainedClassifier,trainedClassifier1,bag)
