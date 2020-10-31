function CarFinderLiveCompare(trainedClassifier1,trainedClassifier2,bag)
% CarFinderLive uses a trained classifier and a featureExtractor function
% to classify streaming webcam images 
% Copyright (c) 2015, MathWorks, Inc.

[fig, ax] = figureSetup;

% Start webcam
wcam = webcam;

% Run live car detection
while ishandle(fig)
    % Step 1: Get Next Frame
    img = snapshot(wcam);
    grayimg = rgb2gray(img);
    
    % Step 2: Extract Features
	imagefeatures = double(encode(bag,grayimg));
    
    % Step 3: Predict car using extracted features
    imagepred1 = predict(trainedClassifier1,imagefeatures);
	imagepred2 = predict(trainedClassifier2,imagefeatures);
    
    % Step 4: Plot Results
    try
        PredName1 = [getClassifierName(trainedClassifier1),':',upper(char(imagepred1))];
        PredName2 = [getClassifierName(trainedClassifier2),':',upper(char(imagepred2))];
        
        im = insertText(img,[640,480],PredName1,...
            'AnchorPoint','RightBottom','FontSize',30,'BoxColor','Green',...
            'BoxOpacity',0.4);  
        imshow(insertText(im,[1,1],PredName2,...
            'AnchorPoint','LeftTop','FontSize',30,'BoxColor','Red',...
            'BoxOpacity',0.4),'Parent',ax);  
        title('Compare Classifiers')
    catch err
    end
    drawnow
end 

function cname = getClassifierName(trainedClassifier)
% getClassifierName extracts name of the classifier from a trained model
    cname = class(trainedClassifier);
	if isa(trainedClassifier,'ClassificationECOC')
        cname = 'SVM';
    end
    if isa(trainedClassifier,'ClassificationKNN')
        cname = 'KNN';
    end
    pos = strfind(cname,'.');
    if ~isempty(pos)
      cname = cname(pos(end)+1:end);
    end

function [fig, ax] = figureSetup
% figureSetup sets up figure window for webcam feed 
    warning('off','images:imshow:magnificationMustBeFitForDockedFigure')
    fig = figure('Name','Car Finder Go!','NumberTitle','off');
    ax = axes;
