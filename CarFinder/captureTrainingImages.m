function captureTrainingImages(MainFolderName, ObjectFolderName,delay)
% CarFinderLive uses a trained classifier and a featureExtractor function
% to classify streaming webcam images 
% Copyright (c) 2015, MathWorks, Inc.

%% Capture delay for automatic capturing
if nargin < 3
    delay = 0.1;
end

%% Set up Webcam
% Note: webcam required a hardware support package.
wcam = webcam;

%% Create folder with label name
fileLocation = fullfile(MainFolderName,ObjectFolderName);
if ~exist(ObjectFolderName,'file')
   mkdir(fileLocation)
end

%% Take pictures
nImages = 1;
fig = figure('Name','Capture Training Images','NumberTitle','off');
ax = axes('Parent',fig);
try
    while ishandle(fig)
        pause(delay);
        img = snapshot(wcam);
        imshow(img,'Parent',ax);
        imwrite(img, fullfile(fileLocation,sprintf('image%d.png', nImages)));
        title(sprintf('Number of images captures: %3.0f',nImages))
        nImages = nImages+1;    
    end
catch err
end

%% Display Info
nImages = size(ls(fullfile(fileLocation,'*.png')),1);
disp('---------Training Images Saved---------')
disp(['Image Label: ',ObjectFolderName])
disp(['Number of images saved: ',num2str(nImages)])
disp(['Image location: ',fullfile(pwd,fileLocation)])

