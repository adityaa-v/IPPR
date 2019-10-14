%%Read Video
videoReader = vision.VideoFileReader('C:\Users\adzar\Downloads\StillHuman.mp4');
%%Create Video Player
videoPlayer = vision.VideoPlayer;
%%Create Foreground Detector  (Background Subtraction)
foregroundDetector = vision.ForegroundDetector('NumGaussians', 3,'NumTrainingFrames', 100);
%%Run on first 100 frames to learn background
for i = 1:75
    videoFrame = step(videoReader);
    foreground = step(foregroundDetector,videoFrame);
   % figure, imshow(I); 
end
% display 75th frame and foreground image  
while  ~isDone(videoReader)
    %Get the next frame
    videoFrame = step(videoReader);
    %Detect foreground pixels
    foreground = step(foregroundDetector,videoFrame);     
    cc = bwconncomp(foreground);
    stats = regionprops(cc, 'Area');
    idx = find([stats.Area] > 50);
    foreground2 = ismember(labelmatrix(cc), idx);  
s  = regionprops(foreground2, 'centroid');
    centroids = cat(1, s.Centroid);
    mycentroid=[00.00,00.00];
    mycentroid = cat(1,centroids,mycentroid);
    %figure, imshow(foreground2);   
end