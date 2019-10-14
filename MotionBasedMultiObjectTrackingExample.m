function MotionBasedMultiObjectTrackingExample()
% Create System objects used for reading video, detecting moving objects,
% and displaying the results.

obj = setupSystemObjects();

tracks = initializeTracks(); % Create an empty array of tracks.

nextId = 1; % ID of the next track

% Detect moving objects, and track them across video frames.
while ~isDone(obj.reader)
    frame = readFrame();
    [centroids, bboxes, mask] = detectObjects(frame);
    predictNewLocationsOfTracks();
    [assignments, unassignedTracks, unassignedDetections] = ...
        detectionToTrackAssignment();

    updateAssignedTracks();
    updateUnassignedTracks();
    deleteLostTracks();
    createNewTracks();

    displayTrackingResults();
end


    function obj = setupSystemObjects()
        % Initialize Video I/O
        % Create objects for reading a video from a file, drawing the tracked
        % objects in each frame, and playing the video.

        % Create a video file reader.
        obj.reader = vision.VideoFileReader('StillHuman.mp4');

        % Create two video players, one to display the video,
        % and one to display the foreground mask.
        obj.maskPlayer = vision.VideoPlayer('Position', [740, 400, 700, 400]);
        obj.videoPlayer = vision.VideoPlayer('Position', [20, 400, 700, 400]);

        % Create System objects for foreground detection and blob analysis

        % The foreground detector is used to segment moving objects from
        % the background. It outputs a binary mask, where the pixel value
        % of 1 corresponds to the foreground and the value of 0 corresponds
        % to the background.

        obj.detector = vision.ForegroundDetector('NumGaussians', 3, ...
            'NumTrainingFrames', 40, 'MinimumBackgroundRatio', 0.7);

        % Connected groups of foreground pixels are likely to correspond to moving
        % objects.  The blob analysis System object is used to find such groups
        % (called 'blobs' or 'connected components'), and compute their
        % characteristics, such as area, centroid, and the bounding box.

        obj.blobAnalyser = vision.BlobAnalysis('BoundingBoxOutputPort', true, ...
            'AreaOutputPort', true, 'CentroidOutputPort', true, ...
            'MinimumBlobArea', 400);
    end

    function tracks = initializeTracks()
        % create an empty array of tracks
        tracks = struct(...
            'id', {}, ...
            'bbox', {}, ...
            'kalmanFilter', {}, ...
            'age', {}, ...
            'totalVisibleCount', {}, ...
            'consecutiveInvisibleCount', {});
    end


    function frame = readFrame()
        frame = obj.reader.step();
    end


    function [centroids, bboxes, mask] = detectObjects(frame)

        % Detect foreground.
        mask = obj.detector.step(frame);

        % Apply morphological operations to remove noise and fill in holes.
        mask = imopen(mask, strel('rectangle', [3,3]));
        mask = imclose(mask, strel('rectangle', [15, 15]));
        mask = imfill(mask, 'holes');

        % Perform blob analysis to find connected components.
        [~, centroids, bboxes] = obj.blobAnalyser.step(mask);
    end


    function predictNewLocationsOfTracks()
        for i = 1:length(tracks)
            bbox = tracks(i).bbox;

            % Predict the current location of the track.
            predictedCentroid = predict(tracks(i).kalmanFilter);

            % Shift the bounding box so that its center is at
            % the predicted location.
            predictedCentroid = int32(predictedCentroid) - bbox(3:4) / 2;
            tracks(i).bbox = [predictedCentroid, bbox(3:4)];
        end
    end

    function [assignments, unassignedTracks, unassignedDetections] = ...
            detectionToTrackAssignment()    
        % Above function header returns output from other functions
        nTracks = length(tracks); 
        % Local variable for tracking
        nDetections = size(centroids, 1); 
        % Number of detections per tracked per centroid       
        cost = zeros(nTracks, nDetections);
        % Compute the cost of assigning each detection to each track.
        % Cost is negative log likelihood of a detection eqaul to a track
        for i = 1:nTracks
            cost(i, :) = distance(tracks(i).kalmanFilter, centroids);
        end
        % For loop which uses kalmanFilter and loops through each track
        costOfNonAssignment = 20;
        % Value of non assignment depends on the range of values returned 
        % by the distance method of the vision.KalmanFilter. Value has been
        % tuned experimentally.
        [assignments, unassignedTracks, unassignedDetections] = ...
            assignDetectionsToTracks(cost, costOfNonAssignment);
        % Assigning the matrix with the values of the tack
    end

    function updateAssignedTracks()
        numAssignedTracks = size(assignments, 1); 
        % Variable for the count
        % and age of the track by 1
        for i = 1:numAssignedTracks
            trackIdx = assignments(i, 1); 
            % Assigns track ID
            detectionIdx = assignments(i, 2); 
            % Assigns detection ID
            centroid = centroids(detectionIdx, :); 
            % Assigns centroid to detection ID
            bbox = bboxes(detectionIdx, :);
            % Correct the estimate of the object's location
            % using the new detection.
            correct(tracks(trackIdx).kalmanFilter, centroid);
            % Replace predicted bounding box with detected bounding box.
            tracks(trackIdx).bbox = bbox;
            % Update track's age.
            tracks(trackIdx).age = tracks(trackIdx).age + 1;
            % Update visibility.
            tracks(trackIdx).totalVisibleCount = ...
                tracks(trackIdx).totalVisibleCount + 1;
            tracks(trackIdx).consecutiveInvisibleCount = 0;
            % Count for the number of visible tracks
        end
    end

    function updateUnassignedTracks()
        for i = 1:length(unassignedTracks)
        % Loops through unassigned tracks
            ind = unassignedTracks(i);
            % Variable for looping within unassignedTracks
            tracks(ind).age = tracks(ind).age + 1;
            % Increase age by 1
            tracks(ind).consecutiveInvisibleCount = ...
                tracks(ind).consecutiveInvisibleCount + 1;
            % Increase consecutive invisible count by 1
        end
    end

    function deleteLostTracks()
        if isempty(tracks)
            return;
        end
        % If tracks list is empty ignore the code underneath
        invisibleForTooLong = 20;
        % Variable for limit of track being invisible
        ageThreshold = 8;
        %Variable for age        
        ages = [tracks(:).age];
        totalVisibleCounts = [tracks(:).totalVisibleCount];
        visibility = totalVisibleCounts ./ ages;
        % Compute the fraction of the track's age for which it was visible           
        lostInds = (ages < ageThreshold & visibility < 0.6) | ...
            [tracks(:).consecutiveInvisibleCount] >= invisibleForTooLong;
        % Find the indices of 'lost' tracks by using assigned variables                
        tracks = tracks(~lostInds);
        % Delete lost tracks.
    end

    function createNewTracks()
        centroids = centroids(unassignedDetections, :);
        bboxes = bboxes(unassignedDetections, :);
        % Variables for unassigned detections
        for i = 1:size(centroids, 1)
        % Loop through centroids
            centroid = centroids(i,:);
            bbox = bboxes(i, :);
            % looping through unassigned detections variable         
            kalmanFilter = configureKalmanFilter('ConstantVelocity', ...
                centroid, [200, 50], [100, 25], 100);
            % Create a Kalman filter object.           
            newTrack = struct(...
                'id', nextId, ...
                'bbox', bbox, ...
                'kalmanFilter', kalmanFilter, ...
                'age', 1, ...
                'totalVisibleCount', 1, ...
                'consecutiveInvisibleCount', 0);
            % Create a new track.
            tracks(end + 1) = newTrack;
            % Add it to the array of tracks.
            nextId = nextId + 1;
            % Increment the next id.
        end
    end

    function displayTrackingResults()        
        frame = im2uint8(frame);
        mask = uint8(repmat(mask, [1, 1, 3])) .* 255;
        % Convert the frame and the mask to uint8 RGB
        minVisibleCount = 8; % Variable for min visible count
        if ~isempty(tracks) % if condition true when tracks is not a field           
            reliableTrackInds = ...
                [tracks(:).totalVisibleCount] > minVisibleCount;
            reliableTracks = tracks(reliableTrackInds);
            % Noisy detections tend to result in short-lived tracks.
            % Only display tracks that have been visible for more than
            % a minimum number of frames.

            % Display the objects. If an object has not been detected
            % in this frame, display its predicted bounding box.
            if ~isempty(reliableTracks)
                % Get bounding boxes.
                bboxes = cat(1, reliableTracks.bbox);
                ids = int32([reliableTracks(:).id]);% Displays ID's                
                labels = cellstr(int2str(ids'));
                % Create labels for objects indicating the ones for
                % which we display the predicted rather than the actual
                % location
                predictedTrackInds = ...
                    [reliableTracks(:).consecutiveInvisibleCount] > 0;
                isPredicted = cell(size(labels));
                isPredicted(predictedTrackInds) = {' predicted'};
                labels = strcat(labels, isPredicted);               
                frame = insertObjectAnnotation(frame, 'rectangle', ...
                    bboxes, labels); % Draw the objects on the frame      
                mask = insertObjectAnnotation(mask, 'rectangle', ...
                    bboxes, labels);% Draw the objects on the mask
            end
        end
        % Display the mask and the frame.
        obj.maskPlayer.step(mask);
        obj.videoPlayer.step(frame);
    end
end
