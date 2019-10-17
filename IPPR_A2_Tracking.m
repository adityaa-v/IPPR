function IPPR_A2_Tracking()

% Main function: calls upon nested functions to track moving objects 
% in video and display results via bounding boxes and binary mask

% Instructions: 
% Before running this script, install the "Computer Vision Toolbox" add-on

obj = setupSystemObjects();  % create System objects

tracks = initialiseTracks(); % create an empty array of tracks

nextId = 1; % next track ID intiialised value 

while ~isDone(obj.reader)
    
    % (loop) while each video frame is read, 
    % 1. detect objects in frame
    % 2. predict new location of assigned objects
    % 3. assign newly detected moving objects to a track 
    % 4. update location of assigned tracks 
    % 5. update unassigned tracks 6. delete
    % 6. tracks with no detection 
    % 7. create new tracks 
    % 8. display results via bounding boxes and binary mask
    % end loop when video runs out of frames to read

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
        
        % load the video using a video reader object
        % change name to desired video to analyse in single quotation marks
        % e.g. 'newVideo.mp4' 
        obj.reader = vision.VideoFileReader('StillHuman.mp4');
        
        % create video player - displays the foreground mask (binary) 
        obj.maskPlayer = vision.VideoPlayer('Position', [740, 400, 700, 400]);
       
        % create player - displays the video normally with bounding boxes 
        obj.videoPlayer = vision.VideoPlayer('Position', [20, 400, 700, 400]);

        % create detector (foreground detection) - distinguishes moving
        % objects from the background
        obj.detector = vision.ForegroundDetector('NumGaussians', 3, ...
            'NumTrainingFrames', 40, 'MinimumBackgroundRatio', 0.7);
        
        % create detector (blob analysis) - finds connected groups of
        % foreground pixels that are likely to correspond to moving objects
        % and compute their characteristics.
        obj.blobAnalyser = vision.BlobAnalysis('BoundingBoxOutputPort', true, ...
            'AreaOutputPort', true, 'CentroidOutputPort', true, ...
            'MinimumBlobArea', 400);
    end

    function tracks = initialiseTracks() 
    
        % create an empty array of tracks with fields:
        %   - ID 
        %   - Bbox
        %   - Kalman filter
        %   - age
        %   - totalVisibleCount
        %   - consecutiveInvisibleCount
        
        tracks = struct(...
            'id', {}, ...
            'bbox', {}, ...
            'kalmanFilter', {}, ...
            'age', {}, ...
            'totalVisibleCount', {}, ...
            'consecutiveInvisibleCount', {});
        
    end

    function frame = readFrame()
        
        % run obj.reader -> reads each frame in video
        frame = obj.reader.step();
        
    end

    function [centroids, bboxes, mask] = detectObjects(frame)

        % run foreground detector in each frame
        mask = obj.detector.step(frame);

        % apply morphological filters 
        mask = imopen(mask, strel('rectangle', [3,3]));
        mask = imclose(mask, strel('rectangle', [15, 15]));
        mask = imfill(mask, 'holes');

        % run blob analysis algorithm  
        [~, centroids, bboxes] = obj.blobAnalyser.step(mask);
    end

    function predictNewLocationsOfTracks()
        
        % loop for every track
        for i = 1:length(tracks)
            bbox = tracks(i).bbox;

            % predict the current location of the track
            predictedCentroid = predict(tracks(i).kalmanFilter);

            % change location of bounding box to centre of new predicted
            % location
            predictedCentroid = int32(predictedCentroid) - bbox(3:4) / 2;
            tracks(i).bbox = [predictedCentroid, bbox(3:4)];
        end
    end

    function [assignments, unassignedTracks, unassignedDetections] = ...
            detectionToTrackAssignment()

        nTracks = length(tracks);
        nDetections = size(centroids, 1);

        % Compute the cost of assigning each detection to each track.
        cost = zeros(nTracks, nDetections);
        for i = 1:nTracks
            cost(i, :) = distance(tracks(i).kalmanFilter, centroids);
        end

        % Solve the assignment problem.
        costOfNonAssignment = 20;
        [assignments, unassignedTracks, unassignedDetections] = ...
            assignDetectionsToTracks(cost, costOfNonAssignment);
    end

    function updateAssignedTracks()
        numAssignedTracks = size(assignments, 1);
        for i = 1:numAssignedTracks
            trackIdx = assignments(i, 1);
            detectionIdx = assignments(i, 2);
            centroid = centroids(detectionIdx, :);
            bbox = bboxes(detectionIdx, :);

            % Correct the estimate of the object's location
            % using the new detection.
            correct(tracks(trackIdx).kalmanFilter, centroid);

            % Replace predicted bounding box with detected
            % bounding box.
            tracks(trackIdx).bbox = bbox;

            % Update track's age.
            tracks(trackIdx).age = tracks(trackIdx).age + 1;

            % Update visibility.
            tracks(trackIdx).totalVisibleCount = ...
                tracks(trackIdx).totalVisibleCount + 1;
            tracks(trackIdx).consecutiveInvisibleCount = 0;
        end
    end

    function updateUnassignedTracks()
        for i = 1:length(unassignedTracks)
            ind = unassignedTracks(i);
            tracks(ind).age = tracks(ind).age + 1;
            tracks(ind).consecutiveInvisibleCount = ...
                tracks(ind).consecutiveInvisibleCount + 1;
        end
    end

    function deleteLostTracks()
        if isempty(tracks)
            return;
        end

        invisibleForTooLong = 20;
        ageThreshold = 8;

        % Compute the fraction of the track's age for which it was visible.
        ages = [tracks(:).age];
        totalVisibleCounts = [tracks(:).totalVisibleCount];
        visibility = totalVisibleCounts ./ ages;

        % Find the indices of 'lost' tracks.
        lostInds = (ages < ageThreshold & visibility < 0.6) | ...
            [tracks(:).consecutiveInvisibleCount] >= invisibleForTooLong;

        % Delete lost tracks.
        tracks = tracks(~lostInds);
    end

    function createNewTracks()
        centroids = centroids(unassignedDetections, :);
        bboxes = bboxes(unassignedDetections, :);

        for i = 1:size(centroids, 1)

            centroid = centroids(i,:);
            bbox = bboxes(i, :);

            % Create a Kalman filter object.
            kalmanFilter = configureKalmanFilter('ConstantVelocity', ...
                centroid, [200, 50], [100, 25], 100);

            % create an empty array of tracks with fields:
            %   - ID 
            %   - Bbox
            %   - Kalman filter
            %   - age
            %   - totalVisibleCount
            %   - consecutiveInvisibleCount            
            newTrack = struct(...
                'id', nextId, ...
                'bbox', bbox, ...
                'kalmanFilter', kalmanFilter, ...
                'age', 1, ...
                'totalVisibleCount', 1, ...
                'consecutiveInvisibleCount', 0);

            % Add it to the array of tracks.
            tracks(end + 1) = newTrack;

            % Increment the next id.
            nextId = nextId + 1;
        end
    end

    function displayTrackingResults()
        % Convert the frame and the mask to uint8 RGB.
        frame = im2uint8(frame);
        mask = uint8(repmat(mask, [1, 1, 3])) .* 255;

        minVisibleCount = 8;
        if ~isempty(tracks)

            % Noisy detections tend to result in short-lived tracks.
            % Only display tracks that have been visible for more than
            % a minimum number of frames.
            reliableTrackInds = ...
                [tracks(:).totalVisibleCount] > minVisibleCount;
            % Set variable equal to tracks
            reliableTracks = tracks(reliableTrackInds);

            % Display the objects. If an object has not been detected
            % in this frame, display its predicted bounding box.
            if ~isempty(reliableTracks)
                % Get bounding boxes for video player
                bboxes = cat(1, reliableTracks.bbox);

                % Get ids from function that found tracks
                ids = int32([reliableTracks(:).id]);

                % Create labels for objects indicating the ones for
                % which we display the predicted rather than the actual
                % location
                labels = cellstr(int2str(ids'));
                % Only adds to matrix if the tracks and count are greater
                % than 0
                predictedTrackInds = ...
                    [reliableTracks(:).consecutiveInvisibleCount] > 0;
                isPredicted = cell(size(labels));
                isPredicted(predictedTrackInds) = {' predicted'};
                labels = strcat(labels, isPredicted);

                % Draw the objects on the frame
                frame = insertObjectAnnotation(frame, 'rectangle', ...
                    bboxes, labels);

                % Draw the objects on the mask
                mask = insertObjectAnnotation(mask, 'rectangle', ...
                    bboxes, labels);
            end
        end

        % Display the mask and the frame.
        obj.maskPlayer.step(mask);
        % Display the video player
        obj.videoPlayer.step(frame);
    end


end