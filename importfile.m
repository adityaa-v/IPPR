function importfile(fileToRead1)
%IMPORTFILE(FILETOREAD1)
%  Imports data from the specified file
%  FILETOREAD1:  file to read

%  Auto-generated by MATLAB on 02-Oct-2019 10:01:21

% Import the file
newData1 = load('-mat', fileToRead1);

% Create new variables in the base workspace from those fields.
vars = fieldnames(newData1);
for i = 1:length(vars)
    assignin('base', vars{i}, newData1.(vars{i}));
end


[fileName,pathName] = uigetfile('*.tif')
dname       = fullfile(pathName,fileName)
filelist = dir([fileparts(dname) filesep '*.tif']);
fileNames = {filelist.name}';
num_frames = (numel(filelist));
I = imread(fullfile(pathname, fileNames{1})); %to show the first image in the selected folder
imshow(I, []);
