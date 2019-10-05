imds=imageDatastore('C:\Users\adzar\Documents\Human3extract\Human3\img','FileExtensions',{'.jpg'});
data=readall(imds)
imshow(data{3})