imds=imageDatastore('C:\Users\astro\Documents\Human3extract\Human3\img','FileExtensions',{'.jpg'});
data=readall(imds)
imshow(data{3})