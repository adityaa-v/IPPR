v = VideoReader('C:\Users\adzar\Downloads\StillHuman.mp4');
frames=read(v);
v=VideoWriter('StillHuman.mp4');
open(v);
implay(frames)
