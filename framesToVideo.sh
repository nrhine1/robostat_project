# !/bin/bash
# Example: >> sh framesToVideo.sh 'figs/%d.jpg' 30 outVid.avi

# MATLAB CMD:
# cmd = sprintf('sh framesToVideo.sh %s %d %s',AbsolutePath(fullfile(pathToFrames,'image-%06d.jpg')),fps,AbsolutePath(fullfile(out_videoname)));

imgStr=$1
fps=$2
vidOut=$3

ffmpeg -r $fps -f image2 -i $imgStr -qscale 1 -r $fps $vidOut
