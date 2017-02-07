#!/bin/bash

FOLDER=$1

for bag in `find $FOLDER -name *.bag`; do
    echo $bag
    avi="${bag%.*}.avi"
    mp4="${bag%.*}.mp4"
    python bag2video.py /bebop/image_raw $bag -p 1 --outfile $avi
    avconv -i $avi -c:v libx264 -c:a copy $mp4
    rm -f $avi
done
