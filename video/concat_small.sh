#!/bin/sh

ffmpeg -i mexca_screencast.mp4 -i segment_1_1864_2580_annotated.mp4 -filter_complex "[0:v] setdar=640/480,tpad=stop_mode=clone:stop_duration=5[v0]; [1:v] scale=640:480:force_original_aspect_ratio=decrease,pad=640:480:-1:-1:color=black,setpts=1.5*PTS[v1];
 [v0][v1] concat=n=2:v=1 [v]" -map "[v]" -c:v libx264 -movflags +faststart demo_small.mp4
