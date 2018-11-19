import os, glob
import cv2
import argparse
from shotdetect import *

parser = argparse.ArgumentParser()
parser.add_argument('--video_directory', type=str, required=True, help='directory of videos')

args = parser.parse_args()

# video_directory = '/data/SaliencyDataset/Video/GAZECOM/videos'
video_directory = args.video_directory

video_path_list = glob.glob(os.path.join(video_directory, '*.*'))

video_counter = 0
frame_counter = 0
resolution_list = []
fps_list = []
shot_list = []
for video_path in video_path_list:
    # print video_path
    video_counter += 1
    video_capture = cv2.VideoCapture(video_path)
    
    detector = shotDetector(video_path, output_dir=None)
    detector.run()

    shot_list.append(len(detector.shots))
    frame_counter += video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
    resolution = (video_capture.get(cv2.CAP_PROP_FRAME_WIDTH), video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)

    if resolution not in resolution_list:
        resolution_list.append(resolution)

    if fps not in fps_list:
        fps_list.append(fps)

resolution_list.sort()
fps_list.sort()

print os.path.basename(video_directory)
print "Total video samples:", video_counter
print "Total frame samples:", frame_counter
print "Resolution:", resolution_list
print "Fps:", fps_list
print "Total shots:", sum(shot_list)
print "Shots per video:", sum(shot_list)/float(video_counter)

