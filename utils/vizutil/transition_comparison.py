import cv2
import os, glob
from jigsaw import *
import shutil
import argparse
import time 
import random

## For create image table of video dataase
parser = argparse.ArgumentParser()
parser.add_argument('--output_path', type=str, required=True)
parser.add_argument('--wildcards_record', type=str, required=True)
parser.add_argument('--samples', type=int, default=8)
parser.add_argument('--interval', type=int, default=5)
###demo wildcards_record.txt path, model_name
# /data/SaliencyDataset/Video/MSU/frames/*/*.*, RGB frame
# /data/SaliencyDataset/Video/MSU/density/sigma32/*/*.*, Ground Truth
# /data/SaliencyDataset/Video/MSU/saliency_map/GBVS/*/*.*, GBVS
# /data/SaliencyDataset/Video/MSU/saliency_map/SALICON/*/*.*, SALICON
# /data/SaliencyDataset/Video/MSU/saliency_map/pqft/*/*.*, PQFT
# /data/SaliencyDataset/Video/MSU/saliency_map/xu_lstm/*/*.*, OMCNN-LSTM
# /data/SaliencyDataset/Video/MSU/saliency_map/vo-v4-2-resnet-dropout-snapshot-2000-display-1-dropout_fulldens-batch-2_1514857787_snapshot-_iter_26000_threshold0/*/*.*, PROPOSED
###4

parser.add_argument('--padding', type=int, default=3)
# parser.add_argument('--reduce_ratio', type=float, default=0.1)
parser.add_argument('--standard_size', type=tuple, default=(192, 108))

args = parser.parse_args()
output_path = args.output_path
samples = args.samples
interval = args.interval
# image_base = args.image_base
wildcards_record = args.wildcards_record
padding = args.padding
# reduce_ratio = args.reduce_ratio
standard_size = args.standard_size

tmp_dir = '/tmp/_transition_comparison'
if not os.path.isdir(tmp_dir):
    os.makedirs(tmp_dir)

rf = open(wildcards_record, 'r')
lines = rf.readlines()
dir_list = []
model_list = []
for line in lines:
    dir_, model_name = line.split(', ')
    dir_list.append(dir_)
    model_list.append(model_name)
print 

videoname_list = os.listdir(dir_list[0])
random.shuffle(videoname_list)

## videoname
# videoname = videoname_list[0]
videoname='videoSRC155'

framename_list = []
# for videoname in videoname_list:

frame_list = os.listdir(os.path.join(dir_list[0], videoname))
frame_number = len(frame_list)
start_index = random.randint(0, frame_number-80)
end_index = start_index + samples*interval
for i in range(start_index, end_index, interval):
    framename_list.append(frame_list[i])


# print videoname, frame_number


counter = 1
for m in range(len(dir_list)):
    # img_array = np.array(glob.glob(dir_list[m]))[perm]
    # print dir_list[m]
    for i in range(samples):
        img_path=os.path.join(dir_list[m], videoname, framename_list[i])

        # print dir_list[m].replace('*/*.*', '%s/%s')
        # img_path = dir_list[m].replace('*/*.*', '%s/%s') % (video_name_list[i], frame_name_list[i])
        # img_path = img_array[i]
        des_name = str(counter).zfill(5)+'.'+img_path.split('.')[-1]
        shutil.copy(img_path, os.path.join(tmp_dir, des_name))

        counter += 1

jigsaw(tmp_dir, output_path, stdsize=standard_size, padding=padding, tall_img=True)

shutil.rmtree(tmp_dir)
