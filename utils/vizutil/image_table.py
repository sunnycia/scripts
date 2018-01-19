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
parser.add_argument('--image_base', type=str, required=True)
parser.add_argument('--padding', type=int, default=3)
# parser.add_argument('--reduce_ratio', type=float, default=0.1)
parser.add_argument('--standard_size', type=tuple, default=(192, 108))

args = parser.parse_args()
output_path = args.output_path
image_base = args.image_base
padding = args.padding
# reduce_ratio = args.reduce_ratio
standard_size = args.standard_size

tmp_dir = '../tmp_imgdir_'+str(int(time.time()))
os.makedirs(tmp_dir)

videoname_list = os.listdir(image_base)

for videoname in videoname_list:
    videodir = os.path.join(image_base, videoname)
    framepath_list = glob.glob(os.path.join(videodir, '*.*'))
    length = len(framepath_list)
    rnd_index = random.randint(0, length-1)
    framepath = framepath_list[rnd_index]

    des_name = videoname + '_' + os.path.basename(framepath)
    shutil.copy(framepath, os.path.join(tmp_dir, des_name))




jigsaw(tmp_dir, output_path, stdsize=standard_size, padding=padding)

shutil.rmtree(tmp_dir)
