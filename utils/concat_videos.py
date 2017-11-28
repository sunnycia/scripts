import glob
import os
import argparse
from random import shuffle
import base64,time

parser = argparse.ArgumentParser()
parser.add_argument('--videodir', type=str, required=True, help="Directory of videos")
parser.add_argument('--outputpath', type=str, required=True, help="Path of output video")
parser.add_argument('--randomorder', type=bool, default=False, help="Directory of saliency video")
parser.add_argument('--concatmethod', type=int, default=2, help="ffmpeg concat method(1 for protocol, 2 for demuxer)")
args = parser.parse_args()

video_dir = args.videodir
output_path = args.outputpath
random_order = args.randomorder

if os.path.isfile(output_path):
    print output_path, "already exists.Abort"

video_path_list = glob.glob(os.path.join(video_dir, '*.*'))
if random_order:
    shuffle(video_path_list)

tmp_file_name='_tmp_file.txt'

if args.concatmethod == 1:
    video_str = '|'.join(video_path_list)
    cmd_str = 'ffmpeg -i "concat:%s" -c copy %s' % (video_str, output_path)
elif args.concatmethod == 2:
    tmp_w_f = open(tmp_file_name, 'w')
    for video_path in video_path_list:
        print >> tmp_w_f, 'file', '\''+video_path+'\''
    cmd_str = 'ffmpeg -f concat -i %s -c copy %s' % (tmp_file_name, output_path)

os.system(cmd_str)

if args.concatmethod == 2:
    os.remove(tmp_file_name)