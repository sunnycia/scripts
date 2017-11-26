import glob
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--videodir', type=str, required=True, help="Directory of videos")
parser.add_argument('--outputpath', type=str, required=True, help="Path of output video")
parser.add_argument('--randomorder', type=bool, default=False, help="Directory of saliency video")
args = parser.parse_args()

video_dir = args.videodir
output_path = args.outputpath
random_order = args.randomorder

if os.path.isfile(output_path):
    print output_path, "already exists.Abort"


video_path_list = glob.glob(os.path.join(video_dir, '*.*'))
video_str = '|'.join(video_path_list)
cmd_str = 'ffmpeg -i "concat:%s" -c copy %s' % (video_str, output_path)
os.system(cmd_str)