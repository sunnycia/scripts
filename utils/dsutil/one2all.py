import os, glob
import shutil
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--videodirbase', type=str, required=True)
parser.add_argument('--alldir', type=str, required=True)

arg = parser.parse_args()
videodir_base = arg.videodirbase
all_dir = arg.alldir
if not os.path.isdir(all_dir):
    os.path.makedirs(all_dir)


frame_path_list = glob.glob(os.path.join(videodir_base, '*', '*.*'))
for frame_path in frame_path_list:
    videoname = frame_path.split('/')[-2]
    frame_name = os.path.basename(frame_path)

    new_name = videoname + '_'+ frame_name
    new_path = os.path.join(all_dir, new_name)

    shutil.copy(frame_path, new_path)
    print new_path#;exit()
