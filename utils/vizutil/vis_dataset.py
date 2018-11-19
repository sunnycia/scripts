import os, glob, shutil
import cv2
from random import shuffle
from grey_to_rgb import *
from jigsaw import *
frame_dir = '/data/SaliencyDataset/Video/VideoSet/ImageSet/Seperate/frame'
density_dir = '/data/SaliencyDataset/Video/VideoSet/ImageSet/Seperate/density/sigma32'
output_path = './dataset_jigsaw.jpg'
# video_name_list = os.listdir(frame_dir)
video_name_list = ['videoSRC001','videoSRC017','videoSRC023','videoSRC043','videoSRC067','videoSRC152','videoSRC211']
shuffle(video_name_list)

def get_frame_index(frame_path):
    frame_index = int(os.path.basename(frame_path).split('.')[0].split('_')[-1])
    return frame_index

tmp_dir = '/tmp/visdataset'
if not os.path.isdir(tmp_dir):
    os.makedirs(tmp_dir)
else:
    shutil.rmtree(tmp_dir)
    
video_num = 5
counter = 0
for video_name in video_name_list[:video_num]:
    frame_path_list = glob.glob(os.path.join(frame_dir, video_name, '*.*'))
    density_path_list = glob.glob(os.path.join(density_dir, video_name, '*.*'))

    frame_path_list.sort(key=get_frame_index)
    density_path_list.sort(key=get_frame_index)
    
    total_frame = len(frame_path_list)

    for i in range(10, 101, 20):
        frame = cv2.imread(frame_path_list[i])
        density = grey_to_rgb(cv2.imread(density_path_list[i]), 'y')

        blend = cv2.addWeighted(frame, 1.0, density, 1.0, 1.0)
        save_name = str(counter).zfill(3)+'.jpg'
        cv2.imwrite(os.path.join(tmp_dir, save_name), blend)
        # cv2.imshow('hey', blend)
        # cv2.waitKey(0);exit()

        counter += 1
jigsaw(tmp_dir, output_path=output_path, stdsize=(192,108), padding=3)

shutil.rmtree(tmp_dir)




