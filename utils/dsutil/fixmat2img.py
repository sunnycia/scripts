import cv2
import os, glob
import scipy.io as sio
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--matdir', type=str, required=True)
parser.add_argument('--imgdir', type=str, required=True)

args = parser.parse_args()

mat_dir = args.matdir
img_dir = args.imgdir

if not os.path.isdir(mat_dir):
    print mat_dir, 'not exists.abort'
    exit()

if not os.path.isdir(img_dir):
    os.makedirs(img_dir)

mat_path_list = glob.glob(os.path.join(mat_dir,  '*.mat'))

for mat_path in mat_path_list:
    video_dir = mat_path.split('/')[-2]
    if not os.path.isdir(os.path.join(img_dir, video_dir)):
        os.makedirs(os.path.join(img_dir, video_dir))


    mat_name = os.path.basename(mat_path)
    img_name = mat_name.split('.')[0]+'.bmp'

    img_path = os.path.join(img_dir, video_dir, img_name)
    mat_data = sio.loadmat(mat_path)['fixation']

    mat_data = mat_data * 255

    # print img_path
    cv2.imwrite(img_path, mat_data);#exit()


