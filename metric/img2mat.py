import cv2
import os, glob
import scipy.io as sio
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--imgdir', type=str, required=True)
parser.add_argument('--matdir', type=str, required=True)
args = parser.parse_args()

img_dir = args.imgdir
mat_dir = args.matdir
if not os.path.isdir(mat_dir):
    os.makedirs(mat_dir)

fixmap_path_list = glob.glob(os.path.join(img_dir, '*.*'))
for fixmap_path in fixmap_path_list:
    fixmap = cv2.imread(fixmap_path, 0)

    output_path = os.path.join(mat_dir, os.path.basename(fixmap_path).split('.')[0]+'.mat')

    sio.savemat(output_path, {'fixation':fixmap})
    print "Done for",output_path,'\r',