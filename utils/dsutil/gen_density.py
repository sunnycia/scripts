import os,glob  
import scipy.io as sio, numpy as np
import cv2
from scipy import ndimage
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--sigma", type=int, required=True)
parser.add_argument("--fixation_base", type=str, default='/data/sunnycia/SaliencyDataset/Video/MSU/fixation/mat')
parser.add_argument("--density_base", type=str, default='/data/sunnycia/SaliencyDataset/Video/MSU/density')
parser.add_argument("--fixationtype", type=str, default='mat')

args = parser.parse_args()
sigma = args.sigma
fixation_base = args.fixation_base
density_base = args.density_base
density_base = os.path.join(density_base, 'sigma'+str(sigma))
if not os.path.isdir(density_base):
    os.makedirs(density_base)

video_name_list = os.listdir(fixation_base)
for video_name in video_name_list:
    fixation_dir = os.path.join(fixation_base, video_name)
    density_dir = os.path.join(density_base, video_name)
    if not os.path.isdir(density_dir):
        os.makedirs(density_dir)

    fixation_path_list = glob.glob(os.path.join(fixation_dir, "*.*"))
        
    for fixation_path in fixation_path_list:
        prefix = os.path.basename(fixation_path).split('.')[0]
        output_path = os.path.join(density_dir, prefix+'.jpg')
        if os.path.isfile(output_path):
            print output_path, "exists, pass"
            continue
        if args.fixationtype =='mat':
            fixation = sio.loadmat(fixation_path)['fixation'].astype(np.float32)
        elif args.fixationtype == 'image':
            fixation = cv2.imread(fixation_path, 0)
            fixation = fixation / 255.
        # print fixation, fixation.max();exit()
        # density_map = ndimage.filters.gaussian_filter(fixation, sigma)
        density_map = cv2.GaussianBlur(fixation, (0, 0), sigma)
        density_map -= np.min(density_map)
        density_map /= np.max(density_map)

        cv2.imwrite(output_path, density_map*255);
