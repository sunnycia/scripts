import glob
import cv2
import matlab.engine
import scipy.io as spio
import os
import numpy as np

os.system('export PATH="$PATH:./metric_script"')
print "Starting matlab engine..."
eng = matlab.engine.start_matlab()
print "Done!"

saliency_dir = '/data/sunnycia/SaliencyDataset/Image/SALICON/DATA/train_val/val2014/saliency/training_output_iter_390000'
fixation_dir = '/data/sunnycia/SaliencyDataset/Image/SALICON/DATA/train_val/val2014/fixation'
density_dir = '/data/sunnycia/SaliencyDataset/Image/SALICON/DATA/train_val/val2014/density'

saliency_path_list = glob.glob(os.path.join(saliency_dir, '*.*'))
fixation_path_list = glob.glob(os.path.join(fixation_dir, '*.*'))
density_path_list = glob.glob(os.path.join(density_dir, '*.*'))
print len(saliency_path_list), len(fixation_path_list), len(density_path_list)
assert len(saliency_path_list) == len(fixation_path_list) == len(density_path_list)

for i in range(len(saliency_path_list)):
    saliency_path = saliency_path_list[i]
    fixation_path = fixation_path_list[i]
    density_path = density_path_list[i]

    assert saliency_path.split('/')[-1].split('.')[0] == \
            fixation_path.split('/')[-1].split('.')[0] == \
            density_path.split('/')[-1].split('.')[0]
    saliency = cv2.imread(saliency_path, 0)
    density = cv2.imread(density_path, 0)

    print "Calculating metric..."
    cc = eng.CC(saliency, density)
    sim = eng.similarity(saliency, denisty)

    print cc, sim;exit()