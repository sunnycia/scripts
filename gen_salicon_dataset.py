import glob
import numpy as np
import os
import cPickle as pkl
import cv2

MEAN_VALUE = np.array([103.939, 116.779, 123.68], dtype=np.float32)   # B G R/ use opensalicon's mean_value
MEAN_VALUE = MEAN_VALUE[None, None, ...]

frame_basedir = '/data/sunnycia/SaliencyDataset/Image/SALICON/DATA/train_val/train2014/images'
density_basedir = '/data/sunnycia/SaliencyDataset/Image/SALICON/DATA/train_val/train2014/density'

frame_path_list = glob.glob(os.path.join(frame_basedir, '*.*'))
density_path_list = glob.glob(os.path.join(density_basedir, '*.*'))
frames = []
densitys = []
for (frame_path, density_path) in zip(frame_path_list, density_path_list):
    frame = cv2.imread(frame_path).astype(np.float32)
    density = cv2.imread(density_path, 0).astype(np.float32)
    frame -= MEAN_VALUE
    frame = cv2.resize(frame, dsize=(480, 288))
    density = cv2.resize(density, dsize=(480, 288))
    frame = np.transpose(frame, (2, 0, 1))[None, :]
    density = density[None, None, ...]
    # print frame.shape, density.shape;exit()
    frames.append(frame)
    densitys.append(density)
    if len(frames) % 1000 == 0:
        break
        # print len(frames)



pkl.dump(frames, open('../dataset/salicon_frame-mini.pkl', 'wb'))
pkl.dump(densitys, open('../dataset/salicon_density-mini.pkl', 'wb'))