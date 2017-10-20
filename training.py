import cv2
import os
import glob
import cPickle as pkl
import numpy as np
from PIL import Image
import pdb
import matplotlib.pyplot as plt
import sys
import time
# export PYTHONPATH="/home/sunnycia/caffe/python:$PYTHONPATH"
import caffe
caffe.set_mode_gpu()
caffe.set_device(0)

## load training data
snapshot=True
solver_path = 'solver.prototxt'
pretrained_model_path= '../pretrained_model/ResNet-50-model.caffemodel'
snapshot_path = '/data/sunnycia/saliency_on_videoset/Train/training_output/salicon/snapshot_iter_100000.solverstate'
frames_path = '../dataset/salicon_frame-mini.pkl'
densitys_path = '../dataset/salicon_density-mini.pkl'

#####
print "Loading data..."
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
    frame = frame/255.
    density = density/255.
    frames.append(frame)
    densitys.append(density)
    if len(frames) % 1000 == 0:
        print len(frames)

#####
# frames = pkl.load(open(frames_path, 'rb'))
# densitys = pkl.load(open(densitys_path, 'rb'))

# load the solver
solver = caffe.AdaDeltaSolver(solver_path)
# solver.net.copy_from(pretrained_model_path) # untrained.caffemodel
if snapshot
solver.restore(snapshot_path)
# solver.net.copy_from(pretrained_model_path) # untrained.caffemodel
start_time = time.time()
idx_counter = 0
epoch=10
# for e in range(epoch):
while time.time() - start_time < 43200*2:
    batch = np.random.permutation(len(frames))
    for i in range(0, len(batch)):
        idx_counter = idx_counter + 1
        # print 'working on ' + str(i) + ' of ' + str(len(batch))
        frame = frames[batch[i]]
        density = densitys[batch[i]]
        
        solver.net.blobs['data'].data[...] = frame
        solver.net.blobs['ground_truth'].data[...] = density
        solver.step(1)
    #     if int(time.time() - start_time) % 10000 == 0:
    #         solver.net.save('train_output/finetuned_salicon_{}.caffemodel'.format(idx_counter))
    # solver.net.save('train_output/finetuned_salicon_{}.caffemodel'.format(idx_counter))
