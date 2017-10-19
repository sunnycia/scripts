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
solver_path = 'solver.prototxt'
pretrained_model_path= 'pretrained_model/ResNet-50-model.caffemodel'
frames_path = 'dataset/frame_mini.pkl'
densitys_path = 'dataset/density_mini.pkl'
print "Loading data..."
frames = pkl.load(open(frames_path, 'rb'))
densitys = pkl.load(open(densitys_path, 'rb'))

# load the solver
solver = caffe.AdaDeltaSolver(solver_path)
solver.net.copy_from(pretrained_model_path) # untrained.caffemodel
start_time = time.time()
idx_counter = 0
while time.time() - start_time < 43200:
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
