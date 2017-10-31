import argparse
import cv2
import os
import glob
import cPickle as pkl
import numpy as np
import sys
import time
import caffe
from random import shuffle
from utils.caffe_tools import CaffeSolver
from utils.file_check import check_path_list
caffe.set_mode_gpu()
caffe.set_device(0)

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_prototxt', type=str, default='train.prototxt', help='the network prototxt')
    parser.add_argument('--solver_prototxt', type=str, default='solver.prototxt', help='the network prototxt')
    parser.add_argument('--use_snapshot', type=bool, default=False, help='Use snapshot mode or not.')
    parser.add_argument('--size', type=int, default=1000, help='Dataset length.Show/Cut')
    parser.add_argument('--debug', type=bool, required=True, help='If debug is ture, a mini set will run into training.Or a complete set will.')
    parser.add_argument('--visualization', type=bool, default=False, help='visualization option')
    return parser.parse_args()

print "Parsing arguments..."
##
plot_figure_dir = '../figure'
pretrained_model_path= '../pretrained_model/ResNet-50-model.caffemodel'
snapshot_path = '/data/sunnycia/saliency_on_videoset/Train/training_output/salicon/snapshot-train_kldloss_withouteuc_iter_100000.solverstate'

# path_list = [solver_path, pretrained_model_path, snapshot_path]
# check_path_list(path_list)

# frames_path = '../dataset/salicon_frame.pkl'
# densitys_path = '../dataset/salicon_density.pkl'
# training_frames = pkl.load(open(frames_path, 'rb'))
# training_densitys = pkl.load(open(densitys_path, 'rb'))
train_frame_basedir = '/data/sunnycia/SaliencyDataset/Image/SALICON/DATA/train_val/train2014/images'
train_density_basedir = '/data/sunnycia/SaliencyDataset/Image/SALICON/DATA/train_val/train2014/density'

validation_frame_basedir = '/data/sunnycia/SaliencyDataset/Image/SALICON/DATA/train_val/val2014/images'
validation_density_basedir = '/data/sunnycia/SaliencyDataset/Image/SALICON/DATA/train_val/val2014/density'

args = get_arguments()

##### Data preparation section.
## load training data
### generate solver file
training_protopath = args.train_prototxt
solver_path = args.solver_prototxt
solverproto = CaffeSolver(trainnet_prototxt_path=training_protopath)

snapshot_prefix = solverproto.sp['snapshot_prefix'][1:-1]


update_dict = {
# 'solver_type':'SGD'
}
extrainfo_dict = {
}

postfix_str=os.path.basename(training_protopath).split('.')[0]
for key in update_dict:
    solverproto.sp[key] = update_dict[key]
    postfix_str += '-'+key +'-'+ update_dict[key]
for key in extrainfo_dict:
    postfix_str += '-'+key+ '-'+ extrainfo_dict[key]
postfix_str += '_'+str(int(time.time()))
snapshot_dirname = os.path.join(os.path.dirname(snapshot_prefix), postfix_str)
print snapshot_dirname;
if not os.path.isdir(snapshot_dirname):
    os.makedirs(snapshot_dirname)
snapshot_prefix = '"'+ snapshot_dirname + '/snapshot-"'
print "snapshot will be save to", snapshot_prefix
solverproto.sp['snapshot_prefix'] = snapshot_prefix
solverproto.write(solver_path);

plot_figure_dir = os.path.join(plot_figure_dir, postfix_str)
if not os.path.isdir(plot_figure_dir):
    os.makedirs(plot_figure_dir)
print "Loss figure will be save to", plot_figure_dir
##

print "Loading data..."
MEAN_VALUE = np.array([103.939, 116.779, 123.68], dtype=np.float32)   # B G R/ use opensalicon's mean_value
MEAN_VALUE = MEAN_VALUE[None, None, ...]
## training dateset
frame_path_list = glob.glob(os.path.join(train_frame_basedir, '*.*'))
density_path_list = glob.glob(os.path.join(train_density_basedir, '*.*'))
training_frames = []
training_densitys = []
for (frame_path, density_path) in zip(frame_path_list, density_path_list):
    frame = cv2.imread(frame_path).astype(np.float32)
    density = cv2.imread(density_path, 0).astype(np.float32)
    frame -= MEAN_VALUE
    frame = cv2.resize(frame, dsize=(480, 288))
    density = cv2.resize(density, dsize=(480, 288))
    frame = np.transpose(frame, (2, 0, 1))
    density = density[None, ...]
    # print frame.shape, density.shape;exit()
    frame = frame/255.
    density = density/255.
    training_frames.append(frame)
    training_densitys.append(density)
    if len(training_frames) % args.size == 0:
        if args.debug==True:
            break
        else:
            print len(training_frames)
##validation dataset
# frame_path_list = glob.glob(os.path.join(validation_frame_basedir, '*.*'))
# density_path_list = glob.glob(os.path.join(validation_density_basedir, '*.*'))
# validation_frames = []
# validation_densitys = []
# for (frame_path, density_path) in zip(frame_path_list, density_path_list):
#     frame = cv2.imread(frame_path).astype(np.float32)
#     density = cv2.imread(density_path, 0).astype(np.float32)
#     frame -= MEAN_VALUE
#     frame = cv2.resize(frame, dsize=(480, 288))
#     density = cv2.resize(density, dsize=(480, 288))
#     frame = np.transpose(frame, (2, 0, 1))
#     density = density[None, ...]
#     # print frame.shape, density.shape;exit()
#     frame = frame/255.
#     density = density/255.
#     validation_frames.append(frame)
#     validation_densitys.append(density)
#     if len(validation_frames) % args.size == 0:
#         if args.debug==True:
#             break
#         else:
#             print len(validation_frames)

##### Training section

# load the solver
if 'solver_type' in update_dict:
    if update_dict['solver_type'] =='SGD':
        solver = caffe.SGDSolver(solver_path)
else:
    solver = caffe.AdaDeltaSolver(solver_path)

if args.use_snapshot == True:
    solver.restore(snapshot_path)
else:
    solver.net.copy_from(pretrained_model_path) # untrained.caffemodel

tart_time = time.time()

max_iter = 1000000
validation_iter = 1000
plot_iter = 100
epoch=10
idx_counter = 0

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# loss_list = [None for i in range(max_iter)]

# x = np.arange(max_iter)
# y=[None for i in range(max_iter)]
# y = np.zeros(max_iter)
x=[]
y=[]
plt.plot(x, y)
# while time.time() - start_time < 43200:
for _iter in range(max_iter):
    batch = np.random.permutation(len(training_frames))
    for i in range(0, len(batch)):
        idx_counter = idx_counter + 1
        # print 'working on ' + str(i) + ' of ' + str(len(batch))
        
        
        # if _iter % validation_iter == 0:
        #     ##do validation
        #     print "Do validation..."
            

        frame = training_frames[batch[i]]
        density = training_densitys[batch[i]]
        solver.net.blobs['data'].data[...] = frame
        solver.net.blobs['ground_truth'].data[...] = density
        solver.step(1)
        # print solver.net.blobs['loss'].data[...];exit()
        x.append(idx_counter)
        # print solver.net.blobs['loss'].data[...].shape, solver.net.blobs['loss'].data[...].tolist();exit()
        y.append(solver.net.blobs['loss'].data[...].tolist())
        # y[idx_counter] = solver.net.blobs['loss'].data[...]
        # print y[:100]
        # y.append(solver.net.blobs['loss'].data)
        # x.append(idx_counter)

        plt.plot(x, y)
        if idx_counter%plot_iter==0:
            plt.xlabel('Iter')
            plt.ylabel('loss')
            plt.savefig(os.path.join(plot_figure_dir, "plot.png"))
            plt.clf()
        if args.visualization:
            plt.show()




 