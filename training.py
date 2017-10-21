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
    return parser.parse_args()

print "Parsing arguments..."
args = get_arguments()

##### Data preparation section.
## load training data
### generate solver file
training_protopath = args.train_prototxt
solver_path = args.solver_prototxt
solverproto = CaffeSolver(trainnet_prototxt_path=training_protopath)

snapshot_prefix = solverproto.sp['snapshot_prefix'][1:-1]
snapshot_dirname = os.path.dirname(snapshot_prefix)
if not os.path.isdir(snapshot_dirname):
    os.makedirs(snapshot_dirname)


update_dict = {
}

postfix_str=training_protopath.split('.')[0]
for key in update_dict:
    solverproto.sp[key] = update_dict[key]
    postfix_str += '-'+key +'-'+ update_dict[key]
snapshot_prefix = '"'+ snapshot_dirname + '/snapshot-' + postfix_str+ '"'
print "snapshot will be save to", snapshot_prefix, "..."
solverproto.sp['snapshot_prefix'] = snapshot_prefix
solverproto.write(solver_path);
##


##
pretrained_model_path= '../pretrained_model/ResNet-50-model.caffemodel'
snapshot_path = '/data/sunnycia/saliency_on_videoset/Train/training_output/salicon/snapshot_iter_100000.solverstate'
frames_path = '../dataset/salicon_frame.pkl'
densitys_path = '../dataset/salicon_density.pkl'

path_list = [solver_path, pretrained_model_path, snapshot_path, frames_path, densitys_path]
# check_path_list(path_list)
# frames = pkl.load(open(frames_path, 'rb'))
# densitys = pkl.load(open(densitys_path, 'rb'))
frame_basedir = '/data/sunnycia/SaliencyDataset/Image/SALICON/DATA/train_val/train2014/images'
density_basedir = '/data/sunnycia/SaliencyDataset/Image/SALICON/DATA/train_val/train2014/density'


print "Loading data..."
MEAN_VALUE = np.array([103.939, 116.779, 123.68], dtype=np.float32)   # B G R/ use opensalicon's mean_value
MEAN_VALUE = MEAN_VALUE[None, None, ...]

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
    frame = np.transpose(frame, (2, 0, 1))
    density = density[None, ...]
    # print frame.shape, density.shape;exit()
    frame = frame/255.
    density = density/255.
    frames.append(frame)
    densitys.append(density)
    if len(frames) % args.size == 0:
        if args.debug==True:
            break
        else:
            print len(frames)

##### Training section

# load the solver
solver = caffe.AdaDeltaSolver(solver_path)
if args.use_snapshot == True:
    solver.restore(snapshot_path)
else:
    solver.net.copy_from(pretrained_model_path) # untrained.caffemodel
start_time = time.time()
epoch=10
idx_counter = 0
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
