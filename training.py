from Dataset import Dataset
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import argparse, cv2, os, glob, sys, time
import cPickle as pkl
import numpy as np
import caffe
from random import shuffle
from utils.caffe_tools import CaffeSolver
from utils.file_check import check_path_list
from caffe.proto import caffe_pb2
import google.protobuf.text_format as txtf

caffe.set_mode_gpu()
# caffe.set_device(0,1)

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_prototxt', type=str, default='prototxt/train.prototxt', help='the network prototxt')
    parser.add_argument('--solver_prototxt', type=str, default='prototxt/solver.prototxt', help='the network prototxt')
    parser.add_argument('--use_snapshot', type=bool, default=False, help='Use snapshot mode or not.')
    parser.add_argument('--size', type=int, default=1000, help='Dataset length.Show/Cut')
    parser.add_argument('--debug', type=bool, default=False, help='If debug is ture, a mini set will run into training.Or a complete set will.')
    parser.add_argument('--visualization', type=bool, default=False, help='visualization option')
    parser.add_argument('--batch', type=int, default=1, help='training mini batch')
    return parser.parse_args()

print "Parsing arguments..."
##
plot_figure_dir = '../figure'
pretrained_model_path= '../pretrained_model/ResNet-50-model.caffemodel'
snapshot_path = '/data/sunnycia/saliency_on_videoset/Train/training_output/salicon/snapshot-train_kldloss_withouteuc_iter_100000.solverstate'

args = get_arguments()
debug_mode = args.debug

##### Data preparation section.
## load training data
### generate solver file & net file
update_solver_dict = {
# 'solver_type':'SGD'
}
extrainfo_dict = {
}
batch = args.batch

## Programatically change network
training_protopath = args.train_prototxt
net = caffe_pb2.NetParameter()
with open(training_protopath) as f:
    s = f.read()
    txtf.Merge(s, net)
layerNames = [l.name for l in net.layer]
data_layer = net.layer[layerNames.index('data')]
old_paramstr = data_layer.python_param.param_str
# print old_paramstr, ('.').join(), type(old_paramstr);exit()
pslist = old_paramstr.split(',')
pslist[0]= str(batch)
new_paramstr = (',').join(pslist)
data_layer.python_param.param_str = new_paramstr

gt_layer = net.layer[layerNames.index('ground_truth')]
old_paramstr = gt_layer.python_param.param_str
# print old_paramstr, ('.').join(), type(old_paramstr);exit()
pslist = old_paramstr.split(',')
pslist[0]= str(batch)
new_paramstr = (',').join(pslist)
gt_layer.python_param.param_str = new_paramstr


print 'writing', training_protopath
with open(training_protopath, 'w') as f:
    f.write(str(net))


solver_path = args.solver_prototxt
solverproto = CaffeSolver(trainnet_prototxt_path=training_protopath)

snapshot_prefix = solverproto.sp['snapshot_prefix'][1:-1]

postfix_str=os.path.basename(training_protopath).split('.')[0]
for key in update_solver_dict:
    solverproto.sp[key] = update_solver_dict[key]
    postfix_str += '-'+key +'-'+ update_solver_dict[key]
for key in extrainfo_dict:
    postfix_str += '-'+key+ '-'+ extrainfo_dict[key]
postfix_str += '-'+'batch'+'-'+str(batch)
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

train_frame_basedir = '/data/sunnycia/SaliencyDataset/Image/SALICON/DATA/train_val/train2014/images'
train_density_basedir = '/data/sunnycia/SaliencyDataset/Image/SALICON/DATA/train_val/train2014/density'
tranining_dataset = Dataset(train_frame_basedir, train_density_basedir, debug=debug_mode)
validation_frame_basedir = '/data/sunnycia/SaliencyDataset/Image/SALICON/DATA/train_val/val2014/images'
validation_density_basedir = '/data/sunnycia/SaliencyDataset/Image/SALICON/DATA/train_val/val2014/density'
# validation_dataset = Dataset(train_frame_basedir, train_density_basedir, debug=debug_mode)

##### Training section
# load the solver
if 'solver_type' in update_solver_dict:
    if update_solver_dict['solver_type'] =='SGD':
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
plot_iter = 500
epoch=10
idx_counter = 0

x=[]
y=[]
z=[] # validation

plt.plot(x, y)
_step=0
while _step * batch < max_iter:

    if _step%validation_iter==0:
        ##do validation
        pass

    frame_minibatch, density_minibatch = tranining_dataset.next_batch(batch)
    # print frame_minibatch.shape;exit()
    solver.net.blobs['data'].data[...] = frame_minibatch
    solver.net.blobs['ground_truth'].data[...] = density_minibatch
    solver.step(1)

    x.append(_step)
    y.append(solver.net.blobs['loss'].data[...].tolist())

    plt.plot(x, y)
    if _step%plot_iter==0:
        plt.xlabel('Iter')
        plt.ylabel('loss')
        plt.savefig(os.path.join(plot_figure_dir, "plot"+str(_step)+".png"))
        plt.clf()
    if args.visualization:
        plt.show()

    _step+=1

import cPickle as pkl
pkl.dump(x, open(os.path.join(plot_figure_dir, "x.pkl")))
pkl.dump(y, open(os.path.join(plot_figure_dir, "y.pkl")))