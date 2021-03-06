#coding=utf-8
import cPickle as pkl
from Dataset import VideoDataset
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
from Flownet import Flownet
from Saliencynet import ImageSaliencyNet
import utils.OpticalFlowToolkit.lib.flowlib as flib

caffe.set_mode_gpu()

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_prototxt', type=str, default='prototxt/train.prototxt', help='the network prototxt')
    parser.add_argument('--solver_prototxt', type=str, default='prototxt/solver.prototxt', help='the network prototxt')
    parser.add_argument('--use_snapshot', type=str, default='', help='Snapshot path.')
    parser.add_argument('--use_model', type=str, default='../pretrained_model/ResNet-50-model.caffemodel', help='Pretrained model')
    parser.add_argument('--size', type=int, default=1000, help='Dataset length.Show/Cut')
    parser.add_argument('--debug', type=bool, default=False, help='If debug is ture, a mini set will run into training.Or a complete set will.')
    parser.add_argument('--visualization', type=bool, default=False, help='visualization option')
    parser.add_argument('--batch', type=int, default=1, help='training mini batch')
    parser.add_argument('--flownet_code', type=str, default='0')
    parser.add_argument('--image_size', type=tuple, default=(480,288))
    return parser.parse_args()
print "Parsing arguments..."
args = get_arguments()

pretrained_model_path= args.use_model
debug_mode = args.debug
snapshot_path = args.use_snapshot
#Check if snapshot exists
if snapshot_path is not '':
    if not os.path.isfile(snapshot_path):
        print snapshot_path, "not exists.Abort"
        exit()

##########################################################################_#####
#  /| |      /                   /                   /                   / |   #
# ( | | ___ (___       ___  ___ (          ___  ___ (___       ___      (__/   #
# | | )|___)|    |   )|   )|   )|___)     |___ |___)|    |   )|   )      / \)  #
# | |/ |__  |__  |/\/ |__/ |    | \        __/ |__  |__  |__/ |__/      |__/\  #
#                                                             |                #
#   __                                                                         #
# |/  |      /                                                 /    /          #
# |   | ___ (___  ___       ___  ___  ___  ___  ___  ___  ___ (___    ___  ___ #
# |   )|   )|    |   )     |   )|   )|___)|   )|   )|   )|   )|    | |   )|   )#
# |__/ |__/||__  |__/|     |__/ |    |__  |__/ |__/||    |__/||__  | |__/ |  / #
###########################|##############|#####################################


"""A1: Update network prototxt"""
# ╦ ╦┌─┐┌┬┐┌─┐┌┬┐┌─┐  ┌┐┌┌─┐┌┬┐┬ ┬┌─┐┬─┐┬┌─  ┌─┐┬─┐┌─┐┌┬┐┌─┐┌┬┐─┐ ┬┌┬┐
# ║ ║├─┘ ││├─┤ │ ├┤   │││├┤  │ ││││ │├┬┘├┴┐  ├─┘├┬┘│ │ │ │ │ │ ┌┴┬┘ │ 
# ╚═╝┴  ─┴┘┴ ┴ ┴ └─┘  ┘└┘└─┘ ┴ └┴┘└─┘┴└─┴ ┴  ┴  ┴└─└─┘ ┴ └─┘ ┴ ┴ └─ ┴ 
batch = args.batch
training_protopath = args.train_prototxt
net = caffe_pb2.NetParameter()
with open(training_protopath) as f:
    s = f.read()
    txtf.Merge(s, net)
layerNames = [l.name for l in net.layer]

''' A1B: update data layer'''
data_layer = net.layer[layerNames.index('data')]
old_paramstr = data_layer.python_param.param_str
pslist = old_paramstr.split(',')
pslist[0]= str(batch)
new_paramstr = (',').join(pslist)
data_layer.python_param.param_str = new_paramstr
'''End of A1B'''

''' A1C: update ground truth layer'''
gt_layer = net.layer[layerNames.index('ground_truth')]
old_paramstr = gt_layer.python_param.param_str
pslist = old_paramstr.split(',')
pslist[0]= str(batch)
new_paramstr = (',').join(pslist)
gt_layer.python_param.param_str = new_paramstr
'''End of A1C'''

print 'writing', training_protopath
with open(training_protopath, 'w') as f:
    f.write(str(net))
"""End of A1"""

"""A2: Update solver prototxt"""
# ╦ ╦┌─┐┌┬┐┌─┐┌┬┐┌─┐  ┌─┐┌─┐┬  ┬  ┬┌─┐┬─┐  ┌─┐┬─┐┌─┐┌┬┐┌─┐┌┬┐─┐ ┬┌┬┐
# ║ ║├─┘ ││├─┤ │ ├┤   └─┐│ ││  └┐┌┘├┤ ├┬┘  ├─┘├┬┘│ │ │ │ │ │ ┌┴┬┘ │ 
# ╚═╝┴  ─┴┘┴ ┴ ┴ └─┘  └─┘└─┘┴─┘ └┘ └─┘┴└─  ┴  ┴└─└─┘ ┴ └─┘ ┴ ┴ └─ ┴ 
update_solver_dict = {
# 'solver_type':'SGD'
}
extrainfo_dict = {
}
solver_path = args.solver_prototxt
solverproto = CaffeSolver(trainnet_prototxt_path=training_protopath)
solverproto.update_solver(dict(update_solver_dict, **extrainfo_dict))

'''A2B: Add postfix to identify a model version'''
merge_dict = dict(update_solver_dict, **extrainfo_dict)
postfix_str=os.path.basename(training_protopath).split('.')[0]
for key in merge_dict:
    postfix_str += '-'+key+ '-'+ merge_dict[key]
postfix_str += '-'+'batch'+'-'+str(batch)
postfix_str += '_'+str(int(time.time()))
if not args.use_snapshot == '':
    postfix_str += '_'+"usesnapshot_"+os.path.dirname(snapshot_path).split('_')[-1]+'_'+os.path.basename(snapshot_path).split('.')[0]
'''end of A2B'''
snapshot_prefix = solverproto.sp['snapshot_prefix'][1:-1]
snapshot_dirname = os.path.join(os.path.dirname(snapshot_prefix), postfix_str)
if not os.path.isdir(snapshot_dirname): 
    os.makedirs(snapshot_dirname)
snapshot_prefix = '"'+ snapshot_dirname + '/snapshot-"'
print "snapshot will be save to", snapshot_prefix
solverproto.sp['snapshot_prefix'] = snapshot_prefix
solverproto.write(solver_path);

# load the solver
if 'solver_type' in update_solver_dict:
    if update_solver_dict['solver_type'] =='SGD':
        solver = caffe.SGDSolver(solver_path)
else:
    solver = caffe.AdaDeltaSolver(solver_path)
if args.use_snapshot == '':
    solver.net.copy_from(pretrained_model_path) # untrained.caffemodel
else:
    solver.restore(snapshot_path)
"""End of A2"""

"""A3 Setup flow net"""
# ╔═╗┌─┐┌┬┐┬ ┬┌─┐  ┌─┐┬  ┌─┐┬ ┬  ┌┐┌┌─┐┌┬┐
# ╚═╗├┤  │ │ │├─┘  ├┤ │  │ ││││  │││├┤  │ 
# ╚═╝└─┘ ┴ └─┘┴    └  ┴─┘└─┘└┴┘  ┘└┘└─┘ ┴ 
flow_net_model_base='utils/flownet2/models'
flownet_dict={
    '0': 'FlowNet2', 
    'c': 'FlowNet2-c', 
    's': 'FlowNet2-s', 
    'ss': 'FlowNet2-ss', 
    'sss': 'FlowNet2-sss', 
    'css': 'FlowNet2-css'
}
flownet_code = args.flownet_code
deploy_postfix = '_deploy.prototxt.template'
model_postfix = '_weights.caffemodel.h5'
deploy_proto_path = os.path.join(flow_net_model_base, flownet_dict[flownet_code], flownet_dict[flownet_code]+deploy_postfix)
caffe_model_path = os.path.join(flow_net_model_base, flownet_dict[flownet_code], flownet_dict[flownet_code]+model_postfix)
flownet = Flownet(deploy_proto_path, caffe_model_path, args.image_size)

# ┌─┐┌─┐┌┬┐┬ ┬┌─┐  ┌─┐┌─┐┬  ┬┌─┐┌┐┌┌─┐┬ ┬  ┌┐┌┌─┐┌┬┐
# └─┐├┤  │ │ │├─┘  └─┐├─┤│  │├┤ ││││  └┬┘  │││├┤  │ 
# └─┘└─┘ ┴ └─┘┴    └─┘┴ ┴┴─┘┴└─┘┘└┘└─┘ ┴   ┘└┘└─┘ ┴ 
saliencynet = ImageSaliencyNet(saliency_deploy_prototxt_path, saliency_caffemodel_path)


# ╔═╗┌─┐┌┬┐┬ ┬┌─┐  ┌┬┐┌─┐┌┬┐┌─┐┌─┐┌─┐┌┬┐
# ╚═╗├┤  │ │ │├─┘   ││├─┤ │ ├─┤└─┐├┤  │ 
# ╚═╝└─┘ ┴ └─┘┴    ─┴┘┴ ┴ ┴ ┴ ┴└─┘└─┘ ┴ 
print "Loading data..."
train_frame_basedir = '/data/sunnycia/SaliencyDataset/Video/MSU/frames'
train_density_basedir = '/data/sunnycia/SaliencyDataset/Video/MSU/density/sigma32'
validation_frame_basedir = '/data/sunnycia/SaliencyDataset/Image/SALICON/DATA/train_val/val2014/images'
validation_density_basedir = '/data/sunnycia/SaliencyDataset/Image/SALICON/DATA/train_val/val2014/density'
tranining_dataset = VideoDataset(train_frame_basedir, train_density_basedir)
tranining_dataset.setup_video_dataset_flow()
# validation_dataset = StaticDataset(train_frame_basedir, train_density_basedir, debug=debug_mode)

# ╔╦╗╦╔═╗╔═╗
# ║║║║╚═╗║  
# ╩ ╩╩╚═╝╚═╝
plot_figure_dir = '../figure'
## Figure dir
plot_figure_dir = os.path.join(plot_figure_dir, postfix_str)
if not os.path.isdir(plot_figure_dir):
    os.makedirs(plot_figure_dir)
print "Loss figure will be save to", plot_figure_dir

####################################
#  /|            /      /          #
# ( |  ___  ___    ___    ___  ___ #
#   | |   )|   )| |   )| |   )|   )#
#   | |    |__/|| |  / | |  / |__/ #
##############################__/###
tart_time = time.time()

max_iter = 5000000
validation_iter = 1000
plot_iter = 500
epoch=10
idx_counter = 0

x=[]
y=[]
z=[] # validation

plt.plot(x, y)
_step=0
while _step < max_iter:
    if _step%validation_iter==0:
        ##do validation
        pass
    # tranining_dataset.get_frame_pair()
    key_frame,cur_frame, cur_frame_gt= tranining_dataset.get_frame_pair()
    spatial_saliency = saliencynet.compute_saliency()
    flow = flownet.get_optical_flow(key_frame, cur_frame)
    
    key_frame = np.transpose(key_frame, (2, 0, 1))[None, ...]
    cur_frame = np.transpose(cur_frame, (2, 0, 1))[None, ...]
    cur_frame_gt = cur_frame_gt[None, None, ...]
    flow = np.transpose(flow, (2, 0, 1))[None, ...]


    # print flow.shape, key_frame.shape, cur_frame.shape,cur_frame_gt.shape;exit()
    # print frame_minibatch.shape;exit()
    solver.net.blobs['data'].data[...] = key_frame
    solver.net.blobs['flow'].data[...] = flow
    solver.net.blobs['ground_truth'].data[...] = cur_frame_gt
    solver.step(1)

    x.append(_step)
    # y1.append(solver.net.blobs['loss'].data[...].tolist())
    # y2.append(solver.net.blobs['loss'].data[...].tolist())
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

pkl.dump(x, open(os.path.join(plot_figure_dir, "x.pkl"), 'wb'))
pkl.dump(y, open(os.path.join(plot_figure_dir, "y.pkl"), 'wb'))