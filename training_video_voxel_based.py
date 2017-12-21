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
from caffe.proto import caffe_pb2
import google.protobuf.text_format as txtf
import utils.OpticalFlowToolkit.lib.flowlib as flib

caffe.set_mode_gpu()
# caffe.set_device(0)
def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_prototxt', type=str, required=True, help='the network prototxt')
    parser.add_argument('--solver_prototxt', type=str, default='prototxt/solver.prototxt', help='the network prototxt')
    parser.add_argument('--use_snapshot', type=str, default='', help='Snapshot path.')
    parser.add_argument('--use_model', type=str, default='../pretrained_model/c3d_ucf101_iter_40000.caffemodel', help='Pretrained model')
    
    parser.add_argument('--plotiter', type=int, default=50, help='training mini batch')
    parser.add_argument('--validiter', type=int, default=500, help='training mini batch')
    parser.add_argument('--savemodeliter', type=int, default=1500, help='training mini batch')
    parser.add_argument('--snapshotincode', type=bool, default=False, help='save snapshot in code')
    
    parser.add_argument('--trainingexampleprops',type=float, default=0.8, help='training dataset.')
    parser.add_argument('--trainingbase',type=str, default='msu', help='training dataset.')
    parser.add_argument('--videolength',type=int,default=16, help='length of video')
    parser.add_argument('--overlap',type=int,default=15, help='dataset overlap')
    parser.add_argument('--batch',type=int,default=25, help='length of video')
    parser.add_argument('--imagesize', type=tuple, default=(112,112))
    
    parser.add_argument('--lastlayer', type=str, default='fc8_new')
    parser.add_argument('--staticsolver',type=bool,default=False)
    parser.add_argument('--debug', type=bool, default=False, help='If debug is ture, a mini set will run into training.Or a complete set will.')

    parser.add_argument('--extramodinfo', type=str, default='', help="add extra model information")
    return parser.parse_args()

print "Parsing arguments..."
args = get_arguments()

pretrained_model_path= args.use_model
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
training_base = args.trainingbase
video_length=args.videolength
image_size = args.imagesize
"""A2: Update solver prototxt"""
# ╦ ╦┌─┐┌┬┐┌─┐┌┬┐┌─┐  ┌─┐┌─┐┬  ┬  ┬┌─┐┬─┐  ┌─┐┬─┐┌─┐┌┬┐┌─┐┌┬┐─┐ ┬┌┬┐
# ║ ║├─┘ ││├─┤ │ ├┤   └─┐│ ││  └┐┌┘├┤ ├┬┘  ├─┘├┬┘│ │ │ │ │ │ ┌┴┬┘ │ 
# ╚═╝┴  ─┴┘┴ ┴ ┴ └─┘  └─┘└─┘┴─┘ └┘ └─┘┴└─  ┴  ┴└─└─┘ ┴ └─┘ ┴ ┴ └─ ┴ 
update_solver_dict = {
# 'solver_type':'RMSPROP',
'display':'1',
# 'base_lr': '0.00001',
# 'weight_decay': '0.000005',
# 'momentum': '0.95',
# 'lr_policy':'"step"',
# 'stepsize':'500',
'snapshot':str(args.savemodeliter)
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
postfix_str += '-'+args.extramodinfo
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

if args.staticsolver is True:
    pass
else:
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

# ╔═╗┌─┐┌┬┐┬ ┬┌─┐  ┌┬┐┌─┐┌┬┐┌─┐┌─┐┌─┐┌┬┐
# ╚═╗├┤  │ │ │├─┘   ││├─┤ │ ├─┤└─┐├┤  │ 
# ╚═╝└─┘ ┴ └─┘┴    ─┴┘┴ ┴ ┴ ┴ ┴└─┘└─┘ ┴ 
print "Loading data..."
if training_base=='msu':
    train_frame_basedir = '/data/sunnycia/SaliencyDataset/Video/MSU/frames'
    train_density_basedir = '/data/sunnycia/SaliencyDataset/Video/MSU/density/sigma32'
elif training_base=='ledov':
    train_frame_basedir = '/data/sunnycia/SaliencyDataset/Video/LEDOV/frames'
    train_density_basedir = '/data/sunnycia/SaliencyDataset/Video/LEDOV/density/sigma32'


tranining_dataset = VideoDataset(train_frame_basedir, train_density_basedir, img_size=(112,112), bgr_mean_list=[98,102,90], sort='rgb')
tranining_dataset.setup_video_dataset_c3d(overlap=args.overlap, training_example_props=args.trainingexampleprops)

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
start_time = time.time()

max_iter = 5000000
validation_iter = args.validiter
plot_iter = args.plotiter
epoch=20
idx_counter = 0
save_model_iter = args.savemodeliter

x=[]
x2 = []
y1=[]
y2=[]
z=[] # validation

# plt.plot(x, y1, color='red', label='training')
# plt.plot(x2, y2, color='orange', label='validation')
# plt.legend()
plt.plot(x,y1, x2,y2,':')
# plt.plot(x,y2)
_step=0
while _step < max_iter:
    if _step%validation_iter==0:
        ##do validation
        pass
    # print _step, 1
    frame_data, density_data = tranining_dataset.get_frame_c3d(mini_batch=batch, phase='training', density_length='one')
    # print _step, 2

    solver.net.blobs['data'].data[...] = frame_data
    solver.net.blobs['ground_truth'].data[...] = density_data
    solver.step(1)
    # print _step, 3

    x.append(_step)
    y1.append(solver.net.blobs['loss'].data[...].tolist())

    if args.debug==1:
        #    ___    ____   ___   __  __  _____
        #   / _ \  / __/  / _ ) / / / / / ___/
        #  / // / / _/   / _  |/ /_/ / / (_ / 
        # /____/ /___/  /____/ \____/  \___/  
        layer_list = ['predict_reshape', 'concat2']
        for layer in layer_list:
            data = solver.net.blobs[layer].data[...].tolist()
            print layer, np.mean(data), np.sum(data)    
        # exit()

    # y2.append(solver.net.blobs['loss5'].data[...].tolist())
    if _step%validation_iter==0:
        print "Doing validation...", tranining_dataset.num_validation_examples, "validation samples in total."
        data_tuple = tranining_dataset.get_frame_c3d(mini_batch=batch, phase='validation', density_length='one')
        total_loss = 0
        total_sample = 0
        while data_tuple is not None:
            print "index in validation epoch:", tranining_dataset.index_in_validation_epoch,'\r',
            valid_frame_data, valid_density_data = data_tuple
            solver.net.blobs['data'].data[...] = valid_frame_data
            solver.net.blobs['ground_truth'].data[...] = valid_density_data
            solver.net.forward()
            loss = solver.net.blobs['loss'].data[...].tolist()

            total_loss += loss
            total_sample += len(valid_frame_data)
            data_tuple = tranining_dataset.get_frame_c3d(mini_batch=batch, phase='validation', density_length='one')
            # print loss,'\r',
        valid_loss = total_loss/float(total_sample)
        print valid_loss
        x2.append(_step)
        y2.append(valid_loss)
    else:
        # y2.append(np.nan)
        pass
    plt.plot(x,y1, x2,y2,':')
    # plt.plot(x, y1, color='orange', label='training')
    # plt.plot(x2, y2, color='red', label='validation')
    # print y2
    # plt.plot(x,y2)

    if _step%plot_iter==0:
        plt.xlabel('Iter')
        plt.ylabel('loss')
        # plt.legend()
        plt.savefig(os.path.join(plot_figure_dir, "plot"+str(_step)+".png"))
        plt.clf()
        plot_dict = {'tr_x':x, 'tr_y':y1, 'va_x':x2,'va_y':y2}

        pkl.dump(plot_dict, open(os.path.join(plot_figure_dir, "plot_dict.pkl"), 'wb'))
        
    # bug of saving solverstate, so save caffemodel manually here
    if _step % save_model_iter == 0 and args.snapshotincode == True:
        if not os.path.isdir(os.path.dirname(snapshot_prefix.replace('"',''))):
            os.makedirs(os.path.dirname(snapshot_prefix.replace('"','')))
        snapshot_path = snapshot_prefix.replace('"','') + str(_step)+'.caffemodel'
        print 'save model to',snapshot_path
        solver.net.save(snapshot_path)
    _step+=1

