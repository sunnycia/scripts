#coding=utf-8
import cPickle as pkl
from Dataset import VideoDataset
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import argparse, cv2, os, glob, sys, time, shutil
import cPickle as pkl
import numpy as np
import caffe
from random import shuffle
from utils.caffe_tools import CaffeSolver
from caffe.proto import caffe_pb2
import google.protobuf.text_format as txtf
import utils.OpticalFlowToolkit.lib.flowlib as flib
# from validation import MetricValidation
from utils.pymetric.metrics import CC, SIM, KLdiv

from make_caffe_network import trimmed_dense3d_network

caffe.set_mode_gpu()
# caffe.set_device(0)
def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_prototxt', type=str, required=True, help='the network prototxt')
    parser.add_argument('--solver_prototxt', type=str, default='prototxt/solver.prototxt', help='the network prototxt')

    parser.add_argument('--use_snapshot', type=str, default='', help='Snapshot path.')
    parser.add_argument('--pretrained_model', type=str, default='../pretrained_model/c3d_ucf101_iter_40000.caffemodel', help='Pretrained model')
    parser.add_argument('--data_augmentation', type=bool, default=True, help='If use data augmentation techinique')
    
    parser.add_argument('--plot_iter', type=int, default=50, help='training mini batch')
    parser.add_argument('--valid_iter', type=int, default=500, help='training mini batch')
    parser.add_argument('--snapshot_dir', type=str, required=True, help='directory to save model and figure')
    parser.add_argument('--snapshot_iter', type=int, default=5000, help='training mini batch')
    parser.add_argument('--snapshot_in_code', type=bool, default=False, help='save snapshot in code')
    
    parser.add_argument('--training_example_props',type=float, default=0.8, help='training dataset.')
    parser.add_argument('--dataset',type=str, default='msu', help='training dataset.')
    parser.add_argument('--clip_length',type=int,default=16, help='length of video')
    parser.add_argument('--overlap',type=int,default=15, help='dataset overlap')
    parser.add_argument('--batch',type=int,default=25, help='training mini-batch')
    parser.add_argument('--height',type=int,default=25, help='image height')
    parser.add_argument('--width',type=int,default=25, help='image width')
    # parser.add_argument('--connection', type=bool, default=False)
    
    parser.add_argument('--debug', type=int, default=0, help='If debug is ture, a mini set will run into training.Or a complete set will.')
    return parser.parse_args()

print "Parsing arguments..."
args = get_arguments()

snapshot_path = args.use_snapshot
# connection = args.connection

# Check if snapshot exists
if snapshot_path is not '':
    if not os.path.isfile(snapshot_path):
        print snapshot_path, "not exists.Abort"
        exit()

snapshot_dir = args.snapshot_dir

batch = args.batch
train_prototxt = args.train_prototxt
dataset = args.dataset
video_length=args.clip_length
# image_size = args.imagesize
data_augmentation = args.data_augmentation

solver_path = args.solver_prototxt

shutil.copy(train_prototxt, os.path.join(snapshot_dir, os.path.basename(train_prototxt)))
shutil.copy(solver_path, os.path.join(snapshot_dir, os.path.basename(solver_path)))

solver = caffe.SGDSolver(solver_path)
solver.net.copy_from(args.pretrained_model)

print "Loading data..."
if dataset=='msu':
    train_frame_basedir = '/data/sunnycia/SaliencyDataset/Video/MSU/frames'
    train_density_basedir = '/data/sunnycia/SaliencyDataset/Video/MSU/density/sigma32'
elif dataset=='ledov':
    train_frame_basedir = '/data/SaliencyDataset/Video/LEDOV/frames'
    train_density_basedir = '/data/SaliencyDataset/Video/LEDOV/density/sigma32'
elif dataset=='hollywood':
    train_frame_basedir = '/data/SaliencyDataset/Video/ActionInTheEye/Hollywood2/frames'
    train_density_basedir = '/data/SaliencyDataset/Video/ActionInTheEye/Hollywood2/density'
elif dataset == 'ucf':
    train_frame_basedir=''
    train_density_basedir=''
elif train_frame_basedir =='voc':
    train_frame_basedir=''
    train_density_basedir=''
elif train_frame_basedir =='dhf1k':
    train_frame_basedir=''
    train_density_basedir=''
tranining_dataset = VideoDataset(train_frame_basedir, train_density_basedir, img_size=(112,112), bgr_mean_list=[98,102,90], sort='rgb')
tranining_dataset.setup_video_dataset_c3d(overlap=args.overlap, training_example_props=args.training_example_props)

plot_figure_dir = os.path.join(snapshot_dir, 'figure')
if not os.path.isdir(plot_figure_dir):
    os.makedirs(plot_figure_dir)
print "Loss and metric figure will be save to", plot_figure_dir

start_time = time.time()

max_iter = 5000000
valid_iter = args.valid_iter
plot_iter = args.plot_iter
epoch=20
idx_counter = 0
save_model_iter = args.snapshot_iter

plot_dict = {
'x':[], 
'x_valid':[], 

'y_loss':[], 
'y_cc':[], 
'y_sim':[], 
'y_kld':[]
}
 
plt.subplot(4, 1, 1)
plt.plot(plot_dict['x'], plot_dict['y_loss'])
plt.ylabel('loss')
plt.subplot(4, 1, 2)
plt.plot(plot_dict['x_valid'], plot_dict['y_cc'])
plt.ylabel('cc metric')
plt.subplot(4, 1, 3)
plt.plot(plot_dict['x_valid'], plot_dict['y_sim'])
plt.ylabel('sim metric')
plt.subplot(4, 1, 4)
plt.plot(plot_dict['x_valid'], plot_dict['y_kld'])
plt.xlabel('iter')
plt.ylabel('kld metric')

_step=0
while _step < max_iter:
    # if connection:
    #     frame_data, density_data, reference_density = tranining_dataset.get_frame_connection_c3d(mini_batch=batch, phase='training', density_length='full', data_augmentation=data_augmentation)
    # else:
        # frame_data, density_data = tranining_dataset.get_frame_c3d(mini_batch=batch, phase='training', density_length='full', data_augmentation=data_augmentation)
    frame_data, density_data = tranining_dataset.get_frame_c3d(mini_batch=batch, phase='training', density_length='full', data_augmentation=data_augmentation)

    solver.net.blobs['data'].data[...] = frame_data
    solver.net.blobs['gt'].data[...] = density_data
    # if connection:
    #     solver.net.blobs['reference_density'].data[...] = reference_density

    solver.step(1)

    plot_dict['x'].append(_step)
    plot_dict['y_loss'].append(solver.net.blobs['loss'].data[...].tolist())

    # if args.debug==1:
        # layer_list = ['predict_reshape', 'concat2']
        # for layer in layer_list:
            # data = solver.net.blobs[layer].data[...].tolist()
            # print layer, np.mean(data), np.sum(data)


    if _step % valid_iter==0:
        print "Doing validation...", tranining_dataset.num_validation_examples, "validation samples in total."
        tmp_cc = []; tmp_sim = []; tmp_kld = []
        data_tuple = tranining_dataset.get_frame_c3d(mini_batch=batch, phase='validation', density_length='full')
        index = 0
        while data_tuple is not None:
            print index,'\r',
            sys.stdout.flush()
            index += 1
            valid_frame_data, valid_density_data = data_tuple
            solver.net.blobs['data'].data[...] = valid_frame_data
            solver.net.blobs['gt'].data[...] = valid_density_data
            solver.net.forward()
            predictions = solver.net.blobs['predict'].data[...].tolist() # shape like this "8,1,16,112,112"

            ##Calculating metric
            for (prediction, ground_truth) in zip(predictions, valid_density_data):
                ## shape like this "1, 16, 112, 112"
                prediction = np.array(prediction[0]);ground_truth = np.array(ground_truth[0])
                for (pred, gt) in zip(prediction, ground_truth):
                    # print CC(pred, gt), SIM(pred, gt), KLdiv(pred, gt)
                    tmp_cc.append(CC(pred, gt))
                    tmp_sim.append(SIM(pred, gt))
                    tmp_kld.append(KLdiv(pred, gt))
            data_tuple = tranining_dataset.get_frame_c3d(mini_batch=batch, phase='validation', density_length='full')

        # print np.mean(tmp_cc), np.mean(tmp_sim), np.mean(tmp_kld);exit()
        tmp_cc = np.array(tmp_cc)[~np.isnan(tmp_cc)]
        tmp_sim = np.array(tmp_sim)[~np.isnan(tmp_sim)]
        tmp_kld = np.array(tmp_kld)[~np.isnan(tmp_kld)]
        plot_dict['x_valid'].append(_step)
        plot_dict['y_cc'].append(np.mean(tmp_cc))
        plot_dict['y_sim'].append(np.mean(tmp_sim))
        plot_dict['y_kld'].append(np.mean(tmp_kld))

    if _step%plot_iter==0:
        plot_xlength=500
        plt.subplot(4, 1, 1)
        plt.plot(plot_dict['x'][-plot_xlength:], plot_dict['y_loss'][-plot_xlength:])
        plt.ylabel('loss')
        plt.subplot(4, 1, 2)
        plt.plot(plot_dict['x_valid'][-plot_xlength:], plot_dict['y_cc'][-plot_xlength:])
        plt.ylabel('cc metric')
        plt.subplot(4, 1, 3)
        plt.plot(plot_dict['x_valid'][-plot_xlength:], plot_dict['y_sim'][-plot_xlength:])
        plt.ylabel('sim metric')
        plt.subplot(4, 1, 4)
        plt.plot(plot_dict['x_valid'][-plot_xlength:], plot_dict['y_kld'][-plot_xlength:])
        plt.xlabel('iter')
        plt.ylabel('kld metric')

        plt.savefig(os.path.join(plot_figure_dir, "plot"+str(_step)+".png"))
        plt.clf()

        pkl.dump(plot_dict, open(os.path.join(plot_figure_dir, "plot_dict.pkl"), 'wb'))
        
    # bug of saving solverstate, so save caffemodel manually here
    if _step % save_model_iter == 0 and args.snapshot_in_code == True:
        if not os.path.isdir(os.path.dirname(snapshot_prefix.replace('"',''))):
            os.makedirs(os.path.dirname(snapshot_prefix.replace('"','')))
        snapshot_path = snapshot_prefix.replace('"','') + str(_step)+'.caffemodel'
        print 'save model to',snapshot_path
        solver.net.save(snapshot_path)
    _step+=1

