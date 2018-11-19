import cPickle as pkl
import os, glob, sys
import numpy as np
import cv2
import caffe
caffe.set_mode_gpu()
from utils.pymetric.metrics import *#CC, SIM, KLdiv
import argparse
from Dataset import VideoDataset

parser = argparse.ArgumentParser()

parser.add_argument('--working_directory', type=str, required=True)
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--result_path', type=str, required=True)
parser.add_argument('--clip_length', type=int, default=16)

args = parser.parse_args()

working_directory = args.working_directory
dataset = args.dataset
result_path = args.result_path
clip_length = args.clip_length

if dataset =='videoset':
    frame_dir = '/data/SaliencyDataset/Video/VideoSet/ImageSet/Seperate/frame'
    density_dir = '/data/SaliencyDataset/Video/VideoSet/ImageSet/Seperate/density/sigma32'
    fixation_dir = '/data/SaliencyDataset/Video/VideoSet/ImageSet/Seperate/fixation'


if not os.path.isdir(result_path):
    os.makedirs(result_path)
log_name = 'result.txt'
log_path = os.path.join(result_path, log_name)
log_f = open(log_path, 'w')

dataset = VideoDataset(frame_dir, density_dir, img_size=(112,112), bgr_mean_list=[98,102,90], sort='rgb')

video_frame_list = os.listdir(frame_dir)
video_density_list = os.listdir(density_dir)
video_fixation_list = os.listdir(fixation_dir)

### Load test clip list
testset_path_list = glob.glob(os.path.join(working_directory, 'testset*pkl'))
testset_path_list.sort()
solver_path = glob.glob(os.path.join(working_directory, '*solver*prototxt'))[0]
timestamp = working_directory.split('_')[-1]

cc_list = []
sim_list = []
kld_list = []
for testset_path in testset_path_list:
    fold_index = int(os.path.basename(testset_path).split('.')[0].split('_')[-1])
    model_path = os.path.join(working_directory, timestamp+'_'+str(fold_index)+'.caffemodel')
    if not os.path.isfile(model_path):
        print >> log_f, model_path, 'not exist'
        continue
    print >> log_f, 'use model', model_path
    print >> log_f, 'use test set', testset_path,

    solver = caffe.AdaDeltaSolver(solver_path)
    solver.net.copy_from(model_path)

    test_tuple_list = pkl.load(open(testset_path, 'rb'))
    print >> log_f, ' test set length', len(test_tuple_list)
    # print  testset_path_list, test_tuple_list;exit()
    dataset.setup_video_dataset_c3d(overlap=8, training_example_props=0.9)
    dataset.validation_tuple_list = test_tuple_list

    # print dataset.validation_tuple_list[:10], test_tuple_list[:10]
    # exit()
    batch_size=2
    data_tuple = dataset.get_frame_c3d(mini_batch=batch_size, phase='validation', density_length='full')
    tmp_cc = [];tmp_sim = [];tmp_kld=[];tmp_aucj = [];tmp_aucb = []
    index = 0
    while data_tuple is not None:
        print index,'of',len(test_tuple_list), '\r',
        sys.stdout.flush()
        frame_data, density_data = data_tuple
        solver.net.blobs['data'].data[...] = frame_data
        solver.net.blobs['ground_truth'].data[...] = density_data
        solver.net.forward()
        predictions = solver.net.blobs['predict'].data[...].tolist() # shape like this "8,1,16,112,112"
        for (prediction, ground_truth) in zip(predictions, density_data):
            ## shape like this "1, 16, 112, 112"
            prediction = np.array(prediction[0]);ground_truth = np.array(ground_truth[0])
            for (pred, gt) in zip(prediction, ground_truth):
                # print CC(pred, gt), SIM(pred, gt), KLdiv(pred, gt)
                tmp_cc.append(CC(pred, gt))
                tmp_sim.append(SIM(pred, gt))
                tmp_kld.append(KLdiv(pred, gt))
        data_tuple = dataset.get_frame_c3d(mini_batch=batch_size, phase='validation', density_length='full')
        index += 1
    tmp_cc = np.array(tmp_cc)[~np.isnan(tmp_cc)]
    tmp_sim = np.array(tmp_sim)[~np.isnan(tmp_sim)]
    tmp_kld = np.array(tmp_kld)[~np.isnan(tmp_kld)]
    cc_list.append(np.mean(tmp_cc))
    sim_list.append(np.mean(tmp_sim))
    kld_list.append(np.mean(tmp_kld))

    print >> log_f, 'cc\tsim\tkld\t'
    print >> log_f, np.mean(tmp_cc), np.mean(tmp_sim), np.mean(tmp_kld);#exit()
    print >> log_f


print >>  log_f, 'final result:'
print >> log_f, 'cc\tsim\tkld\t'
print >> log_f, np.mean(cc_list), np.mean(sim_list), np.mean(kld_list);#exit()

log_f.close()