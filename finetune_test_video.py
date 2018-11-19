'''
Goal/description of this script:
  finetune a pretrained model on a dataset, 
   split the dataset into N fold, 
   finetune N times, 
   train on (N-1)/N data, 
   test on the 1/N data, 
   calculte metric, then average the N results

   the average would be the model performance on this dataset.

'''



import os, sys
import time
import cv2
from Dataset import VideoDataset, get_frame_and_density_dir
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import cPickle as pkl
import caffe
import shutil
caffe.set_mode_gpu()
import argparse
from utils.pymetric.metrics import CC, SIM, KLdiv

parser = argparse.ArgumentParser()
parser.add_argument('--pretrained_model', type=str, required=True)
parser.add_argument('--network_prototxt', type=str, required=True)
parser.add_argument('--solver_prototxt', type=str, required=True)
parser.add_argument('--epoch', type=int, default=10)
# parser.add_argument('--base_lr', type=float, default=0.0001)

parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--batch', type=int, default=2)
parser.add_argument('--fold', type=int, required=True)
parser.add_argument('--valid_iter', type=int, default=3000)
parser.add_argument('--overlap', type=int, default=8)

parser.add_argument('--output_log', type=str, default='fintune_test_log.txt')
parser.add_argument('--output_results', type=str, default='fintune_test_log.txt')
parser.add_argument('--savemodel_dir', type=str, default='../training_output/finetune')


args = parser.parse_args()

pretrained_model = args.pretrained_model
network_prototxt = args.network_prototxt
solver_prototxt = args.solver_prototxt
epoch = args.epoch

time_stamp = str(int(time.time()))
dataset = args.dataset
fold = args.fold
valid_iter = args.valid_iter
batch = args.batch
output_log = args.output_log
output_results = args.output_results
savemodel_dir = args.savemodel_dir
pretrained_model_info = os.path.dirname(pretrained_model).split('_')[-1] +'_'+ os.path.basename(pretrained_model).split('.')[0]
savemodel_dir = os.path.join(savemodel_dir, pretrained_model_info, dataset+'_'+time_stamp)
if not os.path.isdir(savemodel_dir):
    os.makedirs(savemodel_dir)

## copy network and solver to savemodel_dir
solver_path = os.path.join(savemodel_dir, os.path.basename(solver_prototxt))
network_path = os.path.join(savemodel_dir, os.path.basename(network_prototxt))
shutil.copy(solver_prototxt, solver_path)
shutil.copy(network_prototxt, network_path)
solver_af = open(solver_path, 'a+')

print >> solver_af, '\nnet:"'+network_path+'"'
solver_af.close()

log_path = os.path.join(savemodel_dir, output_log)
result_path = os.path.join(savemodel_dir, output_results)

### finetune
frame_basedir, density_basedir=get_frame_and_density_dir(dataset)
if not  frame_basedir:
    print "None preset dataset."
    exit()

dataset = VideoDataset(frame_basedir, density_basedir, 
    img_size=(112,112), bgr_mean_list=[98,102,90], sort='rgb')
dataset.setup_video_dataset_c3d_with_fold(overlap=args.overlap, fold=fold)

output_log = time_stamp+'_'+output_log
#save some critical information
log_f = open(os.path.join(savemodel_dir, output_log), 'w')

print >> log_f, 'dataset'
print >> log_f, 'fold', fold
print >> log_f, 'epoch', epoch

solver_rf = open(solver_path, 'r')
solver_lines = solver_rf.readlines()
print >> log_f, 'solver info'
for solver_line in solver_lines:
    print >> log_f, solver_line
solver_rf.close()



##?? save list_chunk
list_chunk_path= os.path.join(savemodel_dir, 'list_chunk_'+time_stamp+'.pkl')
pkl.dump(np.array(dataset.list_chunk), open(list_chunk_path, 'wb'))

for fold_index in range(fold):
    begin_time_stamp = time.time()
    print >> log_f, 'begin fold', fold, 'at', time.asctime(time.localtime(time.time()))
    solver = caffe.AdaDeltaSolver(solver_path)
    solver.net.copy_from(pretrained_model)

    dataset.get_training_validation_sample_from_chunk(fold_index)
    testset_tuple_list_path = os.path.join(savemodel_dir, 'testset_'+time_stamp+'_'+str(fold_index)+'.pkl')
    pkl.dump(np.array(dataset.validation_tuple_list), open(testset_tuple_list_path, 'wb'))

    figure_name = 'plot_'+time_stamp+'_'+str(fold_index)+'.png'
    plot_dict = {
    'x_valid':[], 
    'y_loss':[], 
    'y_cc':[], 
    'y_sim':[], 
    'y_kld':[]
    }
    plt.subplot(4, 1, 1)
    plt.plot(plot_dict['x_valid'], plot_dict['y_loss'])
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

    itersize = 0
    loss_list = []
    while dataset.num_epoch < epoch:
        # frame_data, density_data = dataset.get_frame_c3d(mini_batch=batch, phase='training', density_length='one')
        frame_data, density_data = dataset.get_frame_c3d(mini_batch=batch, phase='training', density_length='full', data_augmentation=False)

        solver.net.blobs['data'].data[...] = frame_data
        solver.net.blobs['ground_truth'].data[...] = density_data
        solver.step(1)
        loss = solver.net.blobs['loss'].data[...]

        loss_list.append(loss)
        # print 'before forward'
        # print 'after forward'
        if itersize % valid_iter==0:
            tmp_cc=[];tmp_sim=[];tmp_kld=[]
            data_tuple = dataset.get_frame_c3d(mini_batch=batch, phase='validation', density_length='full')
            index = 0
            while data_tuple is not None:
                print index,'\r',
                sys.stdout.flush()
                index += 1
                valid_frame_data, valid_density_data = data_tuple
                solver.net.blobs['data'].data[...] = valid_frame_data
                solver.net.blobs['ground_truth'].data[...] = valid_density_data
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
                data_tuple = dataset.get_frame_c3d(mini_batch=batch, phase='validation', density_length='full')
         
            tmp_cc = np.array(tmp_cc)[~np.isnan(tmp_cc)]
            tmp_sim = np.array(tmp_sim)[~np.isnan(tmp_sim)]
            tmp_kld = np.array(tmp_kld)[~np.isnan(tmp_kld)]

            plot_dict['x_valid'].append(itersize)
            plot_dict['y_loss'].append(np.mean(loss_list))
            plot_dict['y_cc'].append(np.mean(tmp_cc))
            plot_dict['y_sim'].append(np.mean(tmp_sim))
            plot_dict['y_kld'].append(np.mean(tmp_kld))
            # print plot_dict['x_valid'],plot_dict['y_cc'], plot_dict['y_loss']
            plt.subplot(4, 1, 1)
            plt.plot(plot_dict['x_valid'], plot_dict['y_loss'])
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

            plt.savefig(os.path.join(savemodel_dir, figure_name))
            plt.clf()
            loss_list= []
        itersize += 1

    model_path = os.path.join(savemodel_dir, time_stamp+'_'+str(fold_index)+'.caffemodel')
    solver.net.save(model_path)
    print >> log_f,  model_path, 'saved'
    fold_time_duration_in_seconds = time.time() - begin_time_stamp
    print >> log_f, fold_time_duration_in_seconds, 'seconds have passed'

    ## test
    test_tuple_list = dataset.validation_tuple_list
    # save test tuple list

log_f.close()