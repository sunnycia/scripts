import imghdr, imageio
from math import floor
import glob, cv2, os, numpy as np, sys, caffe
from utils.common import tic, toc
from utils.jigsaw import jigsaw
from Saliencynet import FlowbasedVideoSaliencyNet, FramestackbasedVideoSaliencyNet, C3DbasedVideoSaliencyNet,VoxelbasedVideoSaliencyNet
import argparse

caffe.set_mode_gpu()
caffe.set_device(0)

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--videopath', type=str, required=True)
    parser.add_argument('--deploypath',type=str,default='./prototxt/vo-v3_deploy.prototxt')
    parser.add_argument('--modelpath',type=str,default='../training_output/salicon/vo-v3_train_kldloss_withouteuc-batch-1_1510229829/snapshot-_iter_400000.caffemodel')
    parser.add_argument('--layername',type=str,default=None)
    parser.add_argument('--outputdir', type=str, default='../feature_map', help='Output saliency (video) or (image)')

    return parser.parse_args()
print "Parsing arguments..."
args = get_arguments()
preset_list = ['conv1', 'res2a', 'res2b', 'res3a', 'res3b', 'res4a', 'res4b', 'res5a', 'res5b', 'deconv1', 'deconv2', 'predict']
video_path = args.videopath
if not os.path.isfile(video_path):
    print video_path, "not exists, abort."
    exit()

deploy_path = args.deploypath
model_path = args.modelpath
layer_name = args.layername
outputdir = args.outputdir

if not os.path.isdir(outputdir):
    os.makedirs(outputdir)


vs = VoxelbasedVideoSaliencyNet(deploy_proto=deploy_path, caffe_model=model_path, video_length=16, video_size=(112,112), mean_list=[90, 98, 102], infer_type='slide')

vs.setup_video(video_path)
vo_size = vs.video_meta_data['size']
vo_size = (vo_size[0]/10, vo_size[1]/10)

layer_list= []
if layer_name is not None:
    layer_list.append(layer_name)
else:
    layer_list=preset_list

for layer_name in layer_list:
    video_prefix = os.path.basename(video_path).split('.')[0] + '_' + layer_name
    frame_name_wildcard = layer_name+'_channel_%d_frame_%d.jpg'
    cur_dir = os.path.join(outputdir, video_prefix)
    if not os.path.isdir(cur_dir):
        os.makedirs(cur_dir)
    feature_maps = vs.get_feature_map(layer_name) # (channel, length, width, height)

    print np.min(feature_maps), np.max(feature_maps)
    channel_num = len(feature_maps)
    frame_len = len(feature_maps[0])
    for channel in range(len(feature_maps)):
        frames = feature_maps[channel]
        channel_index = str(channel)

        # for i in range(len(frames)):
        for i in range(1):
            frame = frames[i]
            # print np.min(frame), np.max(frame),
            frame = frame - np.min(frame)
            frame = frame / np.max(frame)
            frame = frame * 255
            frame = cv2.resize(frame, dsize = vo_size)
            # print np.min(frame), np.max(frame)
            # print frame.shape;#exit()
            frame_name = frame_name_wildcard % (channel+1, i+1)
            frame_path = os.path.join(cur_dir, frame_name)
            
            cv2.imwrite(frame_path, frame) 

    outputpath = os.path.join(outputdir, video_prefix + '.jpg')


    jigsaw(imageDir=cur_dir, output_path=outputpath, stdsize=vo_size, padding=2)    

