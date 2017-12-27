# IIIII NN   NN FFFFFFF EEEEEEE RRRRRR  EEEEEEE NN   NN  CCCCC  EEEEEEE 
#  III  NNN  NN FF      EE      RR   RR EE      NNN  NN CC    C EE      
#  III  NN N NN FFFF    EEEEE   RRRRRR  EEEEE   NN N NN CC      EEEEE   
#  III  NN  NNN FF      EE      RR  RR  EE      NN  NNN CC    C EE      
# IIIII NN   NN FF      EEEEEEE RR   RR EEEEEEE NN   NN  CCCCC  EEEEEEE 

import imghdr, imageio
from math import floor
import glob, cv2, os, numpy as np, sys, caffe
from utils.common import tic, toc
from Saliencynet import FlowbasedVideoSaliencyNet, FramestackbasedVideoSaliencyNet, C3DbasedVideoSaliencyNet,VoxelbasedVideoSaliencyNet
import argparse
caffe.set_mode_gpu()
caffe.set_device(0)

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--flownet_solver_prototxt', type=str, default="utils/flownet2/models/FlowNet2/FlowNet2_deploy.prototxt.template", help='flownet solver')
    parser.add_argument('--flownet_caffe_model', type=str, default="utils/flownet2/models/FlowNet2/FlowNet2_weights.caffemodel.h5", help='flownet caffe model')
    parser.add_argument('--output_type', type=str, required=True, help='Output saliency (video) or (image)')
    parser.add_argument('--allinone', type=bool, default=False)
    # parser.add_argument('--video_base', type=str, required=True)
    parser.add_argument('--test_base', type=str, required=True)

    # parser.add_argument('--output_dir', type=str, required=True)
    # parser.add_argument('--model_code', type=str, required=True)
    parser.add_argument('--model_code', type=str, required=True)
    parser.add_argument('--videolength', type=int, default=5)
    parser.add_argument('--video_deploy_path',type=str,default='./prototxt/vo-v3_deploy.prototxt')
    parser.add_argument('--video_model_path',type=str,default='../training_output/salicon/vo-v3_train_kldloss_withouteuc-batch-1_1510229829/snapshot-_iter_400000.caffemodel')
    parser.add_argument('--infertype', type=str,default='slice')
    parser.add_argument('--inferoverlap', type=int,default='15')
    parser.add_argument('--threshold', type=float, default=0)

    return parser.parse_args()
print "Parsing arguments..."
args = get_arguments()
output_type = args.output_type
threshold = args.threshold

if __name__ =='__main__':
    # model_base = '../training_output/salicon'
    # subdirs = [name for name in os.listdir(model_base) if os.path.isdir(os.path.join(model_base, name))]
    flownet_deploy_path = args.flownet_solver_prototxt
    flownet_caffe_path = args.flownet_caffe_model

    video_deploy_path = ''
    video_model_path = ''
    if args.model_code=='v1':
        video_deploy_path = "./prototxt/vo-v1_deploy.prototxt"
        video_model_path = "../training_output/salicon/vo-v1_train_kldloss_withouteuc-batch-1_1510204874/snapshot-_iter_450000.caffemodel"
    else:
        # video_deploy_path = "./prototxt/vo-v3_deploy.prototxt"
        # video_model_path = "../training_output/salicon/vo-v3_train_kldloss_withouteuc-batch-1_1510229829/snapshot-_iter_400000.caffemodel"
        video_deploy_path = args.video_deploy_path
        video_model_path = args.video_model_path

    video_base='';saliency_video_base='';saliency_map_base=''
    if args.test_base == 'videoset':
        video_base = '/data/sunnycia/SaliencyDataset/Video/VideoSet/Videos/videos_origin'
        saliency_video_base = '/data/sunnycia/SaliencyDataset/Video/VideoSet/Results/saliency_video'
        saliency_map_base = '/data/sunnycia/SaliencyDataset/Video/VideoSet/Results/saliency_map'
    elif args.test_base == 'msu':
        video_base = '/data/sunnycia/SaliencyDataset/Video/MSU/videos'
        saliency_video_base = '/data/sunnycia/SaliencyDataset/Video/MSU/saliency_video'
        saliency_map_base=  '/data/sunnycia/SaliencyDataset/Video/MSU/saliency_map'
    elif args.test_base == 'diem':
        pass

    model_name = os.path.dirname(video_model_path).split('/')[-1] + '_'+ os.path.basename(video_model_path).split('.')[0] + '_threshold'+str(threshold)
    video_path_list = glob.glob(os.path.join(video_base, "*.*"))
    saliency_video_dir = os.path.join(saliency_video_base, model_name)
    saliency_map_dir = os.path.join(saliency_map_base, model_name)
    if output_type=='video':
        if not os.path.isdir(saliency_video_dir):
            os.makedirs(saliency_video_dir)
    if output_type=='image':
        if not os.path.isdir(saliency_map_dir):
            os.makedirs(saliency_map_dir)

    ## V1
    if args.model_code=='v1':
        vs = FlowbasedVideoSaliencyNet(flownet_deploy_path, flownet_caffe_path, video_deploy_path, video_model_path)

    ## V3
    if args.model_code=='v3':
        vs = FramestackbasedVideoSaliencyNet(video_deploy_path, video_model_path,video_length=args.videolength, infer_type=args.infertype)
        # vs = FramestackbasedVideoSaliencyNet(video_deploy_path, video_model_path)

    ## v4 
    if args.model_code=='v4':
        vs = C3DbasedVideoSaliencyNet(video_deploy_path,video_model_path,video_length=16,video_size=(112,112),rgb_mean_value=[90,98,102])
    if args.model_code=='v4-2':
        vs = VoxelbasedVideoSaliencyNet(deploy_proto=video_deploy_path, caffe_model=video_model_path, video_length=16, video_size=(112,112),mean_list=[90, 98,102], infer_type=args.infertype)

    for video_path in video_path_list:
        vs.setup_video(video_path)
        vs.create_saliency_video(threshold=threshold, overlap=args.inferoverlap)
        if args.output_type=="image":
            video_name = os.path.basename(video_path).split('.')[0].split('_')[0]
            if len(glob.glob(os.path.join(saliency_map_dir, video_name+'*'))) != 0:
                print video_name, "already done, pass."
                continue
            else:
                print len(glob.glob(os.path.join(saliency_map_dir, video_name+'*')))
                print "Handling",video_name
            vs.dump_predictions_as_images(saliency_map_dir, video_name, args.allinone)

        if args.output_type=="video":
            saliency_video_path = os.path.join(saliency_video_dir, os.path.basename(video_path))
            if not os.path.isdir(os.path.dirname(saliency_video_path)):
                os.makedirs(os.path.dirname(saliency_video_path))
            if os.path.isfile(saliency_video_path):
                print saliency_video_path, "exists, pass..."
                continue
            fps = vs.video_meta_data['fps']
            vs.dump_predictions_as_video(saliency_video_path, fps)

        print "Done for video", video_path;#exit()