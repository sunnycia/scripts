 #   _____    _____   _____   _   _   ______   ______   _____  
 #  / ____|  / ____| |_   _| | \ | | |  ____| |  ____| |  __ \ 
 # | (___   | (___     | |   |  \| | | |__    | |__    | |__) |
 #  \___ \   \___ \    | |   | . ` | |  __|   |  __|   |  _  / 
 #  ____) |  ____) |  _| |_  | |\  | | |      | |____  | | \ \ 
 # |_____/  |_____/  |_____| |_| \_| |_|      |______| |_|  \_\
                                                             
import imghdr, imageio
from math import floor
import glob, cv2, os, numpy as np, sys, caffe
from utils.common import tic, toc
from Saliencynet import FlowbasedVideoSaliencyNet, FramestackbasedVideoSaliencyNet, C3DbasedVideoSaliencyNet,VoxelbasedVideoSaliencyNet
import argparse
import time
caffe.set_mode_gpu()
caffe.set_device(0)

def get_arguments():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--flownet_solver_prototxt', type=str, default="utils/flownet2/models/FlowNet2/FlowNet2_deploy.prototxt.template", help='flownet solver')
    # parser.add_argument('--flownet_caffe_model', type=str, default="utils/flownet2/models/FlowNet2/FlowNet2_weights.caffemodel.h5", help='flownet caffe model')
    parser.add_argument('--output_type', type=str, default='image', help='Output saliency (video) or (image)')
    parser.add_argument('--allinone', type=bool, default=False)
    parser.add_argument('--dataset', type=str, required=True)
    # parser.add_argument('--model_base', type=str, default='../training_output/salicon')

    # parser.add_argument('--output_dir', type=str, required=True)
    # parser.add_argument('--model_code', type=str, required=True)
    # parser.add_argument('--model_code', type=str, default='v4-2')
    parser.add_argument('--clip_length', type=int, default=16)

    parser.add_argument('--model_base_dir',type=str,default='')
    parser.add_argument('--video_deploy_path',type=str,default='')
    parser.add_argument('--video_model_path',type=str,default='')

    parser.add_argument('--infertype', type=str,default='slide')
    parser.add_argument('--overlap', type=int,default=8)
    parser.add_argument('--threshold', type=float, default=0)
    parser.add_argument('--debug', type=bool, default=False)

    return parser.parse_args()
print "Parsing arguments..."
args = get_arguments()
output_type = args.output_type
model_base_dir = args.model_base_dir
threshold = args.threshold
overlap = args.overlap
debug = args.debug

if __name__ =='__main__':
    # model_base = '../training_output/salicon'
    # model_base = args.model_base
    # subdirs = [name for name in os.listdir(model_base) if os.path.isdir(os.path.join(model_base, name))]
    
    # flownet_deploy_path = args.flownet_solver_prototxt
    # flownet_caffe_path = args.flownet_caffe_model

    video_deploy_path = ''
    video_model_path = ''
    # if args.model_code=='v1':
    #     video_deploy_path = "./prototxt/vo-v1_deploy.prototxt"
    #     video_model_path = "../training_output/salicon/vo-v1_train_kldloss_withouteuc-batch-1_1510204874/snapshot-_iter_450000.caffemodel"
    # elif model_base_dir != '':
    #     model_dir = os.path.join(model_base, model_base_dir)
        
    #     video_deploy_path= glob.glob(os.path.join(model_dir, 'vo*prototxt'))[0]
    #     model_list = glob.glob(os.path.join(model_dir, '*.caffemodel'))
    #     newest_model = max(model_list, key=os.path.getctime)
    #     video_model_path=newest_model
    # else:
        # video_deploy_path = "./prototxt/vo-v3_deploy.prototxt"
        # video_model_path = "../training_output/salicon/vo-v3_train_kldloss_withouteuc-batch-1_1510229829/snapshot-_iter_400000.caffemodel"
    video_deploy_path = args.video_deploy_path
    video_model_path = args.video_model_path

    video_base='';saliency_video_base='';saliency_map_base=''
    if args.dataset == 'videoset':
        video_base = '/data/SaliencyDataset/Video/VideoSet/Videos/videos_origin'
        saliency_video_base = '/data/SaliencyDataset/Video/VideoSet/Results/saliency_video'
        saliency_map_base = '/data/SaliencyDataset/Video/VideoSet/Results/saliency_map'
    elif args.dataset == 'msu':
        video_base = '/data/SaliencyDataset/Video/MSU/videos'
        saliency_video_base = '/data/SaliencyDataset/Video/MSU/saliency_video'
        saliency_map_base=  '/data/SaliencyDataset/Video/MSU/saliency_map'
    elif args.dataset == 'ledov':
        video_base = '/data/SaliencyDataset/Video/LEDOV/videos'
        saliency_video_base = '/data/SaliencyDataset/Video/LEDOV/saliency_video'
        saliency_map_base = '/data/SaliencyDataset/Video/LEDOV/saliency_map'
    elif args.dataset == 'diem':
        video_base = '/data/SaliencyDataset/Video/DIEM/videos'
        saliency_video_base = '/data/SaliencyDataset/Video/DIEM/saliency_video'
        saliency_map_base = '/data/SaliencyDataset/Video/DIEM/saliency_map'
    elif args.dataset == 'gazecom':
        video_base = '/data/SaliencyDataset/Video/GAZECOM/videos'
        saliency_video_base = '/data/SaliencyDataset/Video/GAZECOM/saliency_video'
        saliency_map_base = '/data/SaliencyDataset/Video/GAZECOM/saliency_map'
    elif args.dataset == 'coutort2':
        video_base = '/data/SaliencyDataset/Video/Coutort2/videos'
        saliency_video_base = '/data/SaliencyDataset/Video/Coutort2/saliency_video'
        saliency_map_base = '/data/SaliencyDataset/Video/Coutort2/saliency_map'
    elif args.dataset == 'hollywood':
        video_base = '/data/SaliencyDataset/Video/ActionInTheEye/Hollywood2/Hollywood2-actions/AVIClips'
        saliency_video_base = '/data/SaliencyDataset/Video/ActionInTheEye/Hollywood2/saliency_video'
        saliency_map_base = '/data/SaliencyDataset/Video/ActionInTheEye/Hollywood2/saliency_map'
    elif args.dataset == 'dhf1k':
        video_base = '/data/SaliencyDataset/Video/DHF1K/videos'
        saliency_video_base = '/data/SaliencyDataset/Video/DHF1K/saliency_video'
        saliency_map_base = '/data/SaliencyDataset/Video/DHF1K/saliency_map'
    else:
        raise NotImplementedError
    # else:
    #     raise 
    model_name = os.path.dirname(video_model_path).split('/')[-1] + '_'+ os.path.basename(video_model_path).split('.')[0] + '_threshold'+str(threshold) + '_overlap'+str(overlap)
    video_path_list = glob.glob(os.path.join(video_base, "*.*"))
    video_path_list.sort()
    saliency_video_dir = os.path.join(saliency_video_base, model_name)
    saliency_map_dir = os.path.join(saliency_map_base, model_name)
    if output_type=='video':
        if not os.path.isdir(saliency_video_dir):
            os.makedirs(saliency_video_dir)
    if output_type=='image':
        if not os.path.isdir(saliency_map_dir):
            os.makedirs(saliency_map_dir)

    # ## V1
    # if args.model_code=='v1':
    #     vs = FlowbasedVideoSaliencyNet(flownet_deploy_path, flownet_caffe_path, video_deploy_path, video_model_path)

    # ## V3
    # if args.model_code=='v3':
    #     vs = FramestackbasedVideoSaliencyNet(video_deploy_path, video_model_path,video_length=args.clip_length, infer_type=args.infertype)
    #     # vs = FramestackbasedVideoSaliencyNet(video_deploy_path, video_model_path)

    ## v4 
    # if args.model_code=='v4':
    #     vs = C3DbasedVideoSaliencyNet(video_deploy_path,video_model_path,video_length=16,video_size=(112,112),rgb_mean_value=[90,98,102])
    # if args.model_code=='v4-2':
        # vs = VoxelbasedVideoSaliencyNet(deploy_proto=video_deploy_path, caffe_model=video_model_path, video_length=16, video_size=(112,112),mean_list=[90, 98,102], infer_type=args.infertype)
    vs = VoxelbasedVideoSaliencyNet(deploy_proto=video_deploy_path, caffe_model=video_model_path, video_length=args.clip_length, video_size=(112,112),mean_list=[90, 98,102], infer_type=args.infertype)

    total_time = 0
    if debug:
        video_path_list = video_path_list[:10]
    for video_path in video_path_list:
        vs.setup_video(video_path)
        print 'Done for setup'
        # globals()['tt'] = time.clock()
        start_time = time.time()
        vs.create_saliency_video(threshold=threshold, overlap=overlap)
        # interv = time.clock() - globals()['tt']
        end_time = time.time()
        interv = end_time - start_time
        total_time += interv
        if debug:
            print interv, 'seconds costs'
            continue

        if args.output_type=="image":
            # video_name = os.path.basename(video_path).split('.')[0].split('_')[0]
            video_name = os.path.basename(video_path).split('.')[0]
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
    print total_time/len(video_path_list)