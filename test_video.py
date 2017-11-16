import imghdr, imageio
from math import floor
import glob, cv2, os, numpy as np, sys, caffe
from utils.tictoc import tic, toc
from Saliencynet import FlowbasedVideoSaliencyNet#, ConsframebasedVideoSaliencyNet
import argparse
caffe.set_mode_gpu()
caffe.set_device(0)

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_prototxt', type=str, default='prototxt/train.prototxt', help='the network prototxt')
    parser.add_argument('--solver_prototxt', type=str, default='prototxt/solver.prototxt', help='the network prototxt')
    parser.add_argument('--flownet_solver_prototxt', type=str, default="utils/flownet2/models/FlowNet2/FlowNet2_deploy.prototxt.template", help='flownet solver')
    parser.add_argument('--flownet_caffe_model', type=str, default="utils/flownet2/models/FlowNet2/FlowNet2_weights.caffemodel.h5", help='flownet caffe model')
    parser.add_argument('--output_type', type=str, required=True, help='Output saliency (video) or (image)')
    parser.add_argument('--allinone', type=bool, default=False)
    parser.add_argument('--video_base', type=str, required=True)
    parser.add_argument('--model_code', type=str, default='v1')
    return parser.parse_args()
print "Parsing arguments..."
args = get_arguments()

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
    if args.model_code=='v3':
        pass

    #MSU dataset
    # test_video_dir = "/data/sunnycia/SaliencyDataset/Video/MSU/videos/"
    #VideoSet dataset
    # output_saliency_video_dir = '/data/sunnycia/SaliencyDataset/Video/MSU/saliency_video'
    # video_base = "/data/sunnycia/SaliencyDataset/Video/VideoSet"
    video_base = args.video_base
    test_video_dir = os.path.join(video_base, "videos")
    output_saliency_video_dir = os.path.join(video_base, 'saliency_video')
    output_saliency_map_dir = os.path.join(video_base, 'saliency_map')


    if not os.path.isdir(output_saliency_video_dir):
        os.makedirs(output_saliency_video_dir)
    # video_path_list = os.listdir(test_video_dir)
    video_path_list = glob.glob(os.path.join(test_video_dir, "*.*"))

    ## V1
    if args.model_code=='v1':
        vs = FlowbasedVideoSaliencyNet(flownet_deploy_path, flownet_caffe_path, video_deploy_path, video_model_path)

    ## V3
    if args.model_code=='v3':
        pass
        # vs = ConsframebasedVideoSaliencyNet(video_deploy_path, video_model_path)

    for video_path in video_path_list:
        video_name = os.path.basename(video_path).split('.')[0].split('_')[0]
        if len(glob.glob(os.path.join(output_saliency_map_dir, args.model_code, video_name+'*'))) != 0:
            print video_name, "already done, pass."
            continue
        else:
            print len(glob.glob(os.path.join(output_saliency_map_dir, video_name+'*')))
            print "Handling",video_name
        vs.setup_video(video_path)
        vs.create_saliency_video()
        fps = vs.video_meta_data['fps']
        if args.output_type=="image":
            vs.setup_video(video_path)
            vs.create_saliency_video()
            output_dir = os.path.join(output_saliency_map_dir, args.model_code)
            vs.dump_predictions_as_images(output_dir, video_name, args.allinone)

        if args.output_type=="video":
            saliency_video_path = os.path.join(output_saliency_video_dir, args.model_code, os.path.basename(video_path))
            if not os.path.isdir(os.path.dirname(saliency_video_path)):
                os.makedirs(os.path.dirname(saliency_video_path))
            if os.path.isfile(saliency_video_path):
                print saliency_video_path, "exists, pass..."
                continue
            vs.dump_predictions_as_video(saliency_video_path, fps)
        print "Done for video", video_path