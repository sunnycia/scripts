import imghdr, imageio
from math import floor
import glob, cv2, os, numpy as np, sys, caffe
from utils.tictoc import tic, toc
from Saliencynet import FlowbasedVideoSaliencyNet
caffe.set_mode_gpu()
caffe.set_device(0)

if __name__ =='__main__':
    # model_base = '../training_output/salicon'
    # subdirs = [name for name in os.listdir(model_base) if os.path.isdir(os.path.join(model_base, name))]
    flownet_deploy_path = "utils/flownet2/models/FlowNet2/FlowNet2_deploy.prototxt.template"
    flownet_caffe_path = "utils/flownet2/models/FlowNet2/FlowNet2_weights.caffemodel.h5"

    video_deploy_path = "./prototxt/vo-v1_deploy.prototxt"
    video_model_path = "../training_output/salicon/vo-v1_train_kldloss_withouteuc-batch-1_1510204874/snapshot-_iter_450000.caffemodel"

    model_version = video_model_path.split('/')[-2].split('.')[0]

    #MSU dataset
    # test_video_dir = "/data/sunnycia/SaliencyDataset/Video/MSU/videos/"
    #VideoSet dataset
    # output_saliency_video_dir = '/data/sunnycia/SaliencyDataset/Video/MSU/saliency_video'

    test_video_dir = "/data/sunnycia/SaliencyDataset/Video/VideoSet/videos"
    output_saliency_video_dir = '/data/sunnycia/SaliencyDataset/Video/VideoSet/saliency_video'
    if not os.path.isdir(output_saliency_video_dir):
        os.makedirs(output_saliency_video_dir)
    # video_path_list = os.listdir(test_video_dir)
    video_path_list = glob.glob(os.path.join(test_video_dir, "*_left.*"))
    vs = FlowbasedVideoSaliencyNet(flownet_deploy_path, flownet_caffe_path, video_deploy_path, video_model_path)
    for video_path in video_path_list:

        # video_path = "/data/sunnycia/SaliencyDataset/Video/MSU/videos/v01_Hugo_2172_left.avi"
        saliency_video_path = os.path.join(output_saliency_video_dir, os.path.basename(video_path))
        
        if os.path.isfile(saliency_video_path):
            print saliency_video_path, "exists, pass..."
            continue

        # fps = reader.get_meta_data()['fps']

        vs.setup_video(video_path)
        vs.create_saliency_video()
        fps = vs.video_meta_data['fps']
        vs.dump_predictions_as_video(saliency_video_path, fps)
        print "Done for video", video_path