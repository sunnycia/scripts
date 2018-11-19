from utils.pymetric.metrics import *
import imghdr, imageio
import scipy.io as sio
from math import floor
import glob, cv2, os, numpy as np, sys, caffe
from utils.common import tic, toc
from Saliencynet import FlowbasedVideoSaliencyNet, FramestackbasedVideoSaliencyNet, C3DbasedVideoSaliencyNet,VoxelbasedVideoSaliencyNet
from Dataset import VideoDataset
import argparse
caffe.set_mode_gpu()
caffe.set_device(0)

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_type', type=str, default='image', help='Output saliency (video) or (image)')
    parser.add_argument('--allinone', type=bool, default=False)
    # parser.add_argument('--video_base', type=str, required=True)
    parser.add_argument('--test_base', type=str, default='videoset')
    parser.add_argument('--modelname', type=str, default=None)
    parser.add_argument('--protocode', type=int, default=1)

    parser.add_argument('--video_deploy_path',type=str,default='./prototxt/vo-v4-2-resnet.prototxt')
    parser.add_argument('--video_model_dir',type=str,default='../training_output/salicon/')
    parser.add_argument('--infertype', type=str,default='slice')
    parser.add_argument('--inferoverlap', type=int,default=15)
    parser.add_argument('--threshold', type=float, default=0)

    return parser.parse_args()
print "Parsing arguments..."
args = get_arguments()
output_type = args.output_type
threshold = args.threshold

ss_blist = [100, 73, 14, 116, 157, 101, 105, 125, 79]
ss_wlist = [41, 179, 54, 47, 123, 36, 40, 98, 129, 30, 29, 204, 99, 53, 80]
ss_list  = ss_blist + ss_wlist

# ss_list = [100]
ss_list.sort()
print ss_list, len(ss_list);#exit()

prototxt_list = [
'prototxt/vo-v4-2.prototxt','prototxt/vo-v4-2-resnet.prototxt','prototxt/vo-v4-2-resnet-catfeat-bnorm.prototxt', 'prototxt/vo-v4-2-resnet-dropout.prototxt'
]

model_list = []
model_name = args.modelname
proto_code = args.protocode

if model_name is None:
    model_list = [
    # (1, 'vo-v4-2-resnet-snapshot-2000-display-1--batch-2_1514034705/snapshot-_iter_72000.caffemodel'),
    # (1, 'vo-v4-2-resnet-base_lr-0.01-snapshot-2000-display-1--batch-2_1514260519_usesnapshot_1514034705_snapshot-_iter_72000/snapshot-_iter_96000.caffemodel'),
    # (1, 'vo-v4-2-resnet-base_lr-0.01-snapshot-2000-display-1--batch-2_1514260519_usesnapshot_1514034705_snapshot-_iter_72000/snapshot-_iter_150000.caffemodel'),
    # (1, 'vo-v4-2-resnet-snapshot-2000-display-1-fulldens-batch-2_1514129205/snapshot-_iter_170000.caffemodel'),
    # (2, 'vo-v4-2-resnet-catfeat-snapshot-2000-display-1-fulldens-batch-2_1514129183/snapshot-_iter_168000.caffemodel'),

    # (1, 'vo-v4-2-resnet-base_lr-0.01-snapshot-2000-display-1--batch-2_1514260519_usesnapshot_1514034705_snapshot-_iter_72000/snapshot-_iter_436000.caffemodel'),
    # (1, 'vo-v4-2-resnet-snapshot-2000-display-1-fulldens-batch-2_1514129205/snapshot-_iter_454000.caffemodel'),
    # (2, 'vo-v4-2-resnet-catfeat-snapshot-2000-display-1-fulldens-batch-2_1514129183/snapshot-_iter_450000.caffemodel'),
    # (1, 'vo-v4-2-resnet-kldloss-snapshot-2000-display-1--batch-2_1514116065/snapshot-_iter_26000.caffemodel'),
    # (1, 'vo-v4-2-resnet-kldloss-snapshot-2000-display-1--batch-2_1514116065/snapshot-_iter_50000.caffemodel'),
    # (1, 'vo-v4-2-resnet-dropout-snapshot-2000-display-1-dropout_fulldens-batch-2_1514857787/snapshot-_iter_20000.caffemodel'), 
    # (1, 'vo-v4-2-resnet-dropout-snapshot-2000-display-1-dropout_fulldens-batch-2_1514857787/snapshot-_iter_26000.caffemodel')
    # (1, 'vo-v4-2-resnet-dropout-snapshot-2000-display-1-dropout_fulldens-batch-2_1514857787/snapshot-_iter_30000.caffemodel'), 
    # (1, 'vo-v4-2-resnet-dropout-snapshot-2000-display-1-dropout_fulldens-batch-2_1514857787/snapshot-_iter_40000.caffemodel'), 
    # (1, 'vo-v4-2-resnet-dropout-snapshot-2000-display-1-dropout_fulldens-batch-2_1514857787/snapshot-_iter_50000.caffemodel'), 
    # (1, 'vo-v4-2-resnet-dropout-snapshot-2000-display-1-dropout_fulldens-batch-2_1514857787/snapshot-_iter_60000.caffemodel'), 
    # (1, 'vo-v4-2-resnet-dropout-snapshot-2000-display-1-dropout_fulldens-batch-2_1514857787/snapshot-_iter_22000.caffemodel'), 
    # (1, 'vo-v4-2-resnet-dropout-snapshot-2000-display-1-dropout_fulldens-batch-2_1514857787/snapshot-_iter_24000.caffemodel'), 
    # (1, 'vo-v4-2-resnet-dropout-snapshot-2000-display-1-dropout_fulldens-batch-2_1514857787/snapshot-_iter_28000.caffemodel'), 
    # (1, 'vo-v4-2-resnet-dropout-snapshot-2000-display-1-dropout_fulldens-batch-2_1514857787/snapshot-_iter_70000.caffemodel'), 
    # (1, 'vo-v4-2-resnet-dropout-snapshot-2000-display-1-dropout_fulldens-batch-2_1514857787/snapshot-_iter_80000.caffemodel'), 
    # (1, 'vo-v4-2-resnet-dropout-snapshot-2000-display-1-dropout_fulldens-batch-2_1514857787/snapshot-_iter_90000.caffemodel'), 
    # (1, 'vo-v4-2-resnet-dropout-snapshot-2000-display-1-dropout_fulldens-batch-2_1514857787/snapshot-_iter_100000.caffemodel')
    # (1, 'vo-v4-2-resnet-dropout-snapshot-2000-display-1-dropout_fulldens-batch-2_1514857787/snapshot-_iter_110000.caffemodel'), 
    # (1, 'vo-v4-2-resnet-dropout-snapshot-2000-display-1-dropout_fulldens-batch-2_1514857787/snapshot-_iter_120000.caffemodel'), 
    # (1, 'vo-v4-2-resnet-dropout01-snapshot-2000-display-1-01dropout_fulldens-batch-2_1514964788/snapshot-_iter_20000.caffemodel'), 
    # (1, 'vo-v4-2-resnet-dropout01-snapshot-2000-display-1-01dropout_fulldens-batch-2_1514964788/snapshot-_iter_26000.caffemodel'), 
    # (1, 'vo-v4-2-resnet-dropout01-snapshot-2000-display-1-01dropout_fulldens-batch-2_1514964788/snapshot-_iter_30000.caffemodel'),
    # (1, 'vo-v4-2-resnet-dropout01-snapshot-2000-display-1-01dropout_fulldens-batch-2_1514964788/snapshot-_iter_40000.caffemodel')
    # (1, 'vo-v4-2-resnet-dropout01-snapshot-2000-display-1-01dropout_fulldens-batch-2_1514964788/snapshot-_iter_50000.caffemodel')
    # (1, 'vo-v4-2-resnet-dropout01-snapshot-2000-display-1-01dropout_fulldens-batch-2_1514964788/snapshot-_iter_60000.caffemodel')
    # (1, 'vo-v4-2-resnet-dropout01-snapshot-2000-display-1-01dropout_fulldens-batch-2_1514964788/snapshot-_iter_70000.caffemodel'),
    # (1, 'vo-v4-2-resnet-dropout01-snapshot-2000-display-1-01dropout_fulldens-batch-2_1514964788/snapshot-_iter_80000.caffemodel'),
    # (1, 'vo-v4-2-resnet-dropout01-snapshot-2000-display-1-01dropout_fulldens-batch-2_1514964788/snapshot-_iter_90000.caffemodel'),
    # (1, 'vo-v4-2-resnet-dropout01-snapshot-2000-display-1-01dropout_fulldens-batch-2_1514964788/snapshot-_iter_100000.caffemodel'),
    # (1, 'vo-v4-2-resnet-dropout01-snapshot-2000-display-1-01dropout_fulldens-batch-2_1514964788/snapshot-_iter_110000.caffemodel')
    # (1, 'vo-v4-2-resnet-dropout01-snapshot-2000-display-1-01dropout_fulldens-batch-2_1514964788/snapshot-_iter_120000.caffemodel'),
    # (1, 'vo-v4-2-resnet-dropout01-snapshot-2000-display-1-01dropout_fulldens-batch-2_1514964788/snapshot-_iter_130000.caffemodel'),
    # (1, 'vo-v4-2-resnet-dropout01-snapshot-2000-display-1-01dropout_fulldens-batch-2_1514964788/snapshot-_iter_140000.caffemodel'),
    # (1, 'vo-v4-2-resnet-dropout01-snapshot-2000-display-1-01dropout_fulldens-batch-2_1514964788/snapshot-_iter_150000.caffemodel')
    (3, 'vo-v4-2-resnet-l1loss-dropout-snapshot-4000-data_aug-batch-2_1526869047/snapshot-_iter_100000.caffemodel'),
    (3, 'vo-v4-2-resnet-l1loss-dropout-snapshot-4000-data_aug-batch-2_1526869047/snapshot-_iter_300000.caffemodel'),
    (3, 'vo-v4-2-resnet-l1loss-dropout-snapshot-4000-data_aug-batch-2_1526869047/snapshot-_iter_500000.caffemodel'),
    (3, 'vo-v4-2-resnet-l1loss-dropout-snapshot-4000-data_aug-batch-2_1526869047/snapshot-_iter_700000.caffemodel')
]
else:
    model_list.append((proto_code, model_name))

# video_deploy_path = args.video_deploy_path

for prototxt_index, model in model_list:
    video_deploy_path = prototxt_list[prototxt_index]
    video_model_path = os.path.join(args.video_model_dir, model)

    video_base='';saliency_map_base=''
    if args.test_base == 'videoset':
        video_base = '/data/SaliencyDataset/Video/VideoSet/Videos/videos_origin'
        saliency_map_base = '/data/SaliencyDataset/Video/VideoSet/Results/ss_saliency_map'

    model_name = os.path.dirname(video_model_path).split('/')[-1] + '_'+ os.path.basename(video_model_path).split('.')[0] + '_threshold'+str(threshold)
    video_path_list = glob.glob(os.path.join(video_base, "*.*"))
    video_path_list.sort()
    video_path_list = np.array(video_path_list)[ss_list]
    # print video_path_list;exit()
    saliency_map_dir = os.path.join(saliency_map_base, model_name)
    if not os.path.isdir(saliency_map_dir):
        os.makedirs(saliency_map_dir)

    vs = VoxelbasedVideoSaliencyNet(deploy_proto=video_deploy_path, caffe_model=video_model_path, video_length=16, video_size=(112,112),mean_list=[90, 98,102], infer_type=args.infertype)

    for video_path in video_path_list:
        video_name = os.path.basename(video_path).split('.')[0].split('_')[0]
        if len(glob.glob(os.path.join(saliency_map_dir, video_name+'*'))) != 0:
            print video_name, "already done, pass."
            continue
        else:
            print len(glob.glob(os.path.join(saliency_map_dir, video_name+'*')))
            print "Handling",video_name

        vs.setup_video(video_path)
        vs.create_saliency_video(threshold=threshold, overlap=args.inferoverlap)

        vs.dump_predictions_as_images(saliency_map_dir, video_name, args.allinone)
        print "Done for video", video_path;#exit()

    ## Evaluation
    saliency_directory = os.path.join(saliency_map_base, model_name)
    # density_directory = '/data/SaliencyDataset/Video/VideoSet/ImageSet/Seperate/density_fc/density-6'
    density_directory = '/data/SaliencyDataset/Video/VideoSet/ImageSet/Seperate/density/sigma32'
    fixation_directory = '/data/SaliencyDataset/Video/VideoSet/ImageSet/Seperate/fixation'

    sub_video_list = os.listdir(saliency_directory)

    cur_cc = []
    cur_sim = []
    cur_auc = []

    for sub_video in sub_video_list:
        cur_sal_dir = os.path.join(saliency_directory, sub_video)
        cur_dens_dir = os.path.join(density_directory, sub_video)
        cur_fix_dir = os.path.join(fixation_directory, sub_video)

        sal_path_list = glob.glob(os.path.join(cur_sal_dir, '*.*'))
        dens_path_list = glob.glob(os.path.join(cur_dens_dir, '*.*'))
        fix_path_list = glob.glob(os.path.join(cur_fix_dir, '*.*'))
        sal_path_list.sort();dens_path_list.sort();fix_path_list.sort()
        # print sal_path_list[:10], dens_path_list[:10], fix_path_list[:10];#exit()
        for sal_path, dens_path, fix_path in zip(sal_path_list, dens_path_list, fix_path_list):
            print "Processing",sal_path
            sal_map = cv2.imread(sal_path, 0).astype(np.float32)
            dens_map = cv2.imread(dens_path, 0).astype(np.float32)
            # fix_map = sio.loadmat(fix_path)['fixation']
            fix_map= cv2.imread(fix_path,0)

            cur_cc.append(CC(sal_map, dens_map))
            cur_sim.append(SIM(sal_map, dens_map))
            # cur_auc.append(AUC_Judd(sal_map, dens_map))
    cur_cc = np.array(cur_cc)
    cur_sim = np.array(cur_sim)

    cc = np.mean(cur_cc[~np.isnan(cur_cc)])
    sim = np.mean(cur_sim[~np.isnan(cur_sim)])
    # auc = np.mean(cur_auc[~np.isnam(cur_auc)])
    w_f = open(os.path.join(saliency_map_base, 'result.txt'), 'a+')
    print >> w_f, model_name
    print >> w_f, 'CC\tSIM\tAUC'
    # print >> w_f, cc, sim, auc
    print >> w_f, cc, sim
    print >> w_f, ''

    w_f.close()
    # saliency_score_cc = np.zeros((length))
    # saliency_score_sim = np.zeros((length))
    # saliency_score_jud = np.zeros((length))

    # saliency_map_list = glob.glob(os.path.join(sal_base, model, "*", "*.*"))
    # density_map_list = glob.glob(os.path.join(dens_dir, "*",  "*.*"))
    # fixation_map_list = glob.glob(os.path.join(fixa_dir,  "*", "*.*"))
    # saliency_map_list.sort(key=path_based_sort)
    # density_map_list.sort(key=path_based_sort)
    # fixation_map_list.sort(key=path_based_sort)