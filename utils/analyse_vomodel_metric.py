#coding=utf-8
import numpy as np
import os, glob
import scipy.io as sio
import argparse
import matplotlib.pyplot as plt
import math
import shutil
import tempfile
import cv2
from utils.common import mean_without_nan

parser = argparse.ArgumentParser()
parser.add_argument('--metricdir', type=str, required=True, help="Directory of video model metric")
# parser.add_argument('--plotdir', type=str, required=True, help="output directory of plot")
parser.add_argument('--outputbase', type=str, required=True, help="output base directory")

args = parser.parse_args()

metric_dir = args.metricdir
output_base = os.path.join(args.outputbase, os.path.basename(metric_dir))
# plot_dir = args.plotdir
plot_dir = os.path.join(output_base, 'plots')
filter_video_dir = os.path.join(output_base, 'videos')
if not os.path.isdir(plot_dir):
    os.makedirs(plot_dir)

def plot_result(metric_arr, output_dir, metric_name, threshold_coef=0.25, output_level='high'):
    print "ploting", metric_name
    # print metric_arr, metric_name
    length = len(metric_arr)
    x = [i for i in range(length)]    
    x = [i for i in range(length)]
    mean = np.mean(metric_arr)
    median = np.median(metric_arr)
    threshold=threshold_coef * mean

    _index = [i for i in range(length) if metric_arr[i] < mean and metric_arr[i] < median]
    # print _index, len(_index)

    plt.plot(x, metric_arr)
    plt.plot(x, [mean for i in range(length)], '--')
    plt.plot(x, [median for i in range(length)], ':')
    # plt.plot(x, [threshold for i in range(length)], ':')

    high=[]
    medium=[]
    low=[]
    for i in range(len(_index)):
        index = _index[i]
        dist = abs(mean-metric_arr[index])
        if dist > threshold*1.5:
            plt.axvline(x=index, ls=':', color='r')
            high.append(index)
        elif dist > threshold:
            plt.axvline(x=index, ls=':', color='yellow')
            medium.append(index)
        else:
            plt.axvline(x=index, ls=':', color='lightgray')
            medium.append(index)

    label_arr = ["%03d"%(i+1) for i in range(length)]
    plt.xticks(x, label_arr, rotation=90)
    ax=plt.gca()
    ax.set_xlabel('video index')
    ax.set_ylabel(metric_name+' metric')

    fig = plt.gcf()
    fig.set_size_inches(30,5)
    output_path = os.path.join(plot_dir, metric_name+".png")
    fig.savefig(output_path, dpi=500)
    # plt.clf()
    # plt.cla()
    plt.close('all')

    if output_level=='high':
        return high
    if output_level=='medimu':
        return medium
    if output_level=='low':
        return low

def filter_video(video_dir, output_dir, metric_name, index_list, video_name_wildcard='videoSRC%s.avi'):
    output_dir = os.path.join(output_dir, metric_name)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    for index in index_list:
        video_name = video_name_wildcard % str(index).zfill(3)
        video_path = os.path.join(video_dir, video_name)

        output_path = os.path.join(output_dir, video_name)
        shutil.copy(video_path, output_path)




#         _         _                      _          _       
#  _ __  | |  ___  | |_   _ __ ___    ___ | |_  _ __ (_)  ___ 
# | '_ \ | | / _ \ | __| | '_ ` _ \  / _ \| __|| '__|| | / __|
# | |_) || || (_) || |_  | | | | | ||  __/| |_ | |   | || (__ 
# | .__/ |_| \___/  \__| |_| |_| |_| \___| \__||_|   |_| \___|
# |_|
video_metric_list = glob.glob(os.path.join(metric_dir, "*.mat"))
video_metric_list.sort()
print video_metric_list;exit()
cc=[]
sim=[]
auc_jud=[]
auc_bor=[]
shuf_auc=[]
kld=[]
nss=[]


for video_metric in video_metric_list:
    video_metric = sio.loadmat(video_metric)
    score = video_metric["saliency_score"];
    # score = np.transpose(score, (1,0))

    # print score[0], np.mean(score[0]), len(score)heaven
    # score = score[~np.isnan(score)] ## exclude nan value
    # score[~np.isnan(score)] = []
    # print score.shape
    # print score[0], np.mean(score[3]), mean_without_nan(score[3])
    cc.append(mean_without_nan(score[0]))
    sim.append(mean_without_nan(score[1]))
    auc_jud.append(mean_without_nan(score[2]))
    auc_bor.append(mean_without_nan(score[3]))
    shuf_auc.append(mean_without_nan(score[4]))
    kld.append(mean_without_nan(score[6]))
    nss.append(mean_without_nan(score[7]))
    # print cc, sim, auc_jud, auc_bor, shuf_auc, kld, nss;exit()

    # print score.shape;exit()
# cc = cc[~np.isnan(cc)]


## plot images
cc_index = plot_result(cc, plot_dir, "cc")
sim_index = plot_result(sim, plot_dir, "sim")
jud_index = plot_result(auc_jud, plot_dir, "auc_jud")
bor_index = plot_result(auc_bor, plot_dir, "auc_bor")
sauc_index = plot_result(shuf_auc, plot_dir, "shuf_auc")
kld_index = plot_result(kld, plot_dir, "kld")
nss_index = plot_result(nss, plot_dir, "nss")

metric_index_dict = {
    'cc':cc_index, 
    'sim':sim_index,
    'jud':jud_index,
    'bor':bor_index,
    'sauc':sauc_index,
    'kld':kld_index,
    'nss':nss_index
}

metric_map = {
    'cc':0, 
    'sim':1,
    'jud':2,
    'bor':3,
    'sauc':4,
    'kld':6,
    'nss':7
}


#             _              _            _      _                   
#  ___   ___ | |  ___   ___ | |_  __   __(_)  __| |  ___   ___   ___ 
# / __| / _ \| | / _ \ / __|| __| \ \ / /| | / _` | / _ \ / _ \ / __|
# \__ \|  __/| ||  __/| (__ | |_   \ V / | || (_| ||  __/| (_) |\__ \
# |___/ \___||_| \___| \___| \__|   \_/  |_| \__,_| \___| \___/ |___/
                                                                   
# baseline model configuration
origin_video_dir = '/data/sunnycia/SaliencyDataset/Video/VideoSet/Videos/videos_origin'
density_video_dir = '/data/sunnycia/SaliencyDataset/Video/VideoSet/Videos/density_videos'
blend_video_dir = '/data/sunnycia/SaliencyDataset/Video/VideoSet/Videos/videos_blend_baseline'
saliency_video_dir = '/data/sunnycia/SaliencyDataset/Video/VideoSet/Results/saliency_video/image_model_result/train_kldloss-kld_weight-100-batch-1_1510102029_usesnapshot_1509584263_snapshot-_iter_100000'
metric_video_dir = ''

# output_base = '../analyse_vomodel'

for metric_name in metric_index_dict:
    index = metric_index_dict[metric_name]
    filter_video(origin_video_dir, os.path.join(filter_video_dir, 'origin_video'), metric_name, index)
    filter_video(density_video_dir, os.path.join(filter_video_dir, 'density_video'), metric_name, index)
    filter_video(blend_video_dir, os.path.join(filter_video_dir, 'blend_video'), metric_name, index)
    filter_video(saliency_video_dir, os.path.join(filter_video_dir, 'saliency_video'), metric_name, index)


 #          _                     _  _              _    _               
 # /\   /\ (_) ___  _   _   __ _ | |(_) ____  __ _ | |_ (_)  ___   _ __  
 # \ \ / / | |/ __|| | | | / _` || || ||_  / / _` || __|| | / _ \ | '_ \ 
 #  \ V /  | |\__ \| |_| || (_| || || | / / | (_| || |_ | || (_) || | | |
 #   \_/   |_||___/ \__,_| \__,_||_||_|/___| \__,_| \__||_| \___/ |_| |_|

viz_dir = os.path.join(output_base, 'visualization')
