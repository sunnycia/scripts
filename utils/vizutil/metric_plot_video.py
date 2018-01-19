#coding=utf-8
import os, glob
import scipy.io as sio
import cv2
import numpy as np
import argparse
from utils.common import mean_without_nan
import matplotlib.pyplot as plt
import time, base64
# ┌─┐┬─┐┌─┐┌─┐┌┬┐┌─┐  ┌─┐┬  ┌─┐┌┬┐  ┬  ┬┬┌┬┐┌─┐┌─┐
# │  ├┬┘├┤ ├─┤ │ ├┤   ├─┘│  │ │ │   └┐┌┘│ ││├┤ │ │
# └─┘┴└─└─┘┴ ┴ ┴ └─┘  ┴  ┴─┘└─┘ ┴    └┘ ┴─┴┘└─┘└─┘
metric_map={
'cc':0, 
'sim':1,
'jud':2,
'bor':3,
'sauc':4,
'kld':6,
'nss':7}
plot_color_list = ['deepskyblue', 'blue', 'olive', 'gold', 'darkolivegreen', 'darkgreen', 'slategray', 'royalblue', 'brown', 'chocolate']
def metric_plot_video(metric_dir, output_dir, temp_file_dir, metric_name, index_list, mat_wildcard='videoSRC%s*.mat', video_length=5, resolution=(1920,1080), plot_thickness=5):
    print "processing metric", metric_name
    output_dir = os.path.join(output_dir, metric_name)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    for index in index_list:
        ## load metric
        mat_name = mat_wildcard % str(index).zfill(3)
        metric_path = glob.glob(os.path.join(metric_dir, mat_name))[0]
        metric_list = sio.loadmat(metric_path)['saliency_score']

        metric = metric_list[metric_map[metric_name]]
        nframe = len(metric)
        fps = nframe/video_length

        output_name = 'videoSRC%s.avi' % str(index).zfill(3)
        output_path =  os.path.join(output_dir, output_name)
        if os.path.isfile(output_path):
            print output_path, 'already exists, pass'
            continue
        print "Video will be save to", output_path
        codec = cv2.VideoWriter_fourcc('D','I','V','X')
        video_writer = cv2.VideoWriter(output_path, codec, fps, resolution)
        x = np.arange(0, nframe, 1)
        y = [None for i in range(nframe)]
        mean_list = [None for i in range(nframe)]
        mean = mean_without_nan(metric)
        plt.plot(x,y)
        plt.plot(x, mean_list, ':')
        tmpfig_path = os.path.join(temp_file_dir, 'plot_fig_'+base64.b64encode(str(time.time()))+'.png')
        for i in range(nframe):
            print "\tprocessing frame",i, '\r', 
            y[i] = metric[i]
            mean_list[i] = mean

            plt.plot(x,y, 'blue', linewidth=plot_thickness)
            plt.plot(x,mean_list,'r:')
            plt.xlabel('Frame Index')
            plt.ylabel(output_name + ' ' + metric_name)
            fig = plt.gcf()
            fig.set_size_inches(8, 5)
            plt.savefig(tmpfig_path, dpi=150)
            figure = cv2.imread(tmpfig_path)
            plot_image = cv2.resize(figure, dsize=resolution)
            video_writer.write(plot_image)
            plt.clf()
        video_writer.release()

        os.remove(tmpfig_path)
        print '\tDone for', output_name

def all_metric_plot_video(metric_dir, output_dir, temp_file_dir, metric_name, index_list, mat_wildcard='videoSRC%s*.mat', video_length=5, resolution=(1920,1080), plot_thickness=5):
    print "processing metric", metric_name
    output_dir = os.path.join(output_dir, metric_name)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    for index in index_list:
        ## load metric
        mat_name = mat_wildcard % str(index).zfill(3)
        metric_path = glob.glob(os.path.join(metric_dir, mat_name))[0]
        metric_list = sio.loadmat(metric_path)['saliency_score']

        metric = metric_list[metric_map['cc']]
        nframe = len(metric)
        fps = nframe/video_length

        output_name = 'videoSRC%s.avi' % str(index).zfill(3)
        output_path =  os.path.join(output_dir, output_name)
        codec = cv2.VideoWriter_fourcc('D','I','V','X')
        video_writer = cv2.VideoWriter(output_path, codec, fps, resolution)
        print "Video will be save to", output_path
        x = np.arange(0, nframe, 1)
        y = [None for i in range(nframe)]
        mean_list = [None for i in range(nframe)]
        plt.plot(x,y)
        hash_string = base64.b64encode(str(time.time()))
        tmpfig_path = os.path.join(temp_file_dir, 'plot_fig'+hash_string+'.png')
        for i in range(nframe):
            print "\tprocessing frame",i, '\r', 
            for metric_name in metric_map:
                if metric_name == "kld" or metric_name=='nss':
                    continue
                metric = metric_list[metric_map[metric_name]]
                y[i] = metric[i]
                plt.plot(x,y, plot_color_list[metric_map[metric_name]], linewidth=plot_thickness)
            plt.xlabel('Frame Index')
            plt.ylabel(output_name + ' ' + metric_name)
            fig = plt.gcf()
            fig.set_size_inches(8, 5)
            plt.savefig(tmpfig_path, dpi=150)
            figure = cv2.imread(tmpfig_path)
            plot_image = cv2.resize(figure, dsize=resolution)
            video_writer.write(plot_image)
            plt.clf()
        video_writer.release()

        os.remove(tmpfig_path)
        print '\tDone for', output_name

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--metricdir', type=str, required=True, help="Directory of video model metric")
    parser.add_argument('--outputbase', type=str, required=True, help="Output base directory")
    args = parser.parse_args()

    metric_dir = args.metricdir
    output_dir = os.path.join(args.outputbase, os.path.basename(metric_dir))

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    video_metric_list = glob.glob(os.path.join(metric_dir, "*.mat"))
    video_metric_list.sort()
    cc=[];    sim=[];    auc_jud=[];    auc_bor=[];    shuf_auc=[];    kld=[];    nss=[]
    for video_metric in video_metric_list:
        video_metric = sio.loadmat(video_metric)
        score = video_metric["saliency_score"];

        cc.append(mean_without_nan(score[0]))
        sim.append(mean_without_nan(score[1]))
        auc_jud.append(mean_without_nan(score[2]))
        auc_bor.append(mean_without_nan(score[3]))
        shuf_auc.append(mean_without_nan(score[4]))
        kld.append(mean_without_nan(score[6]))
        nss.append(mean_without_nan(score[7]))

    index_list = [i+1 for i in range(220)]
    for metric_name in metric_map:
        metric_plot_video(metric_dir, output_dir, '/tmp', metric_name, index_list)
    
    # all_metric_plot_video(metric_dir, output_dir, '/tmp', 'metric[cc_si_aj_ab_sa]', index_list)