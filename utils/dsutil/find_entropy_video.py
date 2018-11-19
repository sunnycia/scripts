import cPickle as pkl
import os, glob
import cv2
import matplotlib.pyplot as plt
from matplotlib import *
import numpy as np
import random
from scipy.spatial import distance

sys.path.insert(0, '../vizutil')
from grey_to_rgb import *

def index_sort(frame_path):
    frame_index = os.path.basename(frame_path).split('.')[0].split('_')[-1]
    return int(frame_index)

def get_dev_list(fixation_base):
    video_dir_list = glob.glob(os.path.join(fixation_base, '*'))
    video_dir_list.sort()
    dev_mean_list = []
    dev_std_list = []
    for video_dir in video_dir_list:
        print 'processing',video_dir
        fixation_list = glob.glob(os.path.join(video_dir, 'frame*'))
        fixation_list.sort(key=index_sort)
        # print fixation_list;exit()
        y_dev_list = []
        x_dev_list = []

        for fixation_path in fixation_list:
            fixation = cv2.imread(fixation_path, 0)
            # print np.std(fixation), np.mean(fixation);continue
            hei, wid = fixation.shape[0], fixation.shape[1]
            # print hei, wid
            ## find all fixate point
            y_arr, x_arr = np.where(fixation>0)

            #normalization
            if norm:
                y_arr = y_arr/float(hei)
                x_arr = x_arr/float(wid)

            ## calculate div
            # print y_arr[:10];exit()
            if len(y_arr)==0:
                y_dev=0
                x_dev=0
            else:
                y_dev = np.std(y_arr)
                x_dev = np.std(x_arr)
            y_dev_list.append(y_dev)
            x_dev_list.append(x_dev)

        # print np.mean(y_dev_list), np.std(y_dev_list);exit()
        dev_mean = (hei/float(hei+wid))*np.mean(y_dev_list) + (wid/float(hei+wid))*np.mean(x_dev_list)
        dev_std = (hei/float(hei+wid))*np.std(y_dev_list) + (wid/float(hei+wid))*np.std(x_dev_list) 
        dev_mean_list.append(dev_mean)
        dev_std_list.append(dev_std)

    # print dev_mean_list, dev_std_list
    pkl.dump(dev_mean_list,open("dev_mean_list.pkl","wb"))
    pkl.dump(dev_std_list,open("dev_std_list.pkl","wb"))

def blend_from_index(frame_base, density_base, index, frame_index = 30, color='r'):
    frame_dir_list = glob.glob(os.path.join(frame_base, '*'))
    density_dir_list = glob.glob(os.path.join(density_base, '*'))

    frame_dir = frame_dir_list[index]
    density_dir = density_dir_list[index]

    frame_path_list = glob.glob(os.path.join(frame_dir, 'frame*'))
    density_path_list = glob.glob(os.path.join(density_dir, 'frame*'))

    frame_path_list.sort(key=index_sort)
    density_path_list.sort(key=index_sort)

    frame_path = frame_path_list[frame_index]
    density_path = density_path_list[frame_index]

    # cv2.imshow('ff',cv2.imread(density_path))
    # cv2.waitKey(0)
    ##
    frame = cv2.imread(frame_path)
    density = grey_to_rgb(cv2.imread(density_path), color)
    blend = cv2.addWeighted(frame, 1, density, 1, 2.0)

    return blend

def plot_avg_dist_to_center(fixation_base, video_index, norm=False):
    fixation_dir_list = glob.glob(os.path.join(fixation_base, '*'))
    fixation_dir = fixation_dir_list[video_index]

    fixation_path_list = glob.glob(os.path.join(fixation_dir, 'frame*'))
    fixation_path_list.sort(key=index_sort)

    fixation_sample = cv2.imread(fixation_path_list[0],0)

    width, height = fixation_sample.shape

    frame_center = (width/2-1, height/2-1)
    if norm == True:
        frame_center=(0.5,0.5)
    print frame_center
    video_dist_list = []
    video_point_center_dist_list = []
    for fixation_path in fixation_path_list:
        fixation = cv2.imread(fixation_path, 0)
        y_arr, x_arr = np.where(fixation>0)
        if norm== True:
            y_arr = [y/float(height) for y in y_arr]
            x_arr = [x/float(width) for x in x_arr]

        point_center=(np.mean(x_arr), np.mean(y_arr))
        pos_list = zip(x_arr, y_arr)

        # print pos_list;exit()
        frame_dist_list = []
        point_center_dist_list = []
        for pos in pos_list:
            frame_dist_list.append(distance.euclidean(pos, frame_center))
            point_center_dist_list.append(distance.euclidean(pos, point_center))
        # print np.mean(frame_dist_list);exit()
        video_dist_list.append(np.mean(frame_dist_list))
        video_point_center_dist_list.append(np.mean(point_center_dist_list))

    # skip
    x = [i+1 for i in range(len(video_dist_list))]
    fig, ax = plt.subplots()
    plt.title(os.path.basename(fixation_dir))
    plt.plot(x[:len(x)-10], video_dist_list[:len(x)-10],label='average distance to frame center')
    plt.plot(x[:len(x)-10], video_point_center_dist_list[:len(x)-10],label='average distance to point clust center')
    plt.legend(shadow=True, fancybox=True)

    # plt.show()
    fig.set_size_inches(7,3)
    # plt.legend(ax, ('low complexity', 'middle complexity', 'high complexity'), loc='upper left', shadow=True,fancybox=True,fontsize='xx-large')


    plt.xlabel('frame index',fontsize=10)
    plt.ylabel('normalized distance',fontsize=10)
    if norm == True:
        plt.ylim(0.0, 1.0)
    else:
        plt.ylim(0.0,1000.0)
    plt.savefig('avg_dist_'+str(video_index)+'.png',bbox_inches='tight')

    plt.close()
    # dist = numpy.linalg.norm(vec1 - vec2)    



def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
def normalize(x):
    x = x-np.min(x)
    x = x/np.max(x)
    return x

if __name__ =='__main__':
    norm = True
    dataset = 'videoset'
    if dataset =='videoset':
        frame_base = '/data/SaliencyDataset/Video/VideoSet/ImageSet/Seperate/frame'
        density_base = '/data/SaliencyDataset/Video/VideoSet/ImageSet/Seperate/density/sigma32'
        fixation_base = '/data/SaliencyDataset/Video/VideoSet/ImageSet/Seperate/fixation'
    # fixation_base = '/data/SaliencyDataset/Video/VideoSet/ImageSet/Seperate/fixation'
    # get_dev_list(fixation_base)
    # split_percentage=np.array([0.25, 0.65, 0.1])

    low_entropy_percentage=0.25
    mid_entropy_percentage=0.65
    high_entropy_percentage=0.1

    dev_mean_list = np.array(pkl.load(open("vs_dev_mean_list.pkl", 'rb')))
    dev_std_list = np.array(pkl.load(open("vs_dev_std_list.pkl", 'rb')))

    samples_list = len(dev_mean_list)
    print dev_mean_list
    # plot dev_mean_list
    x = os.listdir(fixation_base)
    x.sort()
    x = np.array(x)
    fig, ax = plt.subplots()
    perm = dev_mean_list.argsort()
    print perm[:10]
    dev_mean_list = dev_mean_list[perm]
    x = x[perm]
    print x[:10]
    for i in range(len(x)):
        x[i] = x[i].replace('videoSRC', '')
    x_copy = np.copy(x)
    index = np.array([i for i in range(len(dev_mean_list))])

    # plot low
    low_num = int(low_entropy_percentage * len(dev_mean_list))
    mid_num = int(mid_entropy_percentage * len(dev_mean_list))
    high_num = int(high_entropy_percentage * len(dev_mean_list))
    # low_num += len(dev_mean_list) -(low_num+mid_num+high_num)



    sample = 10
    # generate low complexity sample
    low_index_list = perm[:low_num]
    print low_index_list[:20]
    for i in range(sample):
        video_index = random.choice(low_index_list)

        plot_avg_dist_to_center(fixation_base, video_index)

        # blend = blend_from_index(frame_base, density_base, video_index, color='r')
        # cv2.imwrite('mid_'+str(i)+'.jpg', blend)
    # low_index_list = perm[low_num:low_num+mid_num]
    # for i in range(sample)
    #     video_index = random.choice(low_index_list)
    #     blend = blend_from_index(frame_base, density_base, video_index)
    #     cv2.imwrite('low_'+str(i)+'.jpg', blend)
    #     low_index_list = perm[:low_num]
    # for i in range(sample)
    #     video_index = random.choice(low_index_list)
    #     blend = blend_from_index(frame_base, density_base, video_index)
    #     cv2.imwrite('low_'+str(i)+'.jpg', blend)

    lc = plt.bar(index[:low_num], dev_mean_list[:low_num],color='g', label='low complexity')
    mc = plt.bar(index[low_num:low_num+mid_num], dev_mean_list[low_num:low_num+mid_num],color='y',label='middle complexity')
    hc = plt.bar(index[low_num+mid_num:], dev_mean_list[low_num+mid_num:],color='r', label='high complexity')

    # ax.legend()
    plt.xticks(index, x, rotation='vertical', size=9)

    plt.legend((lc, mc, hc), ('low complexity', 'middle complexity', 'high complexity'), loc='upper left', shadow=True,fancybox=True,fontsize='xx-large')
    plt.xlabel('video index',fontsize=18)
    plt.ylabel('complexity score', fontsize=18)
    # plt.xticks(index[low_num:low_num+mid_num], x[low_num:low_num+mid_num], rotation='vertical', size=8)
    
    # plt.bar(x, dev_mean_list, yerr=dev_std_list)
    fig.set_size_inches(15.5*20, 5.5*1)

    plt.savefig('dev_.png',bbox_inches='tight')