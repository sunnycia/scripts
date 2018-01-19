import time
import glob, os

saliency_base = '/data/sunnycia/SaliencyDataset/Video/VideoSet/Results/saliency_map'
# saliency_base = '/data/sunnycia/SaliencyDataset/Video/VideoSet/Results/saliency_map'
output_path = 'fps.txt'
w_f = open(output_path, 'a+')

print >> w_f, saliency_base

sub_dir_list = os.listdir(saliency_base)

for sub_dir in sub_dir_list:
    cur_dir = os.path.join(saliency_base, sub_dir)
    saliency_map_list = glob.glob(os.path.join(cur_dir, '*', '*.*'))

    saliency_map_list.sort(key=lambda x: os.path.getmtime(x))
    duration_in_secends = os.path.getmtime(saliency_map_list[-1]) - os.path.getmtime(saliency_map_list[0])
    print duration_in_secends
    
    fps = len(saliency_map_list) / duration_in_secends

    print >> w_f, sub_dir
    print >> w_f, 'duration_in_seconds:', duration_in_secends
    print >> w_f, 'total saliencymap:', len(saliency_map_list)
    print >> w_f, 'fps:', fps
    print >> w_f, ' '