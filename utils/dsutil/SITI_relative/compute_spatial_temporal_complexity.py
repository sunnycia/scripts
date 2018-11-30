import cv2 
import numpy as np
import os, glob
from SI_TI import SI, TI
import argparse
import matplotlib.pyplot as plt
import cPickle as pkl
parser = argparse.ArgumentParser()
parser.add_argument('--ds_name', type=str, required=True)
# parser.add_argument('--ds_name', type=str, required=True)

args = parser.parse_args()

ds_name = args.ds_name
if ds_name == 'videoset':
    base_dir = '/data/SaliencyDataset/Video/VideoSet/ImageSet/Seperate/frames'
if ds_name == 'savam':
    base_dir = '/data/SaliencyDataset/Video/MSU/frames'
if ds_name == 'diem':
    base_dir = '/data/SaliencyDataset/Video/DIEM/frames'
if ds_name == 'ledov':
    base_dir = '/data/SaliencyDataset/Video/LEDOV/frames'
if ds_name == 'hollywood':
    base_dir = '/data/SaliencyDataset/Video/ActionInTheEye/Hollywood2/frames'
if ds_name == 'gazecom':
    base_dir = '/data/SaliencyDataset/Video/GAZECOM/frames'
if ds_name == 'dhf1k':
    base_dir = '/data/SaliencyDataset/Video/DHF1K/frames'
# ds_list = [base_dir]
# for base_dir in ds_list:

video_list = os.listdir(base_dir)
SI_list = []
TI_list = []

for video_name in video_list:
    print 'processing', video_name
    frame_path_list = glob.glob(os.path.join(base_dir, video_name, '*.*'))
    tmp_si_list=[]
    tmp_ti_list=[]
    for i in range(1, len(frame_path_list)):
        prev_frame=cv2.imread(frame_path_list[i-1],0)
        cur_frame=cv2.imread(frame_path_list[i],0)
        si = SI(cur_frame)
        ti = TI(prev_frame, cur_frame)
        tmp_si_list.append(si)
        tmp_ti_list.append(ti)

    SI_list.append(np.mean(tmp_si_list))
    TI_list.append(np.mean(tmp_ti_list))
# print SI_list
# print TI_list

fig, ax = plt.subplots()
ax.scatter(np.array(SI_list), np.array(TI_list), marker='*')
ax.set_xlabel('SI', fontsize=10)
ax.set_ylabel('TI', fontsize=10)
# ax.grid(True)
fig.tight_layout()

pkl_dict={'SI': SI_list, 'TI': TI_list}
pkl.dump(pkl_dict, open('%s_siti_list.pkl'%ds_name, 'wb'))
plt.savefig('%s_siti.png'%ds_name, dpi=300)
# plt.show()
# show()