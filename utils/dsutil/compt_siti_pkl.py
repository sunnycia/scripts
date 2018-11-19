import cv2 
import numpy as np
import os, glob
from SI_TI import SI, TI
import argparse
import matplotlib.pyplot as plt
# plt.tight_layout()
import cPickle as pkl

# pkl_list = glob.glob(os.path.join('siti_pkl', '*.pkl'))


# for pkl_file in pkl_list:
# for i in range(len(pkl_list)):
#     pkl_file = pkl_list[i]
#     ds_name = os.path.basename(pkl_file).split('_')[0]
#     pkl_dict = pkl.load(open(pkl_file, 'rb'))
#     SI_list = pkl_dict['SI']
#     TI_list = pkl_dict['TI']

#     fig, ax = plt.subplots(i)
#     ax.scatter(np.array(SI_list), np.array(TI_list), marker='*', alpha=0.7)
#     ax.set_xlabel('SI of %s'%ds_name.upper(), fontsize=10)
#     ax.set_ylabel('TI of %s'%ds_name.upper(), fontsize=10)
#     # ax.grid(True)
#     fig.tight_layout()
#     plt.savefig('%s_siti.png'%ds_name, dpi=300)

pkl_file = os.path.join('siti_pkl', 'vsg_siti_list.pkl')
ds_name = os.path.basename(pkl_file).split('_')[0]
pkl_dict = pkl.load(open(pkl_file, 'rb'))
SI_list = pkl_dict['SI']
TI_list = pkl_dict['TI']

ax=plt.subplot(2,3,1)
plt.scatter(np.array(SI_list), np.array(TI_list), marker='*',alpha=0.7)
plt.title('VSG')
plt.xlabel('SI')
plt.ylabel('TI')

# Hide the right and top spines
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)

# Only show ticks on the left and bottom spines
# ax.yaxis.set_ticks_position('left')
# ax.xaxis.set_ticks_position('bottom')

pkl_file = os.path.join('siti_pkl', 'gazecom_siti_list.pkl')
ds_name = os.path.basename(pkl_file).split('_')[0]
pkl_dict = pkl.load(open(pkl_file, 'rb'))
SI_list = pkl_dict['SI']
TI_list = pkl_dict['TI']

ax=plt.subplot(2,3,2)
plt.scatter(np.array(SI_list), np.array(TI_list), marker='*',alpha=0.7)
plt.title('GazeCom')
plt.xlabel('SI')
plt.ylabel('TI')

# Hide the right and top spines
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)

# Only show ticks on the left and bottom spines
# ax.yaxis.set_ticks_position('left')
# ax.xaxis.set_ticks_position('bottom')


pkl_file = os.path.join('siti_pkl', 'diem_siti_list.pkl')
ds_name = os.path.basename(pkl_file).split('_')[0]
pkl_dict = pkl.load(open(pkl_file, 'rb'))
SI_list = pkl_dict['SI']
TI_list = pkl_dict['TI']

ax=plt.subplot(2,3,3)
plt.scatter(np.array(SI_list), np.array(TI_list), marker='*',alpha=0.7)
plt.title('DIEM')
plt.xlabel('SI')
plt.ylabel('TI')

# Hide the right and top spines
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)

# Only show ticks on the left and bottom spines
# ax.yaxis.set_ticks_position('left')
# ax.xaxis.set_ticks_position('bottom')


pkl_file = os.path.join('siti_pkl', 'savam_siti_list.pkl')
ds_name = os.path.basename(pkl_file).split('_')[0]
pkl_dict = pkl.load(open(pkl_file, 'rb'))
SI_list = pkl_dict['SI']
TI_list = pkl_dict['TI']

ax=plt.subplot(2,3,5)
plt.scatter(np.array(SI_list), np.array(TI_list), marker='*',alpha=0.7)
plt.title('SAVAM')
plt.xlabel('SI')
plt.ylabel('TI')

# Hide the right and top spines
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)

# Only show ticks on the left and bottom spines
# ax.yaxis.set_ticks_position('left')
# ax.xaxis.set_ticks_position('bottom')


pkl_file = os.path.join('siti_pkl', 'ledov_siti_list.pkl')
ds_name = os.path.basename(pkl_file).split('_')[0]
pkl_dict = pkl.load(open(pkl_file, 'rb'))
SI_list = pkl_dict['SI']
TI_list = pkl_dict['TI']

ax=plt.subplot(2,3,6)
plt.scatter(np.array(SI_list), np.array(TI_list), marker='*',alpha=0.7)
plt.title('LEDOV')
plt.xlabel('SI')
plt.ylabel('TI')

# Hide the right and top spines
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)

# Only show ticks on the left and bottom spines
# ax.yaxis.set_ticks_position('left')
# ax.xaxis.set_ticks_position('bottom')


fig = plt.gcf()
fig.set_size_inches(15,9)
plt.savefig('SITI.png', dpi=300)

# plt.show()