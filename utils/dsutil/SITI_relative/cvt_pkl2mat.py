import cPickle as pkl
import os, glob
import scipy.io as sio


pkl_dir = 'siti_pkl'
mat_dir = 'siti_mat'
if not os.path.isdir(mat_dir):
    os.makedirs(mat_dir)

pkl_path_list = glob.glob(os.path.join(pkl_dir, '*.pkl'))
for pkl_path in pkl_path_list:
    database_name = os.path.splitext(os.path.basename(pkl_path))[0].split('_')[0]
    save_name = database_name + '_siti.mat'
    save_path = os.path.join(mat_dir, save_name)

    siti_dict = pkl.load(open(pkl_path, 'rb'))
    sio.savemat(save_path, siti_dict)
    print 'Info:', save_path, 'saved.'
    for key in siti_dict:
        print key
        print siti_dict[key]
