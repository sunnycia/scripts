from utils.pymetric.metrics import *
import os
import glob
import scipy.io as scio
import cv2
from utils.common import *
import argparse
# from utils. import Colored
# metric_list = ['CC'color_print, 'SIM', 'AUC_JUD', 'AUC_BOR', 'SAUC', 'EMD', 'KLD', 'NSS']
# mask_code = [0, 0, 0, 0, ]
def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--dsname', type=str, required=True)
    parser.add_argument('--modelname', type=str, default=None)
    return parser.parse_args()
args = get_arguments()

metric_save_path = '../metric-py'
# ds_name='mit1003';
ds_name = args.dsname
debug = args.debug
color = Colored()
## salicon
if ds_name == 'salicon':
    sal_base = '/data/sunnycia/SaliencyDataset/Image/SALICON/DATA/train_val/val2014/saliency/';
    dens_dir = '/data/sunnycia/SaliencyDataset/Image/SALICON/DATA/train_val/val2014/density';
    fixa_dir = '/data/sunnycia/SaliencyDataset/Image/SALICON/DATA/train_val/val2014/fixation';
elif ds_name == 'mit1003':
    ## mit1003
    sal_base = '/data/sunnycia/SaliencyDataset/Image/MIT1003/saliency'
    dens_dir = '/data/sunnycia/SaliencyDataset/Image/MIT1003/ALLFIXATIONMAPS'
    fixa_dir = '/data/sunnycia/SaliencyDataset/Image/MIT1003/fixPts'
elif ds_name == 'nctu':
    sal_base = '/data/sunnycia/SaliencyDataset/Image/NCTU/AllTestImg/saliency'
    dens_dir = '/data/sunnycia/SaliencyDataset/Image/NCTU/AllFixMap/sigma_52'
    fixa_dir = '/data/sunnycia/SaliencyDataset/Image/NCTU/AllFixPtsMap/FixPtsMap_allfixs'
elif ds_name == 'msu':
    sal_base = '/data/sunnycia/SaliencyDataset/Video/MSU/saliency'
    dens_dir = '/data/sunnycia/SaliencyDataset/Video/MSU/density/sigma32_allinone'
    fixa_dir = '/data/sunnycia/SaliencyDataset/Video/MSU/fixation/mat_allinone'
elif ds_name == 'videoset':
    sal_base = '/data/sunnycia/SaliencyDataset/Video/VideoSet/saliency_map'
    dens_dir = '/data/sunnycia/SaliencyDataset/Video/VideoSet/All_in_one/density'
    fixa_dir = '/data/sunnycia/SaliencyDataset/Video/VideoSet/All_in_one/fixation'

# elif ds_name == 'nus':
#     sal_base = '/data/sunnycia/SaliencyDataset/Image/NUS/saliency'
#     dens_dir = '/data/sunnycia/SaliencyDataset/Image/NUS/Density'
#     fixa_dir = '/data/sunnycia/SaliencyDataset/Image/NUS/Fixation'

if ds_name not in sal_base.lower():
    print "Caution the dataset version"
    exit()

def videoset_name_sort(imagepath):
    basename = os.path.basename(imagepath)
    video_index = int(basename.split('_')[0].replace('videoSRC', ''))
    frame_index = int(basename.split('.')[0].split('_')[-1])

    return video_index * 1000 + frame_index

evaluation_list = os.listdir(sal_base)
for evaluation in evaluation_list:
    # if evaluation == 'v1':
    #     continue
    if args.modelname is not None:
        evaluation=args.modelname

    output_path = os.path.join(metric_save_path, ds_name+'_'+evaluation+'.mat')
    print color.red(output_path)
    if os.path.isfile(output_path):
        print output_path, "already exists, skip.."
        continue
    print "Info: evaluating", color.red(evaluation)
    # if already done:
    #     pass this file

    # saliency_map_list = glob.glob(os.path.join(sal_base, "*.*"))
    saliency_map_list = glob.glob(os.path.join(sal_base, evaluation, "*.*"))
    density_map_list = glob.glob(os.path.join(dens_dir,  "*.*"))
    fixation_map_list = glob.glob(os.path.join(fixa_dir,  "*.*"))
    if ds_name =='videoset':
        saliency_map_list.sort(key=videoset_name_sort)
        density_map_list.sort(key=videoset_name_sort)
        fixation_map_list.sort(key=videoset_name_sort)
    else:
        saliency_map_list.sort()
        density_map_list.sort()
        fixation_map_list.sort()
    print len(saliency_map_list), len(density_map_list), len(fixation_map_list)
    assert len(saliency_map_list)==len(density_map_list) and len(density_map_list) == len(fixation_map_list)

    ## compute other map

    length = len(saliency_map_list)
    saliency_score_cc = np.zeros((length))
    saliency_score_sim = np.zeros((length))
    saliency_score_jud = np.zeros((length))
    saliency_score_bor = np.zeros((length))
    saliency_score_sauc = np.zeros((length))
    saliency_score_emd = np.zeros((length))
    saliency_score_kld = np.zeros((length))
    saliency_score_nss = np.zeros((length))
    if debug:
        length = 2

    for i in range(length):
        saliency_map_path = saliency_map_list[i]
        density_map_path = density_map_list[i]
        fixation_map_path = fixation_map_list[i]

        print os.path.basename(saliency_map_path).split('.')[0], os.path.basename(density_map_path).split('.')[0], os.path.basename(fixation_map_path).split('.')[0]
        assert os.path.basename(saliency_map_path).split('.')[0] == os.path.basename(density_map_path).split('.')[0] == os.path.basename(fixation_map_path).split('.')[0]

        saliency_map = cv2.imread(saliency_map_path, 0).astype(np.float32)
        density_map = cv2.imread(density_map_path, 0).astype(np.float32)
        ##resize saliency map to the same size as density map
        h,w = density_map.shape
        saliency_map = cv2.resize(saliency_map, dsize=(w,h))
        # print saliency_map.shape, density_map.shape;exit()

        if os.path.basename(fixation_map_path).split('.')[-1] == 'mat':
            # print "loading mat file"
            fixation_map = scio.loadmat(fixation_map_path)['fixation']
        else:
            fixation_map = cv2.imread(fixation_map_path, 0)
        # print fixation_map_path
        # print type(fixation_map)

        print "Computing metric for", os.path.basename(saliency_map_path), "..."
        tic()
        saliency_score_cc[i] = CC(saliency_map, density_map)
        # print 'cc', toc(), 
        saliency_score_sim[i] = SIM(saliency_map, density_map)
        # print 'sim', toc(), 
        saliency_score_jud[i] = AUC_Judd(saliency_map, fixation_map, False)
        # print 'jud', toc(), 
        saliency_score_bor[i] = AUC_Borji(saliency_map, fixation_map)
        # saliency_score_sauc[i] = AUC_shuffled(saliency_map, density_map, other_map)
        # saliency_score_emd[i] = EMD(saliency_map, density_map)
        # print 'bor', toc(), 
        saliency_score_kld[i] = KLdiv(saliency_map, density_map)
        # print 'kld', toc(), 
        saliency_score_nss[i] = NSS(saliency_map, fixation_map)
        # print 'nss', toc(), 
        # saliency_score_infoGain[i] = CC(saliency_map, density_map)

        print "INFO: Done!" , toc(), "second passed"
        print color.blue([saliency_score_cc[i], saliency_score_sim[i], saliency_score_jud[i], saliency_score_bor[i], saliency_score_sauc[i], saliency_score_emd[i], saliency_score_kld[i], saliency_score_nss[i]])

    saliency_score = [saliency_score_cc, saliency_score_sim, saliency_score_jud, saliency_score_bor, saliency_score_sauc, saliency_score_emd, saliency_score_kld, saliency_score_nss]

    if not debug == True:
        scio.savemat(output_path, {'saliency_score':saliency_score})
        print output_path, 'saved'
    if args.modelname is not None:
        break