from utils.pymetric.metrics import *
import os
import glob
import scipy.io as scio
import cv2
from utils.tictoc import *
import argparse
from utils.color_print import Colored
# metric_list = ['CC', 'SIM', 'AUC_JUD', 'AUC_BOR', 'SAUC', 'EMD', 'KLD', 'NSS']
# mask_code = [0, 0, 0, 0, ]
def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--dsname', type=str, required=True)
    return parser.parse_args()
args = get_arguments()

metric_save_path = '../metric-py'
# ds_name='mit1003';
ds_name = args.dsname
debug = args.debug
color = Colored()
## salicon
if ds_name == 'videoset':
    sal_base = '/data/sunnycia/pwd/saliency_on_videoset/_Saliencymap/';
    dens_dir = '/data/sunnycia/ImageSet/density-6';
    fixa_dir = '/data/sunnycia/ImageSet/fixation';
    vo_name_prefix='videoSRC'
    fr_name_prefix='frame'
    
if ds_name == 'msu':
    pass
if ds_name == 'ledov':
    pass

def path_based_sort(frame_path, videoname_prefix=vo_name_prefix, framename_prefix=fr_name_prefix):
    ##video index
    # print frame_path
    videoname = os.path.dirname(frame_path).split('/')[-1]
    video_index = int(videoname.replace(videoname_prefix, ''))

    ##frame index
    framename = os.path.basename(frame_path)
    frame_index = int(framename.split('.')[0].replace(framename_prefix, ''))

    return video_index*1000+ frame_index


model_list = os.listdir(sal_base)
for model in model_list:
    if os.path.isfile(os.path.join(sal_base,model)):
        continue
    output_path = os.path.join(metric_save_path, ds_name+'_'+model+'.mat')
    print color.red(output_path)
    if os.path.isfile(output_path):
        print output_path, "already exists, skip.."
        continue
    print "Info: evaluating", color.red(model)
    # if already done:
    #     pass this file

    saliency_map_list = glob.glob(os.path.join(sal_base, model, "*", "*.*"))
    density_map_list = glob.glob(os.path.join(dens_dir, "*",  "*.*"))
    fixation_map_list = glob.glob(os.path.join(fixa_dir,  "*", "*.*"))
    saliency_map_list.sort(key=path_based_sort)
    density_map_list.sort(key=path_based_sort)
    fixation_map_list.sort(key=path_based_sort)
    # print saliency_map_list[-10:], density_map_list[-10:], fixation_map_list[-10:];exit()

    # print len(saliency_map_list), len(density_map_list), len(fixation_map_list)
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

        assert os.path.basename(saliency_map_path).split('.')[0] == os.path.basename(density_map_path).split('.')[0] == os.path.basename(fixation_map_path).split('.')[0]

        saliency_map = cv2.imread(saliency_map_path, 0).astype(np.float32)
        density_map = cv2.imread(density_map_path, 0).astype(np.float32)

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