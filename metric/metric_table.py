import numpy as np 
import glob
import os
import scipy.io as scio
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--videoset_name', type=str, required=True)
parser.add_argument('--first_line', type=str, default='model',help='model or metric')
args = parser.parse_args()



# metric_dir = '../metric-py'
videoset_name = args.videoset_name
first_line = args.first_line
metric_list = ['CC', 'SIM', 'AUC_JUD', 'AUC_BOR']

if videoset_name =='videoset':
    metric_dir = '/data/sunnycia/saliency_on_videoset/Train/metric-matlab/videoset'
elif videoset_name == 'msu':
    metric_dir = '/data/sunnycia/saliency_on_videoset/Train/metric-matlab/msu'
elif videoset_name == 'diem':
    metric_dir = '/data/sunnycia/saliency_on_videoset/Train/metric-matlab/diem'
elif videoset_name == 'gazecom':
    metric_dir = '/data/sunnycia/saliency_on_videoset/Train/metric-matlab/gazecom'


else:
    exit()


metric_mat_list = glob.glob(os.path.join(metric_dir, '*.mat'))
save_path=os.path.join(metric_dir, "result.txt")
w_f = open(save_path, 'wf')



##
## 1 metric on the first line
##
if first_line=='metric':
    print >> w_f, 'Model\t&%s\t&%s\t&%s\t&%s\t\\\\' %(metric_list[0], metric_list[1], metric_list[2], metric_list[3])
    for metric_mat in metric_mat_list:
        print "Handling ", metric_mat, "..."
        metric = scio.loadmat(metric_mat)
        # print metric;exit()
        # print metric;exit()

        model_name = os.path.basename(metric_mat).split('.')[0].replace('-result', '')
        saliency_score = metric['result'];
        print saliency_score.flatten();
        score_line = model_name+'&\t'
        for i in range(len(metric_list)):
            score = str(round(saliency_score[i],4))
            score_line+=score+'&\t'
        score_line +='\\\\'
        print >> w_f, score_line


##
## 2 model on the first line
##
if first_line=='model':
    first_line = 'Model\t'
    for metric_mat in metric_mat_list:
        model_name = os.path.basename(metric_mat).split('.')[0].replace('-result', '')
        first_line+='& '+model_name+'\t'
    first_line += '\\\\'
    print >> w_f, first_line;
    for i in range(len(metric_list)):
        score_line = metric_list[i]+'\t'
        for metric_mat in metric_mat_list:
            saliency_score = scio.loadmat(metric_mat)['result']
            score = str(round(saliency_score[i], 4))
            score_line += '& '+score+'\t'
        score_line += '\\\\'

        print >> w_f, score_line
w_f.close()
print 'the result is saved to',save_path