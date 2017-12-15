import numpy as np 
import glob
import os
import scipy.io as scio
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--metricdir', type=str, required=True)
args = parser.parse_args()

# metric_dir = '../metric-py'
metric_dir = args.metricdir
metric_list = ['CC', 'SIM', 'AUC_JUD', 'AUC_BOR', 'SAUC', 'EMD', 'KLD', 'NSS']
metric_mat_list = glob.glob(os.path.join(metric_dir, '*.mat'))

w_f = open(os.path.join(metric_dir, "result.txt"), 'wf')

for metric_mat in metric_mat_list:
    print "Handling ", metric_mat, "..."
    print >> w_f, metric_mat
    for i in range(len(metric_list)):
        print >> w_f, metric_list[i], '\t\t',
    print >> w_f, ''
    metric = scio.loadmat(metric_mat)
    # print metric;exit()
    saliency_score = metric['saliency_score'];
    for line in saliency_score:
        line = np.array(line)
        # print(len(line));exit()
        # print line, line.shape, type(line), np.mean(line),;exit()
        ## delete nan data
        nan_idx = np.isnan(line)

        # print nan_idx
        line = line[~nan_idx]

        mean_line = round(np.mean(line), 4)
        # std_line = np.std(line)
        print >> w_f, mean_line, '\t', 

    print >> w_f, ''
    for line in saliency_score:
        std_line = round(np.std(line), 4)
        print >> w_f, std_line, '\t', 
    print >> w_f, '\n'