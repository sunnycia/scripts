from heapq import nlargest
import numpy as np 
import glob
import os
import scipy.io as scio
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--metricdir', type=str, required=True)
parser.add_argument('--metricindex', type=int, default=2)
parser.add_argument('--examples', type=int, default=5)

args = parser.parse_args()

# metric_dir = '../metric-py'
# metric_list = ['CC', 'SIM', 'AUC_JUD', 'AUC_BOR', 'SAUC', 'EMD', 'KLD', 'NSS']
metric_dir = args.metricdir
metric_mat_list = glob.glob(os.path.join(metric_dir, '*.mat'))

# for metric_mat in metric_mat_list:
#     print metric_mat
# exit()

metric_index = args.metricindex
metric_list = []

for metric_mat in metric_mat_list:
    # print "Handling ", metric_mat, "..."
    metric = scio.loadmat(metric_mat)

    saliency_score = metric['saliency_score'];

    line = np.array(saliency_score[metric_index])
    line = line[~np.isnan(line)]

    metric_list.append(round(np.mean(line), 4))

examples = args.examples
sorted_indexes = np.array(metric_list).argsort()
print "The best",str(examples), "index:",sorted_indexes[:examples]
print "The worst",str(examples), "index:",sorted_indexes[-examples:]
