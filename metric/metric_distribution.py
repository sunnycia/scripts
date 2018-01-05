import os, glob
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import scipy.io as sio, numpy as np
import argparse 

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metricdir', required=True, type=str)
    parser.add_argument('--outputdir', default='../metric_distribution', type=str)
    parser.add_argument('--metricname', default='CC')
    return parser.parse_args()

print "Parsing arguments..."
args = get_arguments()
metric_dir = args.metricdir
output_dir = args.outputdir
metric_name = args.metricname
model_name = os.path.basename(metric_dir)

if not os.path.isdir(metric_dir):
    print metric_dir, 'not exists. Abort.'
    exit()

if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

metric_list = ['cc', 'sim', 'auc_jud', 'auc_bor', 'sauc', 'emd', 'kld', 'nss']
index = metric_list.index(metric_name)

metric_mat_list = glob.glob(os.path.join(metric_dir, '*.mat'))

metric_list = []
for metric_mat in metric_mat_list:
    print "Handling ", metric_mat, "..."
    metric = sio.loadmat(metric_mat)
    # print metric;exit()
    saliency_score = metric['saliency_score'];

    line = saliency_score[index]
    line = line[~np.isnan(line)]
    metric_list.append(round(np.mean(line), 2))

print metric_list
num_bins = 32

fig, ax = plt.subplots()

# the histogram of the data
n, bins, patches = ax.hist(metric_list, num_bins)
print n, bins, patches

ax.set_xlabel(metric_name+' metric')
ax.set_ylabel('video numbers')

# Tweak spacing to prevent clipping of ylabel
fig.tight_layout()

# plt.show()
fig_name = metric_name +'-'+ model_name + '.png'
plt.savefig(os.path.join(output_dir, fig_name))