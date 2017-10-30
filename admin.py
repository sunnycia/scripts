
import glob
import os

outputname = 'admin.log'
w_f = open(outputname, 'w')

model_version = 'salicon'

#*#**#*#*#*#**#*#**##*#*#**#*#*#**#*#*#**#*#*#**#*#*#**#*#*#**#*#*#**#*#*#*#***##**#*#*#**#*#*##**##*
model_base = '../training_output/'
model_path = os.path.join(model_base, model_version)
model_list = glob.glob(os.path.join(model_path, '*', '*.caffemodel'))
model_name_list = [os.path.basename(name).split('.')[0].replace('snapshot-', '') for name in model_list]
# print model_name_list;exit()
model_list.sort(key=os.path.getmtime)
print >> w_f, "INFO: Existing model name"
for model in model_list:
    print >> w_f, '\t', model
print >> w_f, len(model_list), 'in total.'

#*#**#*#*#*#**#*#**##*#*#**#*#*#**#*#*#**#*#*#**#*#*#**#*#*#**#*#*#**#*#*#*#***##**#*#*#**#*#*##**##*
cat2000_saliency_base = '/data/sunnycia/SaliencyDataset/Image/CAT2000/trainSet/combine/saliency'
salicon_saliency_base = '/data/sunnycia/SaliencyDataset/Image/SALICON/DATA/train_val/val2014/saliency'
mit1003_saliency_base = '/data/sunnycia/SaliencyDataset/Image/MIT1003/saliency'

saliency_dataset_list = [cat2000_saliency_base, salicon_saliency_base, mit1003_saliency_base]
print >> w_f, "INFO: Existing done saliency."
for saliency_check in saliency_dataset_list:
    print >> w_f, 'INFO: ', saliency_check
    saliency_subdir_list = os.listdir(cat2000_saliency_base)
    saliency_subdir_list.sort()
    for saliency_subdir in saliency_subdir_list:
        print >> w_f, '\t', saliency_subdir
    print >> w_f, len(saliency_subdir_list), 'in total.'
    for model_name in model_name_list:
        pass

#*#**#*#*#*#**#*#**##*#*#**#*#*#**#*#*#**#*#*#**#*#*#**#*#*#**#*#*#**#*#*#*#***##**#*#*#**#*#*##**##*
metric_base = '/data/sunnycia/saliency_on_videoset/Train/metric'
metric_list = glob.glob(os.path.join(metric_base, 'mit*.mat'))
print >> w_f, "INFO: Existing done mit1003 metric."
for metric in metric_list:
    print >>w_f, '\t', metric
print >> w_f, len(metric_list), 'in total.'

metric_list = glob.glob(os.path.join(metric_base, 'salicon*.mat'))
print >> w_f, "INFO: Existing done salicon metric."
for metric in metric_list:
    print >>w_f, '\t', metric
print >> w_f, len(metric_list), 'in total.'


#*#**#*#*#*#**#*#**##*#*#**#*#*#**#*#*#**#*#*#**#*#*#**#*#*#**#*#*#**#*#*#*#***##**#*#*#**#*#*##**##*
w_f.close()
lines = open(outputname, 'r').readlines()
for line in lines:
    print line, 