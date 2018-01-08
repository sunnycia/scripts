import os, glob
import time
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', type=str, required=True)
parser.add_argument('--modeliter', type=int, required=True)
args = parser.parse_args()

model_dir = args.modeldir
model_iter = args.modeliter
modelname_wildcard = 'snapshot-_iter_%d.caffemodel'

current_index = model_iter
while True:
    time.sleep(5)

    ## check if model exists
    model_name = modelname_wildcard % current_index
    model_path = os.path.join(model_dir, model_name)
    short_model_path = os.path.join(os.path.basename(model_dir), model_name)
    # print "Checking", short_model_path,'\r',
    if not os.path.isfile(model_path):
        continue
    print ''
    current_index += model_iter
    cmd = 'python ss_test_video.py --modelname="%s"' % short_model_path
    print "Executing", cmd
    os.system(cmd)