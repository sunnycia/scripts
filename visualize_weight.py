import os
import caffe
import argparse
from utils.caffe_tools import visualize_weights

# python visualize_weight.py --deploy='prototxt/deploy_3layer_deconv.prototxt' 
# --model='../training_output/salicon/train_kldloss_withouteuc-batch-8_1509584263/snapshot-_iter_100000.caffemodel' 
# --output='upsample1.jpg' --layer='upsample_1'

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--deploy', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--viz', type=bool, default=False)
    parser.add_argument('--layer', type=str, default='conv1')
    return parser.parse_args()

print "Parsing argument.."
args = get_arguments()

net = caffe.Net(args.deploy, args.model, caffe.TEST)

##check if output directory exists
output_dir = os.path.dirname(args.output)
if not output_dir == '':
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

if not args.layer == 'all':
    visualize_weights(net, args.layer, filename=args.output, visualize=args.viz)
else:
    #NOT IMPLEMENT YET
    pass