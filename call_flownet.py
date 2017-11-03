import argparse
import os 
from Flownet import Flownet

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--code', type=str, default='0', help='the network prototxt')
    return parser.parse_args()
args = get_arguments()


flow_net_model_base='flownet2/models'
flownet_dict={
    '0': 'FlowNet2', 
    'c': 'FlowNet2-c', 
    's': 'FlowNet2-s', 
    'ss': 'Flownet2-ss'
}

code=args.code
deploy_postfix = '_deploy.prototxt.template'
model_postfix = '_weights.caffemodel.h5'
deploy_proto_path = os.path.join(flow_net_model_base, flownet_dict[code], flownet_dict[code]+deploy_postfix)
caffe_model_path = os.path.join(flow_net_model_base, flownet_dict[code], flownet_dict[code]+model_postfix)

# print deploy_proto_path, caffe_model_path
fn = Flownet(deploy_proto_path, caffe_model_path, (480, 270))

of = fn.get_optical_flow('frame5.bmp', 'frame7.bmp')
# fn.writeFlow("test.flo", of)