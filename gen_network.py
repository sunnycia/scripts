import argparse
from caffe_basic_module.caffe_basic_module import *
from caffe_basic_module.caffe_3dnet_module import *
# CAFFE_ROOT = osp.join(osp.dirname(__file__), '..', 'caffe')
# if osp.join(CAFFE_ROOT, 'python') not in sys.path:
#     sys.path.insert(0, osp.join(CAFFE_ROOT, 'python'))
import caffe
print caffe.__file__, 'in gen module'
from caffe.proto import caffe_pb2

def vo_v4_2_resnet_BNdeconv_l1loss_dropout():
    pass

def ns_v1_c3d_resnet18(model_name, batch, clip_length, height,width,loss, phase='train'):
    model = caffe_pb2.NetParameter()
    model.name = model_name
    nums=[2,2,2,2]
    layers = []
    data_channel=3
    data_param_str = str(batch)+','+str(data_channel)+','+str(clip_length)+','+str(height)+','+str(width)
    gt_param_str = str(batch)+',1'+','+str(clip_length)+','+str(height)+','+str(width)
    
    layers.append(Data_python('data', ['data'], param_str=data_param_str))
    if phase=='train':
        layers.append(Data_python('gt', ['gt'], param_str=gt_param_str))
    elif phase=='deploy':
        pass
    else:
        raise NotImplementedError

    layers.append(Conv3d('conv1', 'data', 64, kernel_size=7, kernel_depth=3, stride=2,temporal_stride=1,pad=3,temporal_pad=1))
    layers.extend(Bn_Sc('conv1', layers[-1].top[0], False))
    layers.extend(Act('conv1', layers[-1].top[0], act_type='ReLU'))

    layers.extend(Res3dLayer('res2', layers[-1].top[0], nums[0], 64, stride=1, temporal_stride=1, layer_type='first'))
    layers.extend(Res3dLayer('res3', layers[-1].top[0], nums[1], 128, stride=2, temporal_stride=2))
    layers.extend(Res3dLayer('res4', layers[-1].top[0], nums[2], 256, stride=2, temporal_stride=2))
    layers.extend(Res3dLayer('res5', layers[-1].top[0], nums[3], 512, stride=2, temporal_stride=2))

    ############################# upsample layers
    layers.append(Bilinear_upsample_3d('deconv1', layers[-1].top[0], 128, factor=4, temporal_factor=2))
    # layers.extend(Bn_Sc('deconv1', layers[-1].top[0]))
    # layers.extend(Act('deconv1', layers[-1].top[0]))
    layers.append(Dropout('deconv2', layers[-1].top[0], dropout_ratio=0.5))
    layers.append(Bilinear_upsample_3d('deconv2', layers[-1].top[0], 32, factor=2, temporal_factor=2))
    # layers.extend(Bn_Sc('deconv2', layers[-1].top[0]))
    # layers.extend(Act('deconv2', layers[-1].top[0]))
    layers.append(Dropout('deconv2', layers[-1].top[0], dropout_ratio=0.5))
    layers.append(Bilinear_upsample_3d('predict', layers[-1].top[0], 1, factor=2, temporal_factor=2))
    # layers.extend(Bn_Sc('predict', layers[-1].top[0]))
    # layers.extend(Act('predict', layers[-1].top[0]))

    if phase=='train':
        layers.extend(LossLayer('loss', ['predict', 'gt'], loss_type=loss))
    elif phase=='deploy':
        pass
    else:
        raise NotImplementedError
    print layers
    model.layer.extend(layers)
    return model

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--network_path', type=str, required=True, help='path of output network prototxt')
    parser.add_argument('--clip_length', type=int)
    parser.add_argument('--height', type=int)
    parser.add_argument('--width', type=int)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--loss', type=str, default='EuclideanLoss')
    parser.add_argument('--model', type=str, default='vo_v4_2_connect_resnet_dropout')
    args = parser.parse_args()
    
    model_name= args.model
    model = eval(model_name)(model_name=model_name, 
                         batch=args.batch,
                         clip_length=args.clip_length,
                         height=args.height,
                         width=args.width,
                         loss=args.loss)
    print args.network_path
    with open(args.network_path, 'w') as f:
        f.write(pb.text_format.MessageToString(model))

    model = eval(model_name)(model_name=model_name, 
                         batch=args.batch,
                         clip_length=args.clip_length,
                         height=args.height,
                         width=args.width,
                         loss=args.loss,
                         phase='deploy')

    with open(os.path.join(os.path.dirname(args.network_path), model_name+'_deploy.prototxt'), 'w') as f:
        f.write(pb.text_format.MessageToString(model))