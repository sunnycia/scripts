import sys
caffe_path = '../C3D-v1.1-tmp/python'
sys.path.insert(0, caffe_path)
import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2
from math import ceil

# use the 3d convolution setting of C3D-v1.1
# https://github.com/facebook/C3D/tree/master/C3D-v1.1
def conv3d(netspec, name_prefix, bottom, num_output, kernel_size=3,t_kernel_size=3, pad=1, t_pad=1, stride=1, t_stride=1,dropout=0):

    bottom = netspec[name_prefix+'_conv'] = L.Convolution3D(bottom, 
                          param=[dict(lr_mult=1, decay_mult=1)], bias_term=False,
                          convolution3d_param=dict(num_output=num_output,  weight_filler=dict(type='msra'), 
                          kernel_size=kernel_size, kernel_depth=t_kernel_size, pad=pad,temporal_pad=t_pad,
                          stride=stride, temporal_stride=t_stride,))
    bottom = netspec[name_prefix+'_bn'] = L.BatchNorm(bottom, batch_norm_param=dict(use_global_stats=False), in_place=True,
                              param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=1, decay_mult=1),
                                     dict(lr_mult=1, decay_mult=1)])
    bottom = netspec[name_prefix+'_scale'] = L.Scale(bottom, in_place=True,param=[dict(lr_mult=1, decay_mult=1), 
                                                    dict(lr_mult=1, decay_mult=1)],
                                                    scale_param=dict(bias_term=True))
    if dropout > 0:
        bottom = netspec[name_prefix+'_relu'] = L.Dropout(L.ReLU(bottom, in_place=True), dropout_param=dict(dropout_ratio=dropout))
    else:
        bottom = netspec[name_prefix+'_relu'] = L.ReLU(bottom, in_place=True)

    # return name_prefix+'_relu'
    return bottom 

def deconv3d(netspec, name_prefix, bottom, num_output, factor=2,t_factor=2,dropout=0.5):

    ## calculate ks,pad,stride base on factor
    ks = lambda factor: int(2.*factor-factor%2.)
    pd = lambda factor: int(ceil((factor-1)/2.))
    st = lambda factor: int(factor)

    kernel_size = ks(factor);     kernel_depth = ks(t_factor)
    pad = pd(factor);             t_pad = pd(t_factor)
    stride = st(factor);          t_stride = st(factor)



    bottom = netspec[name_prefix+'_deconv'] = L.Deconvolution3D(bottom, 
                          param=[dict(lr_mult=1, decay_mult=1),
                          dict(lr_mult=1, decay_mult=1)], 
                          convolution3d_param=dict(kernel_size=kernel_size, kernel_depth=kernel_depth,
                                              pad=pad,temporal_pad=t_pad,stride=stride, temporal_stride=t_stride,
                                              num_output=num_output, weight_filler=dict(type='msra'), bias_term=False))
    bottom = netspec[name_prefix+'_bn'] = L.BatchNorm(bottom, batch_norm_param=dict(use_global_stats=False), in_place=True,
                              param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=1, decay_mult=1),
                                     dict(lr_mult=1, decay_mult=1)])
    bottom = netspec[name_prefix+ '_scale'] = L.Scale(bottom, in_place=True,param=[dict(lr_mult=1, decay_mult=1), 
                            dict(lr_mult=1, decay_mult=1)],
                            scale_param=dict(bias_term=True))

    if dropout > 0:
        bottom = netspec[name_prefix + '_relu'] = L.Dropout(L.ReLU(bottom, in_place=True), dropout_param=dict(dropout_ratio=dropout))
    else:
        bottom = netspec[name_prefix + '_relu'] = L.ReLU(bottom, in_place=True)
    return bottom


def trim_dense_block(netspec, name_prefix, bottom, layer_output=16, dropout=0,phase='train'):
    if phase =='train':
        use_global_stats = False
    else:
        use_global_stats=True
    
    current_bottom = conv3d(netspec, name_prefix+'_conv1', bottom,num_output=layer_output,dropout=dropout)
    concat1 = netspec[name_prefix+'_concat1'] = L.Concat(bottom, current_bottom,axis=1)

    current_bottom = conv3d(netspec, name_prefix+'_conv2', current_bottom, num_output=layer_output, dropout=dropout)
    concat2 = netspec[name_prefix+'_concat2'] = L.Concat(concat1, current_bottom,axis=1)

    current_bottom = conv3d(netspec, name_prefix+'_conv3', current_bottom, num_output=layer_output, dropout=dropout)
    concat3 = netspec[name_prefix+'_concat3'] = L.Concat(concat2, current_bottom,axis=1)

    current_bottom = conv3d(netspec, name_prefix+'_conv4', current_bottom, num_output=layer_output, dropout=dropout)
    current_bottom = netspec[name_prefix+'_concat4'] = L.Concat(concat3, current_bottom,axis=1)

    # name_prefix + 'concat4'
    # return name_prefix+'concat4'
    return current_bottom


def trimmed_dense3d_feature_network(netspec, input_layer, dropout=0):
    # input_layer = 
    # input size(3,16,112,112), output feature map = (1024,2,7,7)

    current_bottom = conv3d(netspec, 'conv1', input_layer, 64, kernel_size=7, t_kernel_size=3,pad=3,t_pad=1,stride=2,t_stride=1,dropout=dropout)
    # feature map (64,16,56,56)

    current_bottom = trim_dense_block(netspec, 'block1', current_bottom, dropout=dropout)
    
    current_bottom = conv3d(netspec, 'trans1', current_bottom, 192,dropout=dropout)
    current_bottom = netspec['tran1_pool'] = L.Pooling3D(current_bottom, pooling3d_param=dict(pool=0,kernel_depth=2,kernel_size=2,stride=2,temporal_stride=2))
    # feature map (192,8,28,28)

    current_bottom = trim_dense_block(netspec, 'block2',current_bottom,dropout=dropout)
    
    current_bottom = conv3d(netspec, 'trans2', current_bottom, 448,dropout=dropout)
    current_bottom = netspec['trans2_pool'] = L.Pooling3D(current_bottom, pooling3d_param=dict(pool=0,kernel_depth=2,kernel_size=2,stride=2,temporal_stride=2))
    # feature map (448,4,14,14)


    current_bottom = trim_dense_block(netspec, 'block3', current_bottom,dropout=dropout)
    current_bottom = conv3d(netspec, 'trans3',current_bottom,1024,stride=2,t_stride=2,dropout=dropout)
    # current_bottom = netspec['tans3_pool'] = L.Pooling3D(current_bottom, pooling3d_param=dict(pool=0,kernel_depth=2,kernel_size=2,stride=2,temporal_stride=2))
    # feature map (1024,2,7,7)

    # return trans3
    return current_bottom

def saliency_reconstruct_network(netspec, feature, dropout=0.5):

    # input feature map (1024,2,7,7)

    bottom = deconv3d(netspec, 'deconv1', feature, 256, dropout=0.5)
    # feature map (256,4,14,14)
    bottom = deconv3d(netspec, 'deconv2', bottom, 64, dropout=0.5)
    # feature map (256,8,28,28)
    bottom = deconv3d(netspec, 'deconv3', bottom, 16, dropout=0.5)
    # feature map (64,16,56,56)
    

    bottom = netspec['predict'] = L.Deconvolution3D(bottom, 
                          param=[dict(lr_mult=1, decay_mult=1),
                          dict(lr_mult=1, decay_mult=1)], 
                          convolution3d_param=dict(bias_term=False,kernel_size=4, kernel_depth=1,
                                              pad=1,temporal_pad=0,stride=2, temporal_stride=1, 
                                              num_output=1, weight_filler=dict(type='msra')))
    # prediction = deconv3d(netspec, 'prediction', bottom, 1, dropout=0.5)
    # feature map (1,16,112,112)

    return bottom

def trimmed_dense3d_network(input_data_shape='2,3,16,112,112', input_label_shape='2,1,16,112,112'):

    n  = caffe.NetSpec()

    data = n['data']  = L.Python(python_param=dict(param_str=input_data_shape, module="CustomData",layer="CustomData"))
    ground_truth = n['ground_truth']  = L.Python(python_param=dict(param_str=input_label_shape, module="CustomData",layer="CustomData"))

    feature = trimmed_dense3d_feature_network(netspec=n,input_layer=data,dropout=0.5)

    prediction = saliency_reconstruct_network(netspec=n, feature=feature,dropout=0.5)

    n['loss'] = L.L1Loss(prediction, ground_truth)

    return n.to_proto()

if __name__=='__main__':

    print trimmed_dense3d_network()
