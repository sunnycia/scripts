from caffe.proto import caffe_pb2
import google.protobuf.text_format as txtf

net = caffe_pb2.NetParameter()
fn = 'net/play.prototxt'
with open(fn) as f:
    s = f.read()
    txtf.Merge(s, net)

# net.name = 'my new net'
layerNames = [l.name for l in net.layer]
# print layerNames;exit()
idx = layerNames.index('data')
l = net.layer[idx]
print l.python_param, l ;
batch = 128
l.python_param.param_str = str(batch)+",3,288,480"
# print l.python_param, l ;exit()

outFn = 'net/newplay.prototxt'
print 'writing', outFn
with open(outFn, 'w') as f:
    f.write(str(net))