import os, sys, numpy as np
import cv2
import argparse
from scipy import misc
import caffe
import tempfile
from math import ceil

class Flownet():
    def __init__(self, deploy_proto, caffe_model, img_size=(1920,1080), gpu_id=0):
        self.deploy_proto = deploy_proto
        self.caffe_model = caffe_model
        self.img_size = img_size
        self.width = img_size[0]
        self.height = img_size[1]
        self.gpu_id = gpu_id
        self.net = self.set_up_network()

    def set_up_network(self):
        vars = {}
        vars['TARGET_WIDTH'] = self.img_size[0]
        vars['TARGET_HEIGHT'] = self.img_size[1]

        divisor = 64.
        vars['ADAPTED_WIDTH'] = int(ceil(self.width/divisor) * divisor)
        vars['ADAPTED_HEIGHT'] = int(ceil(self.height/divisor) * divisor)

        vars['SCALE_WIDTH'] = self.width / float(vars['ADAPTED_WIDTH']);
        vars['SCALE_HEIGHT'] = self.height / float(vars['ADAPTED_HEIGHT']);

        tmp = tempfile.NamedTemporaryFile(mode='w', delete=False)
        proto = open(self.deploy_proto).readlines()
        for line in proto:
            for key, value in vars.items():
                tag = "$%s$" % key
                line = line.replace(tag, str(value))
            tmp.write(line)
        tmp.flush()

        caffe.set_device(self.gpu_id)
        caffe.set_mode_gpu()
        return caffe.Net(tmp.name, self.caffe_model, caffe.TEST)

    def get_optical_flow(self, img1_path, img2_path):

        num_blobs = 2
        input_data = []
        # img1 = misc.imread(img1_path)
        # img2 = misc.imread(img2_path)
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        ##assert img1.shape==img2.shape==self.img_size
        if len(img1.shape) < 3: 
            input_data.append(img1[np.newaxis, np.newaxis, :, :])
        else:
            input_data.append(img1[np.newaxis, :, :, :].transpose(0, 3, 1, 2))
        if len(img2.shape) < 3: 
            input_data.append(img2[np.newaxis, np.newaxis, :, :])
        else:
            input_data.append(img2[np.newaxis, :, :, :].transpose(0, 3, 1, 2))
        
        input_dict = {}

        for blob_idx in range(num_blobs):
            input_dict[self.net.inputs[blob_idx]] = input_data[blob_idx]

        print('Network forward pass using %s.' % self.caffe_model)
        i = 1
        while i<=5:
            i+=1

            self.net.forward(**input_dict)

            containsNaN = False
            for name in self.net.blobs:
                blob = self.net.blobs[name]
                has_nan = np.isnan(blob.data[...]).any()

                if has_nan:
                    print('blob %s contains nan' % name)
                    containsNaN = True

            if not containsNaN:
                print('Succeeded.')
                break
            else:
                print('**************** FOUND NANs, RETRYING ****************')

        optical_flow = np.squeeze(self.net.blobs['predict_flow_final'].data).transpose(1, 2, 0)
        self.writeFlow('output.flo', optical_flow)
        # return optical_flow

    def writeFlow(self, output_name, optical_flow):
        f = open(output_name, 'wb')
        f.write('PIEH'.encode('utf-8'))
        np.array([optical_flow.shape[1], optical_flow.shape[0]], dtype=np.int32).tofile(f)
        optical_flow = optical_flow.astype(np.float32)
        optical_flow.tofile(f)
        f.flush()
        f.close() 
        print "Successfully saved into", output_name

    def readFlow(file_name):
        if file_name.endswith('.pfm') or file_name.endswith('.PFM'):
            return readPFM(file_name)[0][:,:,0:2]

        f = open(file_name, 'rb')

        header = f.read(4)
        if header.decode("utf-8") != 'PIEH':
            raise Exception('Flow file header does not contain PIEH')

        width = np.fromfile(f, np.int32, 1).squeeze()
        height = np.fromfile(f, np.int32, 1).squeeze()

        flow = np.fromfile(f, np.float32, width * height * 2).reshape((height, width, 2))

        return flow.astype(np.float32)