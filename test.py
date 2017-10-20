import cv2
import os
import numpy as np
import sys
import time
import caffe
caffe.set_mode_gpu()
caffe.set_device(0)
MEAN_VALUE = np.array([103.939, 116.779, 123.68])   # BGR
MEAN_VALUE = MEAN_VALUE[:,None, None]

class VideoSaliency:
    def __init__(self, prototxtpath='salicon.prototxt', model='salicon_osie.caffemodel'):
        self.net = caffe.Net(prototxtpath, model, caffe.TEST) 
        
    def preprocess_image(self, img_arr, sub_mean=True):
        img_arr = img_arr.astype(np.float32)
        h, w, c = img_arr.shape
        # subtract mean
        if sub_mean:
            img_arr[:, :, 0] -= MEAN_VALUE[0] # B
            img_arr[:, :, 1] -= MEAN_VALUE[1] # G
            img_arr[:, :, 2] -= MEAN_VALUE[2] # R

        ## this version simply resize the test image
        img_arr = cv2.resize(img_arr, dsize=(480, 288))
        # put channel dimension first
        img_arr = np.transpose(img_arr, (2,0,1))
        img_arr = img_arr[None, :]
        img_arr = img_arr / 255. # convert to float precision
        return img_arr
    
    def postprocess_saliency_map(self, sal_map):
        sal_map = sal_map - np.amin(sal_map)
        sal_map = sal_map / np.amax(sal_map)
        sal_map *= 255
        return sal_map

    def compute_saliency(self, image_path):
        img_arr = cv2.imread(image_path)
        h, w, c = img_arr.shape
        img_arr = self.preprocess_image(img_arr, False)
        print img_arr.shape
        assert img_arr.shape == (1, 3, 288, 480)

        self.net.blobs['data'].data[...] = img_arr
        self.net.forward()
        sal_map = self.net.blobs['saliency_map'].data
        sal_map = sal_map[0,0,:,:]

        saliency_map = self.postprocess_saliency_map(sal_map)

        return saliency_map


if __name__ =='__main__':
    vs = VideoSaliency('deploy.prototxt', '../training_output/ver1/training_output_iter_390000.caffemodel')
    saliency_map = vs.compute_saliency('../test_imgs/face.jpg')
    cv2.imwrite('../test_imgs/frame140.bmp', saliency_map)


