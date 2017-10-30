import glob
import cv2
import os
import numpy as np
import sys
import caffe
from utils.tictoc import tic, toc
caffe.set_mode_gpu()
caffe.set_device(0)
MEAN_VALUE = np.array([103.939, 116.779, 123.68])   # BGR
MEAN_VALUE = MEAN_VALUE[:,None, None]

class VideoSaliency:
    def __init__(self, prototxtpath, model):
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
        sal_map = cv2.resize(sal_map, dsize=(self.w, self.h))
        return sal_map

    def compute_saliency(self, image_path):
        img_arr = cv2.imread(image_path)
        self.h, self.w, self.c = img_arr.shape # store the image's original height and width
        img_arr = self.preprocess_image(img_arr, False)
        # print img_arr.shape
        assert img_arr.shape == (1, 3, 288, 480)

        self.net.blobs['data'].data[...] = img_arr
        self.net.forward()
        sal_map = self.net.blobs['saliency_map'].data
        sal_map = sal_map[0,0,:,:]
        saliency_map = self.postprocess_saliency_map(sal_map)
        return saliency_map

if __name__ =='__main__':
    model_base = '../training_output/salicon'
    subdirs = [name for name in os.listdir(model_base) if os.path.isdir(os.path.join(model_base, name))]

    for subdir in subdirs:
        model_path_list = glob.glob(os.path.join(model_base, subdir, "*.caffemodel"))
        for model_path in model_path_list:
            print model_path
            # print sub_dirs;exit()
            # model_path = '../training_output/ver1/training_output_iter_390000.caffefemodel'

            vs = VideoSaliency('deploy.prototxt', model_path)
            # test script for a single image
            # saliency_map = vs.compute_saliency('../test_imgs/face.jpg')
            # cv2.imwrite('../test_imgs/frame140.bmp', saliency_map)

            version_postfix = ''
            model_version = model_path.split('/')[-1].split('.')[0]+version_postfix
            # test_img_dir = '/data/sunnycia/SaliencyDataset/Image/SALICON/DATA/train_val/val2014/images'
            # test_img_dir  = '/data/sunnycia/SaliencyDataset/Image/CAT2000/trainSet/combine/Stimuli'
            test_img_dir  = '/data/sunnycia/SaliencyDataset/Image/MIT1003/ALLSTIMULI'

            test_img_path_list = glob.glob(os.path.join(test_img_dir, '*.*'))
            test_output_dir = os.path.join(os.path.dirname(test_img_dir), 'saliency', model_version)

            if not os.path.isdir(test_output_dir):
                os.makedirs(test_output_dir)
            else:
                print test_output_dir, 'exists, pass.'
                continue

            for test_img_path in test_img_path_list:
                img_name = test_img_path.split('/')[-1]
                start_time = tic()
                saliency_map = vs.compute_saliency(test_img_path)
                output_path = os.path.join(test_output_dir, img_name)
                cv2.imwrite(output_path, saliency_map)
                duration = toc()
                print output_path, "saved. %s passed" % duration
