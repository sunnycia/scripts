import glob, cv2, os, numpy as np, sys, caffe
from utils.tictoc import tic, toc
import Saliencynet.Saliencynet
caffe.set_mode_gpu()
caffe.set_device(0)


if __name__ =='__main__':
    model_base = '../training_output/salicon'
    subdirs = [name for name in os.listdir(model_base) if os.path.isdir(os.path.join(model_base, name))]

    for subdir in subdirs:
        model_path_list = glob.glob(os.path.join(model_base, subdir, "*.caffemodel"))
        for model_path in model_path_list:
            print model_path
            # print sub_dirs;exit()
            # model_path = '../training_output/ver1/training_output_iter_390000.caffefemodel'

            sn = Saliencynet('deploy.prototxt', model_path)
            # test script for a single image
            # saliency_map = sn.compute_saliency('../test_imgs/face.jpg')
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
                saliency_map = sn.compute_saliency(test_img_path)
                output_path = os.path.join(test_output_dir, img_name)
                cv2.imwrite(output_path, saliency_map)
                duration = toc()
                print output_path, "saved. %s passed" % duration
