import imghdr
from math import floor
import glob, cv2, os, numpy as np, sys, caffe
from utils.tictoc import tic, toc
from Saliencynet import Saliencynet
caffe.set_mode_gpu()
caffe.set_device(0)

if __name__ =='__main__':
    model_base = '../training_output/salicon'
    subdirs = [name for name in os.listdir(model_base) if os.path.isdir(os.path.join(model_base, name))]

    for subdir in subdirs:
        model_path_list = glob.glob(os.path.join(model_base, subdir, "*.caffemodel"))
        model_path_list.sort(key=os.path.getmtime)
        length = len(model_path_list)
        if length == 0:
            continue
        # print model_path_list;continue
        if length-1 == floor(length/2.):
            selection_list=[length-1]
        else:
            selection_list = [length-1, int(floor(length/2.))]

        # for model_path in model_path_list:
        for selection in selection_list:
            model_path = model_path_list[selection]
            # print model_path;
            version_postfix = ''
            model_version = model_path.split('/')[-2].split('.')[0]+version_postfix;
            print model_version
            # continue
            # print sub_dirs;exit()
            # model_path = '../training_output/ver1/training_output_iter_390000.caffefemodel'
            if '5layer' in model_path:
                sn = Saliencynet('prototxt/deploy_5layer_deconv.prototxt', model_path)
            else:
                sn = Saliencynet('prototxt/deploy_3layer_deconv.prototxt', model_path)
            # test script for a single image
            # saliency_map = sn.compute_saliency('../test_imgs/face.jpg')
            # cv2.imwrite('../test_imgs/frame140.bmp', saliency_map)

            # test_img_dir = '/data/sunnycia/SaliencyDataset/Image/SALICON/DATA/train_val/val2014/images'
            # test_img_dir  = '/data/sunnycia/SaliencyDataset/Image/CAT2000/trainSet/combine/Stimuli'
            # test_img_dir  = '/data/sunnycia/SaliencyDataset/Image/MIT1003/ALLSTIMULI'
            # test_img_dir  = '/data/sunnycia/SaliencyDataset/Image/NUS/Color'
            # test_img_dir = '/data/sunnycia/SaliencyDataset/Image/NCTU/AllTestImg/Limages'

            test_img_dir = '/data/sunnycia/SaliencyDataset/Video/MSU/frames_allinone'

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
                ## Check if test_img_path is a valid image file
                if imghdr.what(test_img_path) is None:
                    print test_img_path, "is not an image file"
                    continue
                saliency_map = sn.compute_saliency(test_img_path)
                output_path = os.path.join(test_output_dir, img_name)
                cv2.imwrite(output_path, saliency_map)
                duration = toc()
                print output_path, "saved. %s passed" % duration 