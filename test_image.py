import argparse
import imghdr
from math import floor
import glob, cv2, os, numpy as np, sys, caffe
from utils.common import tic, toc
from Saliencynet import ImageSaliencyNet
caffe.set_mode_gpu()
caffe.set_device(0)

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelpath', type=str, required=True)
    parser.add_argument('--modelbase',type=str,default='../training_output/salicon')
    parser.add_argument('--modelname', type=str, default=None, help='the network prototxt')
    parser.add_argument('--testset', type=str, default=None, help='the network prototxt')
    parser.add_argument('--iterselection',type=str, default='newest',help='(full) or (half) or (newest)')
    parser.add_argument('--versionpostfix',type=str, default='',help='manual added model version postfix')
    return parser.parse_args()
print "Parsing arguments..."
args = get_arguments()
iter_selection = args.iterselection

test_set = args.testset
test_img_dir_dict={
    'salicon':'/data/sunnycia/SaliencyDataset/Image/SALICON/DATA/train_val/val2014/images',
    'cat2000':'/data/sunnycia/SaliencyDataset/Image/CAT2000/trainSet/combine/Stimuli',
    'mit1003':'/data/sunnycia/SaliencyDataset/Image/MIT1003/ALLSTIMULI',
    'nus': '/data/sunnycia/SaliencyDataset/Image/NUS/Color',
    'nctu': '/data/sunnycia/SaliencyDataset/Image/NCTU/AllTestImg/Limages',
    'msu': '/data/sunnycia/SaliencyDataset/Video/MSU/frames_allinone',
    'videoset': '/data/sunnycia/SaliencyDataset/Video/VideoSet/All_in_one/frame',
    'mit300': '/data/sunnycia/SaliencyDataset/Image/MIT300/BenchmarkIMAGES',
    'hdreye-hdr': '/data/SaliencyDataset/Image/HDREYE/images/HDR', 
    'hdreye-ldr': '/data/SaliencyDataset/Image/HDREYE/images/LDR-JPG'
}

if __name__ =='__main__':
    model_path = args.modelpath
    version_postfix = args.versionpostfix
    model_version = model_path.split('/')[-2].split('.')[0]+model_path.split('/')[-1].split('.')[0]+version_postfix;
    if '5layer' in model_path:
        sn = ImageSaliencyNet('prototxt/deploy_5layer_deconv.prototxt', model_path)
    else:
        sn = ImageSaliencyNet('prototxt/deploy_3layer_deconv.prototxt', model_path)

    test_img_dir = test_img_dir_dict[test_set]
    test_img_path_list = glob.glob(os.path.join(test_img_dir, '*.*'))
    test_output_dir = os.path.join(os.path.dirname(test_img_dir), 'saliency', model_version)

    if not os.path.isdir(test_output_dir):
        os.makedirs(test_output_dir)
    elif len(glob.glob(os.path.join(test_output_dir,'*.*'))) == 0:
        pass
    else:
        print test_output_dir, 'exists, pass.'
        exit()

    for test_img_path in test_img_path_list:
        img_name = test_img_path.split('/')[-1]
        start_time = tic()
        ## Check if test_img_path is a valid image file
        # if imghdr.what(test_img_path) is None or not test_img_path.endswith('.hdr'):
        #     print test_img_path, "is not an image file"
        #     exit()
        try:
            saliency_map = sn.compute_saliency(test_img_path)
        except:
            continue
        output_path = os.path.join(test_output_dir, img_name.split('.')[0]+'.jpg')
        cv2.imwrite(output_path, saliency_map)
        duration = toc()
        print output_path, "saved. %s passed" % duration 

'''
    model_base = args.modelbase
    subdirs = [name for name in os.listdir(model_base) if os.path.isdir(os.path.join(model_base, name))]

    for subdir in subdirs:
        if args.modelname is not None:
            subdir=args.modelname
        model_path_list = glob.glob(os.path.join(model_base, subdir, "*.caffemodel"))
        model_path_list.sort(key=os.path.getmtime)
        # print model_path_list;exit()
        length = len(model_path_list)
        if length == 0:
            continue
        # print model_path_list;continue
        selection_list=[]
        if iter_selection=='half':
            if length-1 == floor(length/2.):
                selection_list=[length-1]
            else:
                selection_list = [length-1, int(floor(length/2.))]
        elif iter_selection=='full':
            selection_list=[i for i in range(length)]
        elif iter_selection=='newest':
            selection_list=[length-1]

        # for model_path in model_path_list:
        for selection in selection_list:
            # print selection_list,selection;exit()
            model_path = model_path_list[selection]
            # print model_path;exit()
            version_postfix = ''
            model_version = model_path.split('/')[-2].split('.')[0]+model_path.split('/')[-1].split('.')[0]+version_postfix;
            print model_version
            # continue
            # print sub_dirs;exit()
            # model_path = '../training_output/ver1/training_output_iter_390000.caffefemodel'
            if '5layer' in model_path:
                sn = ImageSaliencyNet('prototxt/deploy_5layer_deconv.prototxt', model_path)
            else:
                sn = ImageSaliencyNet('prototxt/deploy_3layer_deconv.prototxt', model_path)
            # test script for a single image
            # saliency_map = sn.compute_saliency('../test_imgs/face.jpg')
            # cv2.imwrite('../test_imgs/frame140.bmp', saliency_map)

            test_img_dir = test_img_dir_dict[test_set]
            test_img_path_list = glob.glob(os.path.join(test_img_dir, '*.*'))
            test_output_dir = os.path.join(os.path.dirname(test_img_dir), 'saliency', model_version)

            if not os.path.isdir(test_output_dir):
                os.makedirs(test_output_dir)
            elif len(glob.glob(os.path.join(test_output_dir,'*.*'))) == 0:
                pass
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
        if args.modelname is not None:
            print "done for",args.modelname
            break
'''