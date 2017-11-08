import cv2
import cPickle
import os
import numpy as np
from scipy.spatial import distance
import glob
import argparse
import imageio


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--videobase', type=str, required=True, help='the network prototxt')
    parser.add_argument('--outputbase', type=str, required=True, help='the network prototxt')
    parser.add_argument('--vowildcard', type=str, default='*.*', help='Dataset length.Show/Cut')
    parser.add_argument('--imgfmt', type=str, default='jpg')
    parser.add_argument('--imgwildcard', type=str, default='frame_%s.%s', help='wildcard of save image name')
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--verbose', type=bool, default='True')
    return parser.parse_args()

print "Parsing arguments..."
args = get_arguments()
vowildcard = args.vowildcard
imgwildcard = args.imgwildcard
imgfmt = args.imgfmt

video_base = args.videobase
if not os.path.isdir(video_base):
    print video_base, "not exists."
    exit()
output_base = args.outputbase
if not os.path.isdir(output_base):
    os.makedirs(output_base)


video_path_list = glob.glob(os.path.join(video_base, vowildcard))
print args.debug    
if args.debug is True:
    print len(video_path_list)
    exit()

for video_path in video_path_list:
    video_name = os.path.basename(video_path).split('.')[0]
    save_dir = os.path.join(output_base, video_name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    video_reader = imageio.get_reader(video_path)
    for frame_idx, frame in enumerate(video_reader):
        frame_name = imgwildcard % (str(frame_idx+1), imgfmt)
        frame_path = os.path.join(save_dir, frame_name)
        if args.verbose:
            print "Handling",video_name, str(frame_idx)
            print "\tSave as", frame_path
        cv2.imwrite(frame_path, frame)

    video_reader.close()

