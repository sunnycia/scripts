####sss###uu###uu###nnnn#####nnnn####yy##yyy###cccc##iii####aaa#####
##sss#####uu###uu##nn###nn##nn###nn###y#yyy##cc######iii###a##aa####
#####sss##uu###uu##nn###nn##nn###nn####yy####cc######iii##aaaaaaa###
##sss######uuuu####nn###nn##nn###nn####yy######cccc##iii#aa####aaa##
import cv2
import os, glob
import cPickle
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--framedir", type=str, required=True, help="frame base directory")
parser.add_argument("--outputdir", type=str, required=True, help="output video directory")
parser.add_argument("--videolength", type=int, required=True, help="output video length")
parser.add_argument("--framewildcard", type=str, default='frame_%d.jpg', help="output video length")
parser.add_argument("--resolution", type=tuple, default=(1920, 1080), help="Resolution of video")
args = parser.parse_args()

frame_basedir = args.framedir
output_basedir = args.outputdir
if not os.path.isdir(output_basedir):
    os.makedirs(output_basedir)
video_length = args.videolength
frame_wildcard = args.framewildcard
resolution = args.resolution

fourcc = cv2.VideoWriter_fourcc('D','I','V','X')

def sort_frame_path(frame_path):
    frame_name = os.path.basename(frame_path)
    frame_index = int(frame_name.split('.')[0].split('_')[-1])
    return frame_index

video_dir_list = os.listdir(frame_basedir)
for video_dir in video_dir_list:
    video_name = video_dir + '.avi'
    output_path = os.path.join(output_basedir, video_name)

    frame_dir = os.path.join(frame_basedir, video_dir)
    frame_path_list = glob.glob(os.path.join(frame_dir, '*.*'))
    # print frame_path_list;exit()
    frame_path_list.sort(key=sort_frame_path)
    # print frame_path_list;exit()

    nframe = len(frame_path_list)
    fps = nframe/video_length
    out_video = cv2.VideoWriter(output_path, fourcc, fps, resolution)
    for frame_path in frame_path_list:
        frame = cv2.resize(cv2.imread(frame_path), dsize=resolution)

        out_video.write(frame)
    out_video.release()
    print "Done for video",video_name
