import cv2
import cPickle
import os
import numpy as np
from scipy.spatial import distance


video_dir = "/data/qp_00"
output_dir = "/home/sunnycia/pwd/saliency_on_videoset/_Dataset/frames-lab"

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)
    
def slice_video(vopath, frame_dir, format="jpg", rescale=1):
    ## a table for videocapture.get(idpropt)
    # 0 
    # 1 
    # 2 frame duration
    # 3 frame width
    # 4 frame heigh
    # 5 frame per second fps
    # 6
    # 7 frame count
    # 8

    print "Handling", vopath
    if not os.path.isdir(frame_dir):
        os.makedirs(frame_dir)
    
    ori_video = cv2.VideoCapture(vopath)
    total_frames= int(ori_video.get(7))
    # print total_frames;exit()
    
    for i in range(1, total_frames+1):
        imagename = "frame%s.%s" % (str(i), format)
        frameimagePath = os.path.join(frame_dir, imagename)
        
        status, frame = ori_video.read()
        if not status:
            print "Read frame failed."
            break
        cv2.imwrite(frameimagePath, frame)

videoname = "videoSRC001_1920x1080_30_qp_00.avi"
videopath = os.path.join(video_dir, videoname)
frame_dir = os.path.join(output_dir, videoname.split('.')[0])
slice_video(videopath, frame_dir)