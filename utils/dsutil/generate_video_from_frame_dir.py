####sss###uu###uu###nnnn#####nnnn####yy##yyy###cccc##iii####aaa#####
##sss#####uu###uu##nn###nn##nn###nn###y#yyy##cc######iii###a##aa####
#####sss##uu###uu##nn###nn##nn###nn####yy####cc######iii##aaaaaaa###
##sss######uuuu####nn###nn##nn###nn####yy######cccc##iii#aa####aaa##
import cv2
import os
import cPickle

voinfo_path = "/home/sunnycia/pwd/saliency_on_videoset/videoinfo.pkl"
voinfo = cPickle.load(open(voinfo_path, "rb"))

# frame_basedir = "/data/sunnycia/ImageSet/uncertainty"
frame_basedir = "/data/sunnycia/pwd/saliency_on_videoset/_Saliencymap/MSFUSION"
voname = "videoSRC035"
frame_dir = os.path.join(frame_basedir, voname)
# vo_dir = "/data/sunnycia/ImageSet/weighted_spatial"
vo_dir = frame_basedir
if not os.path.isdir(vo_dir):
    os.makedirs(vo_dir)

def sortframe(framename):
    name = framename.split('.')[0]
    idx_str = name.replace('frame', '')
    idx = int(idx_str)
    return idx
    
resolution = (1920, 1080)
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
fps = voinfo[voname]['fps']
salvoname = voname+".avi"
salvopath = os.path.join(vo_dir, salvoname)
frame_list = os.listdir(frame_dir)
frame_list.sort(key=sortframe)
out_video = cv2.VideoWriter(salvopath, fourcc, fps, resolution)
for framename in frame_list:
    frame_path = os.path.join(frame_dir, framename)
    frame = cv2.imread(frame_path)
    out_video.write(frame)
out_video.release()
print salvopath, "Done!"