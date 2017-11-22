import cv2
import os
import cPickle
####sss###uu###uu###nnnn#####nnnn####yy##yyy###cccc##iii####aaa#####
##sss#####uu###uu##nn###nn##nn###nn###y#yyy##cc######iii###a##aa####
#####sss##uu###uu##nn###nn##nn###nn####yy####cc######iii##aaaaaaa###
##sss######uuuu####nn###nn##nn###nn####yy######cccc##iii#aa####aaa##

#####################################################################
######################SUNNYCIA#######################################
####################################CHEATCODE#######################$
## voname_prefix ---> same with listdir
## voname_postfix
## voname
## salvoname
##
##
##
##
##
##
##

voinfo_path = "/home/sunnycia/pwd/saliency_on_videoset/videoinfo.pkl"
voinfo = cPickle.load(open(voinfo_path, "rb"))
# for vo in voinfo:
    # print vo;
# exit()

# model = "ISEEL"
# saliency_dir = "/data/sunnycia/pwd/saliency_on_videoset/_Saliencymap/%s" % model
# salvo_dir = "/data/sunnycia/pwd/saliency_on_videoset/_Saliencyvo/%s" % model

saliency_dir = '/data/sunnycia/ImageSet/uncertainty';
salvo_dir = '/data/sunnycia/ImageSet/uncertainty_vo';
if not os.path.isdir(salvo_dir):
    os.makedirs(salvo_dir)

voname_prefix_list = os.listdir(saliency_dir)


def sortframe(framename):
    name = framename.split('.')[0]
    idx_str = name.replace('frame', '')
    idx = int(idx_str)
    return idx
    
resolution = (1920, 1080)
# voname_postfix = '_1920x1080_30_qp_00.avi'
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')

for voname in voname_prefix_list:
    print "Handling", voname,
    # voname = voname_prefix+voname_postfix
    fps = voinfo[voname]['fps'];

    frame_dir = os.path.join(saliency_dir, voname)
    salvoname = voname + ".avi"
    salvopath = os.path.join(salvo_dir, salvoname)

    ## check if salvo exist
    if os.path.isfile(salvopath):
        print "Already exist."
        continue
    
    frame_list = os.listdir(frame_dir)
    frame_list.sort(key=sortframe)
    out_video = cv2.VideoWriter(salvopath,fourcc, fps, resolution)
    for framename in frame_list:
        frame_path = os.path.join(frame_dir, framename)
        frame = cv2.imread(frame_path)
        # print frame.mean()
        out_video.write(frame)

    out_video.release()
    print salvopath, "DONE"
    