import os, glob
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--basedir', type=str, required=True)
parser.add_argument('--type', type=str, required=True, help="toone or tomany")
args = parser.parse_args()

base_dir = args.basedir
## check if base_dir exists
if not os.path.isdir(base_dir):
    print base_dir, "not exists. Exit"
    exit()
convert_type = args.type
if convert_type == 'toone':
    video_dir_list = os.listdir(base_dir)
    for video_dir in video_dir_list:
        image_path_list = glob.glob(os.path.join(base_dir, video_dir, "*.*"))
        if len(image_path_list) == 0:
            print "Type error. exit"
            exit()
        for image_path in image_path_list:
            image_basename = os.path.basename(image_path)
            new_basename = video_dir+'_'+image_basename
            new_path = os.path.join(base_dir, new_basename)
            shutil.move(image_path, new_path)
        shutil.rmtree(os.path.join(base_dir, video_dir))
elif convert_type == 'tomany':
    image_path_list = glob.glob(os.path.join(base_dir, "*.*"))
    if len(image_path_list) == 0:
        print "Type error. exit"
        exit()
    for image_path in image_path_list:
        imagename = os.path.basename(image_path)
        video_dir = imagename.split('_')[0]
        cur_dir = os.path.join(base_dir, video_dir)
        if not os.path.isdir(cur_dir):
            os.makedirs(cur_dir)
        newname = imagename.replace(video_dir+'_', '')
        new_path = os.path.join(cur_dir, newname)
        shutil.move(image_path, new_path)