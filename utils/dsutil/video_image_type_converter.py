import os, glob
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--basedir', type=str, required=True)
parser.add_argument('--type', type=str, required=True, help="toone or tomany")
parser.add_argument('--group', type=bool, default=False, help="if many subfolder to handle")
args = parser.parse_args()

base_dir = args.basedir
## check if base_dir exists
if not os.path.isdir(base_dir):
    print base_dir, "not exists. Exit"
    exit()
convert_type = args.type

def many_to_one(basedir):
    video_dir_list = os.listdir(basedir)
    for video_dir in video_dir_list:
        image_path_list = glob.glob(os.path.join(basedir, video_dir, "*.*"))
        if len(image_path_list) == 0:
            print "Type error. exit"
            exit()
        for image_path in image_path_list:
            image_basename = os.path.basename(image_path)
            new_basename = video_dir+'_'+image_basename
            new_path = os.path.join(basedir, new_basename)
            shutil.move(image_path, new_path)
        shutil.rmtree(os.path.join(basedir, video_dir))

def group_many_to_one(basedir):
    subdir_list = os.listdir(base_dir)
    for subdir in subdir_list:
        working_dir = os.path.join(basedir, subdir)
        video_dir_list = os.listdir(working_dir)
        for video_dir in video_dir_list:
            image_path_list = glob.glob(os.path.join(working_dir, video_dir, "*.*"))
            if len(image_path_list) == 0:
                print "Type error. exit"
                exit()
            for image_path in image_path_list:
                image_basename = os.path.basename(image_path)
                new_basename = video_dir+'_'+image_basename
                new_path = os.path.join(working_dir, new_basename)
                shutil.move(image_path, new_path)
            shutil.rmtree(os.path.join(working_dir, video_dir))

def one_to_many(basedir):
    image_path_list = glob.glob(os.path.join(basedir, "*.*"))
    if len(image_path_list) == 0:
        print "Type error. exit"
        exit()
    for image_path in image_path_list:
        imagename = os.path.basename(image_path)
        video_dir = imagename.split('_')[0]
        cur_dir = os.path.join(basedir, video_dir)
        if not os.path.isdir(cur_dir):
            os.makedirs(cur_dir)
        newname = imagename.replace(video_dir+'_', '')
        newname = newname.replace('frame__', 'frame_')
        new_path = os.path.join(cur_dir, newname)
        shutil.move(image_path, new_path)

def group_one_to_many(basedir):
    subdir_list = os.listdir(basedir)
    for subdir in subdir_list:
        working_dir = os.path.join(basedir, subdir)
        image_path_list = glob.glob(os.path.join(working_dir, "*.*"))
        if len(image_path_list) == 0:
            print "Type error. exit"
            exit()
        for image_path in image_path_list:
            imagename = os.path.basename(image_path)
            video_dir = imagename.split('_')[0]
            cur_dir = os.path.join(working_dir, video_dir)
            if not os.path.isdir(cur_dir):
                os.makedirs(cur_dir)
            newname = imagename.replace(video_dir+'_', '')
            newname = newname.replace('frame__', 'frame_')
            new_path = os.path.join(cur_dir, newname)
            shutil.move(image_path, new_path)



if convert_type == 'toone':
    if not args.group:
        many_to_one(base_dir)
    else:
        group_many_to_one(base_dir)
elif convert_type == 'tomany':
    if not args.group:
        one_to_many(base_dir)
    else:
        group_one_to_many(base_dir)