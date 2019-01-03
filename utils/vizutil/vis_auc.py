import os, glob
import argparse
import random 

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dsname', type=str, required=True, help='dataset name')
    parser.add_argument('--video_name', type=str, default='', help='video name in this dataset')
    parser.add_argument('--model_name', type=str, default='vo-v4-2-resnet-BNdeconv-l1loss-dropout-base_lr-0.01-snapshot-4000-resnet_addBN_2_deconv-batch-2_1529493349_snapshot-_iter_500000_threshold0_overlap0', help='video name in this dataset')
    
    return parser.parse_args()

parser = get_arguments()

dsname = parser.dsname
video_name = parser.video_name
model_name = parser.model_name

if dsname == 'vsg':
    frame_basedir='/data/SaliencyDataset/Video/VideoSet/ImageSet/Seperate/frames'
    fixation_basedir='/data/SaliencyDataset/Video/VideoSet/ImageSet/Seperate/fixation'
    density_basedir='/data/SaliencyDataset/Video/VideoSet/ImageSet/Seperate/density/sigma32'
    saliency_basedir=os.path.join('/data/SaliencyDataset/Video/VideoSet/Results/saliency_map', model_name)
else:
    raise NotImplementedError

check_list = [frame_basedir, fixation_basedir, density_basedir, saliency_basedir]


if video_name != '':
    for check_dir in check_list:
        if not os.path.isdir(os.path.join(check_dir, video_name)):
            print os.path.join(check_dir, video_name), 'not exists. abort.'
            exit()
else:
    # random an index
    sub_dir_list = os.listdir(frame_basedir)
    random_index = random.randint(0, len(sub_dir_list))
    video_name = sub_dir_list[random_index]
    for check_dir in check_list:
        if not os.path.isdir(os.path.join(check_dir, video_name)):
            print os.path.join(check_dir, video_name), 'not exists. abort.'
            exit()

frame_dir = os.path.join(frame_basedir, video_name)
fixation_dir = os.path.join(fixation_basedir, video_name)
density_dir = os.path.join(density_basedir, video_name)
saliency_dir = os.path.join(saliency_basedir, video_name)

frame_index = 15
frame_list = os.listdir(frame_dir)
# random_index = random.randint(0, len(frame_list))

frame_name = os.path.splitext(frame_list[frame_index])[0]+'*'
frame_path = glob.glob(os.path.join(frame_dir, frame_name))[0]
fixation_path = glob.glob(os.path.join(fixation_dir, frame_name))[0]
density_path = glob.glob(os.path.join(density_dir, frame_name))[0]
saliency_path = glob.glob(os.path.join(saliency_dir, frame_name))[0]

# cmd = 'matlab -nodisplay -nodesktop -r "frame_path=\'%s\';density_path=\'%s\';fixation_path=\'%s\';saliency_path=\'%s\';vis_auc;exit()"' % (frame_path,density_path,fixation_path, saliency_path)
cmd = 'matlab -r "frame_path=\'%s\';density_path=\'%s\';fixation_path=\'%s\';saliency_path=\'%s\';vis_auc;exit()"' % (frame_path,density_path,fixation_path, saliency_path)
os.system(cmd)
