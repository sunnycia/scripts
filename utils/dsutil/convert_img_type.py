import cv2
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--base', type=str, required=True)
parser.add_argument('--new_ext', type=str, default='.jpg')

args = parser.parse_args()
base = args.base
new_ext = args.new_ext

for root, dirs, files in os.walk(base):
    path = root.split(os.sep)
    # print((len(path) - 1) * '---', os.path.basename(root))


    for file in files:
        # print os.path.join(root, dirs, file)
        img_path = os.path.join(root, file)
        filename, file_extension = os.path.splitext(img_path)
        if file_extension==new_ext:
            continue
        new_image_name = os.path.basename(img_path).split('.')[0]+new_ext
        if 'frame_' in new_image_name:
            pass
        else:
            new_image_name = new_image_name.replace('frame', 'frame_')
        new_path = os.path.join(os.path.dirname(img_path), new_image_name)
        # print new_path;exit()
        cv2.imwrite(new_path, cv2.imread(img_path))
        os.remove(img_path)

    # for dire in dirs:
    #     print os.path.join(root, dire)
        # print(len(path) * '---', file)