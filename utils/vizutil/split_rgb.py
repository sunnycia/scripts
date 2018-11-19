import numpy as np
import cv2
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--image_path', type=str, required=True)

args = parser.parse_args()

image_path = args.image_path
image_prefix = os.path.splitext(image_path)[0]
rgb_img = cv2.imread(image_path)

b,g,r = np.split(rgb_img, 3, axis=2)

cv2.imwrite(image_prefix+'_b.jpg', b)
cv2.imwrite(image_prefix+'_g.jpg', g)
cv2.imwrite(image_prefix+'_r.jpg', r)
