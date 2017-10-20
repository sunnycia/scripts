import numpy as np
import os
import cPickle
import cv2

# (480, 270)
downsample_rate = 4
MEAN_VALUE = np.array([103.939, 116.779, 123.68], dtype=np.float32)   # B G R/ use opensalicon's mean_value

frame_basedir = '/data/sunnycia/ImageSet/frame'
density_basedir = '/data/sunnycia/ImageSet/density-6'

videoidx_list = os.listdir(frame_basedir)
videoidx_list.sort()

def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0
    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg
    return out

##generate frame dataset
frame_list = []
print 'Generating frame_list...'
for videoidx in videoidx_list:
    video_dir = os.path.join(frame_basedir, videoidx)
    imagenames = os.listdir(video_dir)
    imagenames.sort()
    for imagename in imagenames:
        imagepath = os.path.join(video_dir, imagename)
        img_arr = cv2.imread(imagepath).astype(np.float32)
        h, w, c = img_arr.shape
        # sub mean value
        img_arr[:, :, 0] -= MEAN_VALUE[0] # B
        img_arr[:, :, 1] -= MEAN_VALUE[1] # G
        img_arr[:, :, 2] -= MEAN_VALUE[2] # R
        # down sample
        img_arr = cv2.resize(img_arr, dsize=(w/downsample_rate, h/downsample_rate))
        # make border, convert (270, 480) to (288, 480)
        img_arr = cv2.copyMakeBorder(img_arr,9,9,0,0,cv2.BORDER_REPLICATE)
        # switch channel, H, W, C-> C, H, W (3, 270, 480)
        img_arr = np.transpose(img_arr, (2, 0, 1))
        # normalize to 0~1
        img_arr = img_arr/255.
        # float32
        # img_arr = img_arr.astype(np.float32)
        # (3, 270, 480) 1.0 float32
        # print img_arr.shape, img_arr.max(),img_arr.dtype;exit()
        frame_list.append(img_arr)
    print "\tDone for", videoidx
print 'Done for frame'

##generate density dataset
density_list = []
print 'Generating density_list...'
for videoidx in videoidx_list:
    video_dir = os.path.join(density_basedir, videoidx)
    imagenames = os.listdir(video_dir)
    imagenames.sort()
    for imagename in imagenames:
        imagepath = os.path.join(video_dir, imagename)
        img_arr = cv2.imread(imagepath, 0).astype(np.float32)
        h, w= img_arr.shape
        # down sample
        img_arr = cv2.resize(img_arr, dsize=(w/downsample_rate, h/downsample_rate))
        # make border, convert (270, 480) to (288, 480)
        img_arr = cv2.copyMakeBorder(img_arr,9,9,0,0,cv2.BORDER_REPLICATE)
        # switch channel
        img_arr = img_arr[None, :]
        # normalize to 0~1
        img_arr = img_arr / 255.
        # float32
        # img_arr = img_arr.astype(np.float32)
        # (1, 270, 480) 1.0 float32
        # print img_arr.shape, img_arr.max(),img_arr.dtype;exit()
        density_list.append(img_arr)
    print "\tDone for", videoidx
print 'Done for density'

## generate mini dataset for debugging
print "Dumping mini set"
cPickle.dump(frame_list[:1000], open('../dataset/frame_mini.pkl', 'wb'))
cPickle.dump(density_list[:1000], open('../dataset/density_mini.pkl', 'wb'))

##generate frame dataset
assert len(frame_list) == len(density_list)
print "Dumping..."
cPickle.dump(frame_list, open('../dataset/frames.pkl', 'wb'))
cPickle.dump(density_list, open('../dataset/densitys.pkl', 'wb'))

'''
density_list = []
frame_list = []

for videoidx in videoidx_list:
    density_dir = os.path.join(density_basedir, videoidx)        
    frame_dir = os.path.join(frame_basedir, videoidx)
    
    imagenames = os.listdir(frame_dir)
    for imagename in imagenames:
        framepath = os.path.join(frame_dir, imagename)
        densitypath = os.path.join(density_dir, imagename)

        ######frame
        frame_arr = cv2.imread(framepath)
        # switch channel, H, W, C-> C, H, W (3, 270, 480)
        frame_arr = np.transpose(frame_arr, (2, 0, 1))
        #
        c, h, w = frame_arr.shape
        # down sample to (270, 480, 3)
        frame_arr = cv2.resize(frame_arr, dsize=(w/downsample_rate, h/downsample_rate))
        # sub MEAN_VALUE
        print frame_arr[0, 0, 0];
        frame_arr = frame_arr - MEAN_VALUE
        frame_arr[:, :, 0] -= MEAN_VALUE[0]
        frame_arr[:, :, 1] -= MEAN_VALUE[1]
        frame_arr[:, :, 2] -= MEAN_VALUE[2]
        print frame_arr[0, 0, 0];exit()
        # normalize to 0~1
        frame_arr = frame_arr/255.
        # float32
        frame_arr = frame_arr.astype(np.float32)
        # (3, 270, 480) 1.0 float32
        # print frame_arr.shape, frame_arr.max(),frame_arr.dtype;exit()
        frame_list.append(frame_arr)

        ######density
        density_arr = cv2.imread(densitypath, 0)
        h, w= density_arr.shape
        # down sample 
        density_arr = cv2.resize(density_arr, dsize=(w/downsample_rate, h/downsample_rate))
        # switch channel
        density_arr = density_arr[None, :]
        # normalize to 0~1
        density_arr = density_arr / 255.
        # float32
        density_arr = density_arr.astype(np.float32)
        # (1, 270, 480) 1.0 float32
        # print density_arr.shape, density_arr.max(),density_arr.dtype;exit()
        density_list.append(density_arr)
    print "\tDone for", videoidx
print 'Done for density'
'''