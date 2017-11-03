import caffe
import numpy as np, cv2
class Saliencynet:

    def __init__(self, deploy_proto, caffe_model):
        self.net = caffe.Net(deploy_proto, caffe_model, caffe.TEST) 
        self.MEAN_VALUE = np.array([103.939, 116.779, 123.68])   # BGR
        self.MEAN_VALUE = MEAN_VALUE[:,None, None]
        self.std_wid = 480
        self.std_hei = 288

    def preprocess_image(self, img_arr, sub_mean=True):
        img_arr = img_arr.astype(np.float32)
        h, w, c = img_arr.shape
        # subtract mean
        if sub_mean:
            img_arr[:, :, 0] -= MEAN_VALUE[0] # B
            img_arr[:, :, 1] -= MEAN_VALUE[1] # G
            img_arr[:, :, 2] -= MEAN_VALUE[2] # R

        ## this version simply resize the test image
        img_arr = cv2.resize(img_arr, dsize=(self.std_wid, self.std_hei))
        # put channel dimension first
        img_arr = np.transpose(img_arr, (2,0,1))
        img_arr = img_arr[None, :]
        img_arr = img_arr / 255. # convert to float precision
        return img_arr
    
    def compute_saliency(self, image_path):
        img_arr = cv2.imread(image_path)
        self.h, self.w, self.c = img_arr.shape # store the image's original height and width
        img_arr = self.preprocess_image(img_arr, False)
        # print img_arr.shape
        assert img_arr.shape == (1, 3, self.std_hei, self.std_wid)

        self.net.blobs['data'].data[...] = img_arr
        self.net.forward()
        sal_map = self.net.blobs['saliency_map'].data
        sal_map = sal_map[0,0,:,:]
        saliency_map = self.postprocess_saliency_map(sal_map)
        return saliency_map
    
    def postprocess_saliency_map(self, sal_map):
        sal_map = sal_map - np.amin(sal_map)
        sal_map = sal_map / np.amax(sal_map)

        sal_map *= 255
        sal_map = cv2.resize(sal_map, dsize=(self.w, self.h))
        return sal_map