from Flownet import Flownet
import os, imageio
import caffe
import numpy as np, cv2

class ImageSaliencyNet:
    def __init__(self, deploy_proto, caffe_model):
        self.net = caffe.Net(deploy_proto, caffe_model, caffe.TEST) 
        self.MEAN_VALUE = np.array([103.939, 116.779, 123.68])   # BGR
        self.MEAN_VALUE = self.MEAN_VALUE[:,None, None]
        self.std_wid = 480
        self.std_hei = 288

    def preprocess_image(self, img_arr, sub_mean=True):
        img_arr = img_arr.astype(np.float32)
        h, w, c = img_arr.shape
        # subtract mean
        if sub_mean:
            img_arr[:, :, 0] -= self.MEAN_VALUE[0] # B
            img_arr[:, :, 1] -= self.MEAN_VALUE[1] # G
            img_arr[:, :, 2] -= self.MEAN_VALUE[2] # R

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

class FlowbasedVideoSaliencyNet:
    def __init__(self, flownet_deploy_proto, flownet_caffe_model, video_deploy_proto, video_caffe_model, key_frame_interval=10):
        self.std_wid = 480
        self.std_hei = 288
        self.key_frame_interval = key_frame_interval
        self.flownet = Flownet(flownet_deploy_proto, flownet_caffe_model, (self.std_wid, self.std_hei))
        # self.flow_net = caffe.Net(flownet_deploy_proto, flownet_caffe_model, caffe.TEST) 
        self.video_net = caffe.Net(video_deploy_proto, video_caffe_model, caffe.TEST)
        self.MEAN_VALUE = np.array([103.939, 116.779, 123.68])   # BGR
        self.MEAN_VALUE = self.MEAN_VALUE[:,None, None]

    def preprocess_image(self, img_arr, sub_mean=True):
        img_arr = img_arr.astype(np.float32)
        h, w, c = img_arr.shape
        # subtract mean
        if sub_mean:
            img_arr[:, :, 0] -= self.MEAN_VALUE[0] # B
            img_arr[:, :, 1] -= self.MEAN_VALUE[1] # G
            img_arr[:, :, 2] -= self.MEAN_VALUE[2] # R

        ## this version simply resize the test image
        img_arr = cv2.resize(img_arr, dsize=(self.std_wid, self.std_hei))
        # put channel dimension first
        img_arr = np.transpose(img_arr, (2,0,1))
        img_arr = img_arr[None, :]
        img_arr = img_arr / 255. # convert to float precision
        return img_arr


    def postprocess_saliency_map(self, sal_map):
        sal_map = sal_map - np.amin(sal_map)
        sal_map = sal_map / np.amax(sal_map)
        sal_map *= 255
        sal_map = cv2.resize(sal_map, dsize=(self.std_wid, self.std_hei))
        return sal_map
    '''
    def compute_key_frame_saliency(self, processed_frame_arr):
        img_arr = cv2.imread(image_path)
        self.h, self.w, self.c = img_arr.shape # store the image's original height and width
        img_arr = self.preprocess_image(img_arr, False)

        # print img_arr.shape
        assert img_arr.shape == (1, 3, self.std_hei, self.std_wid)

        self.net_image.blobs['data'].data[...] = img_arr
        self.net_image.forward()
        sal_map = self.net_image.blobs['saliency_map'].data
        sal_map = sal_map[0,0,:,:]
        saliency_map = self.postprocess_saliency_maps(sal_map)
        return saliency_map
    '''

    def compute_frame_saliency(self, processd_key_frame_arr, processed_non_key_frame_arr):
        pass
        
    
    def setup_video(self, video_path):
        if not os.path.isfile(video_path):
            print video_path, "not exists, abort."
            return
        print "Setting up", video_path
        video_reader = imageio.get_reader(video_path)
        self.video_meta_data = video_reader.get_meta_data()
        self.frames = []
        for frame_idx, frame in enumerate(video_reader):
            self.frames.append(cv2.resize(frame, dsize=(self.std_wid, self.std_hei)))

    def create_saliency_video(self):
        if self.frames is None:
            print "setup video first!"
            return
        index_in_key=0
        self.predictions = []
        for i in range(len(self.frames)):
            if i % self.key_frame_interval == 0:
                key_frame_index=i
            cur_frame_index = i
            key_frame = self.frames[key_frame_index]
            cur_frame = self.frames[cur_frame_index]

            print "Processing frame",key_frame_index, "and frame", cur_frame_index

            optical_flow = self.flownet.get_optical_flow(key_frame, cur_frame)

            self.video_net.blobs['data'].data[...] = np.transpose(key_frame, (2,0,1))[None, ...]
            self.video_net.blobs['flow'].data[...] = np.transpose(optical_flow, (2, 0, 1))[None, ...]
            self.video_net.forward()
            sal_map = self.video_net.blobs['saliency_map'].data
            sal_map = sal_map[0,0,:,:]
            saliency_map = self.postprocess_saliency_map(sal_map)
            self.predictions.append(saliency_map)
        
    def dump_predictions_as_video(self, output_path, fps):
        if self.predictions is None:
            print "create video saliency first"
            return
        video_writer = imageio.get_writer(output_path, fps=fps)

        for saliency_map in self.predictions:
            video_writer.append_data(saliency_map)
        video_writer.close()


    def dump_predictions_as_images(self, output_directory):
        if self.predictions is None:
            print "create video saliency first"
            return
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)

        index = 1
        for saliency_map in self.predictions:
            outputname = "frame_%d.jpg" % index
            output_path = os.path.join(output_directory, outputname)
        print "Done for", output_directory
