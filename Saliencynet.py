from math import ceil
from Flownet import Flownet
import os, imageio
import caffe
import numpy as np, cv2
import time

class SaliencyNet:
    pass

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

class FramestackbasedVideoSaliencyNet:
    def __init__(self, video_deploy_proto, video_caffe_model, video_length=5,bgr_mean_value=[103.939, 116.779, 123.68], infer_type='slice'):
        self.std_wid = 480
        self.std_hei = 288
        self.video_length=video_length
        self.video_net = caffe.Net(video_deploy_proto, video_caffe_model, caffe.TEST)
        self.BGR_MEAN_VALUE = np.array(bgr_mean_value)   # BGR
        self.BGR_MEAN_VALUE = self.BGR_MEAN_VALUE[None, None, ...]
        self.infer_type = infer_type

    def preprocess_image(self, img_arr):
        img_arr = img_arr.astype(np.float32)
        h, w, c = img_arr.shape
        # subtract mean
        img_arr = img_arr - self.BGR_MEAN_VALUE

        ## this version simply resize the test image
        img_arr = cv2.resize(img_arr, dsize=(self.std_wid, self.std_hei))
        # put channel dimension first
        img_arr = np.transpose(img_arr, (2,0,1))
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

    def setup_video(self, video_path): ## slice or slide
        if not os.path.isfile(video_path):
            print video_path, "not exists, abort."
            return
        print "Setting up", video_path
        try_time = 5
        video_reader = ''
        for i in range(try_time):
            try:    
                video_reader = imageio.get_reader(video_path)
            except:
                print "Catch exception, retry..."
                time.sleep(0.5)
            if not video_reader == '':
                break
        self.video_meta_data = video_reader.get_meta_data()
        # print self.video_meta_data;exit()
        self.frames = []
        for frame_idx, frame in enumerate(video_reader):
            # self.frames.append(cv2.resize(frame, dsize=(self.std_wid, self.std_hei)))
            self.frames.append(self.preprocess_image(frame))
        if self.infer_type=='slide':
            prefix_frames = [self.frames[0] for i in range(self.video_length)]
            self.frames = prefix_frames + self.frames
        # self.ori_size = 

    def create_saliency_video(self):
        if self.frames is None:
            print "setup video first!"
            return
        self.predictions = []
        if self.infer_type=='slice':
            step = self.video_length
        elif self.infer_type=='slide':
            step = 1
        for i in range(0, len(self.frames)-1, step):
            if i + self.video_length > len(self.frames)-1:
                if self.infer_type=='slice':
                    resd_frame_num = len(self.frames)-(i)
                    frame_stack = self.frames[len(self.frames)-1-self.video_length:len(self.frames)-1]
                    raw_predictions = self.predict_frame_stack(frame_stack)
                    tmp_predictions = []
                    for raw_prediction in raw_predictions:
                        prediction = self.postprocess_saliency_map(raw_prediction)
                        tmp_predictions.append(prediction)
                    for i in range(resd_frame_num):
                        index = self.video_length-1-(resd_frame_num-1-i)
                        print index
                        self.predictions.append(tmp_predictions[index])
                break
            frame_stack = []
            print "Processing",
            for j in range(i, i + self.video_length):
                print j,
                frame_stack.append(self.frames[j])
            print ""
            # frame_stack = np.dstack(frame_stack)
            # self.video_net.blobs['data'].data[...] = np.transpose(frame_stack, (2,0,1))[None, ...]
            # self.video_net.forward()
            # sal_maps = self.video_net.blobs['saliency_map_stack'].data[0, ...]
            raw_predictions = self.predict_frame_stack(frame_stack)
            for raw_prediction in raw_predictions:
                prediction = self.postprocess_saliency_map(raw_prediction)
                self.predictions.append(prediction)

        print len(self.predictions),self.video_meta_data['nframes'], len(self.frames)
        assert len(self.predictions) == self.video_meta_data['nframes']

    def predict_frame_stack(self, frame_stack):
        if self.infer_type=='slice':
            output_blob_name= 'saliency_map_stack'
        elif self.infer_type=='slide':
            output_blob_name='saliency_map'
        print frame_stack[0].shape, frame_stack[1].shape
        frame_stack = np.vstack(frame_stack)
        self.video_net.blobs['data'].data[...] = frame_stack
        self.video_net.forward()
        return self.video_net.blobs[output_blob_name].data[0, ...]

    def dump_predictions_as_video(self , output_path, fps):
        if self.predictions is None:
            print "create video saliency first"
            return
        video_writer = imageio.get_writer(output_path, fps=fps)

        for saliency_map in self.predictions:
            video_writer.append_data(saliency_map)
        video_writer.close()

    def dump_predictions_as_images(self, output_directory, video_name, allinone):
        if self.predictions is None:
            print "create video saliency first"
            return
        if not allinone:
            output_directory = os.path.join(output_directory, video_name)

        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)

        index = 1
        for saliency_map in self.predictions:
            if not allinone:
                outputname = "frame_%d.jpg" % index
            else:
                outputname = "%s_frame_%d.jpg" % (video_name,index)
            output_path = os.path.join(output_directory, outputname)
            print "save to", output_path
            cv2.imwrite(output_path, saliency_map)
            index += 1
        print "Done for", output_directory

class C3DbasedVideoSaliencyNet:
    def __init__(self, video_deploy_proto, video_caffe_model, video_length, video_size, rgb_mean_value):
        self.std_wid = video_size[0]
        self.std_hei = video_size[1]
        self.video_length = video_length
        self.RGB_MEAN_VALUE = np.array(rgb_mean_value)[None,None, ...]
        self.video_net = caffe.Net(video_deploy_proto, video_caffe_model, caffe.TEST)

    def setup_video(self,video_path):
        if not os.path.isfile(video_path):
            print video_path, "not exists, abort."
            return
        print "Setting up", video_path
        try_time = 5
        video_reader = ''
        for i in range(try_time):
            try:    
                video_reader = imageio.get_reader(video_path)
            except:
                print "Catch exception, retry..."
                time.sleep(0.5)
            if not video_reader == '':
                break
        self.video_meta_data = video_reader.get_meta_data()
        # print self.video_meta_data;exit()
        self.frames = []
        for frame_idx, frame in enumerate(video_reader):
            self.frames.append(self.preprocess_image(frame, channel_order='rgb'))

    def preprocess_image(self,image, channel_order='bgr'):
        image = cv2.resize(image, dsize=(self.std_wid, self.std_hei))
        image = image - self.RGB_MEAN_VALUE
        image = image/255.
        if channel_order == 'rgb':
            image= image[:, :, ::-1]
        return image
    def postprocess_saliency(self, raw_prediction, threshold, gaussian=True):
        # threshold is between 1 to 0
        raw_prediction = (raw_prediction - raw_prediction.min())/raw_prediction.max()
        prediction = raw_prediction*255

        # ret1,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
        ##threshold image result
        scaled_threshold = threshold * 255
        prediction[np.where(prediction < threshold)] = 0

        if gaussian==True:
            ## do gaussian blur to smooth the thresholded contour
            prediction = cv2.GaussianBlur(prediction,(5,5),0)
            # pass
        prediction = cv2.resize(prediction, dsize=self.video_meta_data['size'])
        return prediction

    def create_saliency_video(self, threshold):
        ## generate tuple list first
        self.prediction_list = []
        for i in range(0, len(self.frames), self.video_length):
            if i + self.video_length >= len(self.frames):
                end_frame_index = len(self.frames)-1;start_frame_index = end_frame_index - self.video_length+1
                input_data = np.transpose(np.array(self.frames[start_frame_index:end_frame_index+1])[None, ...], (0,4,1,2,3))
                print input_data.shape;
                self.video_net.blobs['data'].data[...] = input_data
                self.video_net.forward()
                raw_prediction_list = self.video_net.blobs['output'].data[0, 0,...]
                # output = flat_output
                assert len(raw_prediction_list) == self.video_length
                # print i
                for j in range(len(self.frames)-i, self.video_length):
                    # print j
                    prediction = self.postprocess_saliency(raw_prediction_list[j], threshold)
                    self.prediction_list.append(prediction)      
                break
            start_frame_index = i;end_frame_index = i + self.video_length - 1
            input_data = np.transpose(np.array(self.frames[start_frame_index:end_frame_index+1])[None, ...], (0,4,1,2,3))
            # print input_data
            print input_data.shape;
            self.video_net.blobs['data'].data[...] = input_data
            self.video_net.forward()
            raw_prediction_list = self.video_net.blobs['output'].data[0,0,...]
            # output = flat_output
            assert len(raw_prediction_list) == self.video_length, str(len(raw_prediction_list))+'is not equal to '+str(self.video_length)
            for raw_prediction in raw_prediction_list:
                self.prediction_list.append(self.postprocess_saliency(raw_prediction, threshold))
        assert len(self.prediction_list) == len(self.frames),"Prediction not complete."+str(len(self.prediction_list))+' not equal to '+str(len(self.frames))

    def dump_predictions_as_video(self,output_path, fps):
        if self.prediction_list is None:
            print "create video saliency first"
            return
        video_writer = imageio.get_writer(output_path, fps=fps)

        for saliency_map in self.prediction_list:
            # print saliency_map.shape;
            video_writer.append_data(saliency_map)
        video_writer.close()

    def dum_predictions_as_images(self,output_directory, video_name, allinone):
        pass

class VoxelbasedVideoSaliencyNet:
    def __init__(self, deploy_proto, caffe_model, video_size, video_length, mean_list, infer_type):
        self.std_wid=video_size[0];self.std_hei=video_size[1];self.video_length=video_length

        self.video_net=caffe.Net(deploy_proto, caffe_model, caffe.TEST)
        self.MEAN_VALUE = np.array(mean_list)[None, None, ...] # h,w,c 
        self.infer_type=infer_type
        if not os.path.isfile(deploy_proto):
            print "Cannot find", deploy_proto
        if not os.path.isfile(caffe_model):
            print "Cannot find", caffe_model

    def preprocess_image(self, img_arr):
        ## img_arr is in bgr color channel sort
        img_arr = img_arr.astype(np.float32)
        h, w, c = img_arr.shape
        # subtract mean
        img_arr = img_arr - self.MEAN_VALUE
        img_arr = cv2.resize(img_arr, dsize=(self.std_wid, self.std_hei))
        img_arr = np.transpose(img_arr, (2,0,1)) # put channel dimension first
        img_arr = img_arr / 255. # normalization to 1
        # print img_arr.shape
        return img_arr    
    # def postprocess_saliency_map(self, sal_map):
    #     sal_map = sal_map - np.amin(sal_map);sal_map = sal_map / np.amax(sal_map);sal_map *= 255
    #     sal_map = cv2.resize(sal_map, dsize=self.video_meta_data['size'])
    #     return sal_map
    def postprocess_saliency_map(self, raw_prediction, threshold=0, gaussian=False, range_smooth=False):
        # threshold is between 1 to 0
        raw_prediction = (raw_prediction - raw_prediction.min())/raw_prediction.max()
        prediction = raw_prediction*255

        # ret1,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
        ##threshold image result
        scaled_threshold = int(threshold * 255)
        prediction[np.where(prediction < scaled_threshold)] = 0
        if gaussian==True:
            ## do gaussian blur to smooth the thresholded contour
            prediction = cv2.GaussianBlur(prediction,(5,5),0)
        if range_smooth==True:
            prediction = prediction - scaled_threshold

            prediction = prediction / prediction.max()
            prediction = prediction * 255
            prediction[np.where(prediction<0)]=0

            # pass
        prediction = cv2.resize(prediction, dsize=self.video_meta_data['size'])
        return prediction

    def setup_video(self, video_path):
        if not os.path.isfile(video_path):
            print video_path, "not exists, abort."
            return
        print "Setting up", video_path
        try_time = 5 

        video_capture = cv2.VideoCapture(video_path)
        video_size = (int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = int(ceil(video_capture.get(cv2.CAP_PROP_FPS)))
        frame_count = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
        self.video_meta_data = {
            'size': video_size,
            'fps': fps,
            'frame_count': frame_count
        }
        self.frames = []
        status, frame = video_capture.read()
        while status:
            self.frames.append(self.preprocess_image(frame))
            status, frame = video_capture.read()

        # video_reader = ''
        # for i in range(try_time):
        #     try:
        #         video_reader = imageio.get_reader(video_path)
        #     except:
        #         print "Catch exception, retry..."
        #         time.sleep(0.5)
        #     if not video_reader == '':
        #         break
        # self.video_meta_data = video_reader.get_meta_data()
        # print self.video_meta_data;exit()
        # self.frames = []
        # for frame_idx, frame in enumerate(video_reader):
        #     processed_frame = self.preprocess_image(frame)
        #     # print processed_frame.shape;
        #     self.frames.append(processed_frame)
        # print len(self.frames), self.frames[0]
        # if self.infer_type=='slide':
        #     prefix_frames = [self.frames[0] for i in range(self.video_length)]
        #     self.frames = prefix_frames + self.frames

    def create_saliency_video(self,overlap=15, threshold=0):
        self.prediction_list = []
        assert overlap < self.video_length, "overlap should not greater than video_length"
        step = self.video_length - overlap

        for i in range(0, len(self.frames), step):
            start_frame_index = i;end_frame_index = i + self.video_length - 1
            if end_frame_index > len(self.frames)-1:
                break
            print start_frame_index, end_frame_index
            cur_frame = np.array(self.frames[start_frame_index:end_frame_index+1])
            input_data = np.transpose(cur_frame[None, ...], (0, 2, 1, 3, 4))
            self.video_net.blobs['data'].data[...] = input_data
            self.video_net.forward()
            raw_prediction_list = self.video_net.blobs['predict'].data[0,0,...]
            # output = flat_output
            assert len(raw_prediction_list) == self.video_length, str(len(raw_prediction_list))+'is not equal to '+str(self.video_length)
            

            if i == 0:
                for raw_prediction in raw_prediction_list:
                    self.prediction_list.append(self.postprocess_saliency_map(raw_prediction, threshold))
            else:
                for k in range(self.video_length-step,self.video_length):
                    raw_prediction = raw_prediction_list[k]
                    self.prediction_list.append(self.postprocess_saliency_map(raw_prediction, threshold))

        assert len(self.prediction_list) == len(self.frames),"Prediction not complete."+str(len(self.prediction_list))+' not equal to '+str(len(self.frames))

    def dump_predictions_as_video(self, output_path, fps):
        if self.prediction_list is None:
            print "create video saliency first"
            return
        video_writer = imageio.get_writer(output_path, fps=fps)

        for saliency_map in self.prediction_list:
            video_writer.append_data(saliency_map)
        video_writer.close()
        
    def dump_predictions_as_images(self, output_directory, video_name, allinone):
        if self.prediction_list is None:
            print "create video saliency first"
            return
        if not allinone:
            output_directory = os.path.join(output_directory, video_name)

        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)

        index = 1
        for saliency_map in self.prediction_list:
            if not allinone:
                outputname = "frame_%d.jpg" % index
            else:
                outputname = "%s_frame_%d.jpg" % (video_name,index)
            output_path = os.path.join(output_directory, outputname)
            print "save to", output_path
            cv2.imwrite(output_path, saliency_map)
            index += 1
        print "Done for", output_directory