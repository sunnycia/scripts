import os, glob, cv2, numpy as np

class StaticDataset():
    def __init__(self, frame_basedir, density_basedir, debug):
        MEAN_VALUE = np.array([103.939, 116.779, 123.68], dtype=np.float32)   # B G R/ use opensalicon's mean_value
        self.MEAN_VALUE = MEAN_VALUE[None, None, ...]
        self.img_size = (480, 288)
        frame_path_list = glob.glob(os.path.join(frame_basedir, '*.*'))
        density_path_list = glob.glob(os.path.join(density_basedir, '*.*'))
        if debug is True:
            print "Debug mode"
            frame_path_list = frame_path_list[:1000]
            density_path_list = density_path_list[:1000]

        self.data = []
        self.labels = []
        print len(frame_path_list)
        for (frame_path, density_path) in zip(frame_path_list, density_path_list):
            frame = cv2.imread(frame_path).astype(np.float32)
            density = cv2.imread(density_path, 0).astype(np.float32)

            frame =self.pre_process_img(frame, False)
            density = self.pre_process_img(density, True)
            self.data.append(frame)
            self.labels.append(density)
            if len(self.data) % 1 == 0:
                print len(self.data), '\r',
        print 'Done'
        self.num_examples = len(self.data)
        
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)
        
        self.completed_epoch = 0
        self.index_in_epoch = 0

    def pre_process_img(self, image, greyscale=False):
        if greyscale==False:
            image = image-self.MEAN_VALUE
            image = cv2.resize(image, dsize = self.img_size)
            image = np.transpose(image, (2, 0, 1))
            image = image / 255.
        else:
            image = cv2.resize(image, dsize = self.img_size)
            image = image[None, ...]
            image = image / 255.
        return image

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self.index_in_epoch
        self.index_in_epoch += batch_size
        if self.index_in_epoch > self.num_examples:
            # Finished epoch
            self.completed_epoch += 1
            # Shuffle the data
            perm = np.arange(self.num_examples)
            np.random.shuffle(perm)
            self.data = self.data[perm]
            self.labels = self.labels[perm]
            # Start next epoch
            start = 0
            self.index_in_epoch = batch_size
            assert batch_size <= self.num_examples
        end = self.index_in_epoch
        return self.data[start:end], self.labels[start:end]

class VideoDataset():
    def __init__(self, frame_basedir, density_basedir, flownet):
        MEAN_VALUE = np.array([103.939, 116.779, 123.68], dtype=np.float32)   # B G R/ use opensalicon's mean_value
        self.MEAN_VALUE = MEAN_VALUE[None, None, ...]
        self.img_size = (480, 288)
        self.dataset_dict={}
        self.T = 8
        # self.step = 4
        self.frame_basedir = frame_basedir
        self.density_basedir = density_basedir
        self.setup_video_dataset()

    def setup_video_dataset():
        video_subdir_list = os.listdir(self.frame_basedir)
        for video_subdir in video_subdir_list:
            video_name = video_subdir
            frame_count = len(os.listdir(os.path.join(self.frame_basedir, video_subdir)))
            video_dict={'frame_count'}

    def get_frame_pair():
        pass

    # def 