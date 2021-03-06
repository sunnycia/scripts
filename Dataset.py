import os, glob, cv2, numpy as np
import random
from random import shuffle

# def get_frame_and_density_dir(dataset):
#     if dataset=='msu':
#         frame_basedir = '/data/sunnycia/SaliencyDataset/Video/MSU/frames'
#         density_basedir = '/data/sunnycia/SaliencyDataset/Video/MSU/density/sigma32'
#     elif dataset=='ledov':
#         frame_basedir = '/data/SaliencyDataset/Video/LEDOV/frames'
#         density_basedir = '/data/SaliencyDataset/Video/LEDOV/density/sigma32'
#     elif dataset=='hollywood':
#         frame_basedir = '/data/SaliencyDataset/Video/ActionInTheEye/Hollywood2/frames'
#         density_basedir = '/data/SaliencyDataset/Video/ActionInTheEye/Hollywood2/density'
#     elif dataset == 'videoset':
#         frame_basedir='/data/SaliencyDataset/Video/VideoSet/ImageSet/Seperate/frame'
#         density_basedir='/data/SaliencyDataset/Video/VideoSet/ImageSet/Seperate/density/sigma32'
#     else:
#         return None, None
#     return frame_basedir, density_basedir

class StaticDataset():
    def __init__(self, frame_basedir, density_basedir, debug, img_size=(480, 288), training_example_props=0.8):
        MEAN_VALUE = np.array([103.939, 116.779, 123.68], dtype=np.float32)   # B G R/ use opensalicon's mean_value
        self.MEAN_VALUE = MEAN_VALUE[None, None, ...]
        self.img_size = img_size
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
            # assert frame_path
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
    def __init__(self, frame_basedir, density_basedir, img_size=(112,112), video_length=16, stack=5, bgr_mean_list=[103.939, 116.779, 123.68],sort='bgr', size_before_crop=(128,128)):
        MEAN_VALUE = np.array(bgr_mean_list, dtype=np.float32)   # B G R/ use opensalicon's mean_value
        if sort=='rgb':
            MEAN_VALUE= MEAN_VALUE[::-1]
        self.MEAN_VALUE = MEAN_VALUE[None, None, ...]
        self.img_size = img_size
        self.size_before_crop = size_before_crop
        self.dataset_dict={}
        self.video_length = video_length
        self.step = 1
        # self.stack = stack
        assert self.step < self.video_length
        self.frame_basedir = frame_basedir
        self.density_basedir = density_basedir
        self.video_dir_list = glob.glob(os.path.join(self.frame_basedir, "*"))
        self.data_augmentation=False
        self.augmentation_list = []
        self.num_epoch = 0
        self.index_in_training_epoch = 0
        self.list_chunk=[]
        self.training_tuple_list = []
        self.validation_tuple_list = []
        # self.setup_video_dataset()

    def setup_video_dataset_flow(self):
        # self.video_dir_list = os.listdir(self.frame_basedir)
        # self.video_dir_list = glob.glob(os.path.join(self.frame_basedir, "*"))

        self.tuple_list = []

        for i in range(len(self.video_dir_list)):
            video_dir = self.video_dir_list[i]
            frame_list = glob.glob(os.path.join(video_dir, '*.*'))
            # print frame_list;exit()

            total_frame = len(frame_list)

            for j in range(total_frame):
                ceil = j + self.video_length
                if ceil > total_frame:
                    break
                ## random pick self.step frame in this interval
                pick_list = random.sample(range(j+2, ceil), self.step)
                for pick in pick_list:
                    assert pick < ceil and pick != j
                    tup = (i, j+1, pick) # video index, key frame index, current frame index
                    self.tuple_list.append(tup)
        
        shuffle(self.tuple_list)
        self.num_examples = len(self.tuple_list)
        print self.num_examples, "samples generated..."

        self.num_epoch = 0
        self.index_in_epoch = 0

    def setup_video_dataset_stack(self, random=False, overlap=0):
        self.tuple_list = []

        if random:
            pass
        else:
            for i in range(len(self.video_dir_list)):
                video_dir = self.video_dir_list[i]
                frame_list = glob.glob(os.path.join(video_dir, '*.*'))

                total_frame = len(frame_list)
                # print  self.video_length*overlap,self.video_length,overlap
                assert overlap < self.video_length, "overlap should smaller than videolength."
                step = self.video_length - overlap
                for j in range(0, total_frame, step):
                    ceil = j + self.video_length
                    if ceil > total_frame-1:
                        break
                    tup = tuple([k for k in range(j+1, j+1+self.video_length)])
                    self.tuple_list.append((i,tup)) #video index & frame stack index
                # print self.tuple_list

        shuffle(self.tuple_list)
        self.num_examples = len(self.tuple_list)
        print self.num_examples, "samples generated...";#exit()

        self.num_epoch = 0
        self.index_in_epoch = 0

    def chunkify(self, lst,n):
        return [lst[i::n] for i in xrange(n)]
    def get_training_validation_sample_from_chunk(self, fold_index):
        # self.list_chunk[fold_index] is the validation dataset
        if fold_index >= self.fold:
            print 'fold number not match list_chunk size'
            exit()
        train_list = []
        for i in range(self.fold):
            if i == fold_index:
                continue
            train_list = train_list + self.list_chunk[i]
            # train_list = [train_list self.list_chunk[i]] ## combine a training dataset
        test_list = self.list_chunk[fold_index]

        # return train_list, test_list
        self.training_tuple_list = train_list
        self.validation_tuple_list = test_list

        self.num_training_examples = len(self.training_tuple_list)

        self.num_validation_examples = len(self.validation_tuple_list)
        
        print self.num_examples, "samples generated in total,",self.num_training_examples,"training samples,",self.num_validation_examples,"validation samples";#exit()

        self.num_epoch = 0
        self.index_in_training_epoch = 0
        self.index_in_validation_epoch = 0

        # print len(self.training_tuple_list),len(self.validation_tuple_list);exit()
    def setup_video_dataset_c3d_with_fold(self, overlap=0, fold=10, skip_head=10): ## skip those bad data in the previous of a video
        # pass
        self.tuple_list = []
        self.fold=fold
        assert overlap < self.video_length, "overlap should smaller than videolength."
        step = self.video_length - overlap
        for i in range(len(self.video_dir_list)):
            video_dir = self.video_dir_list[i]
            frame_list = glob.glob(os.path.join(video_dir,'*.*'))
            total_frame = len(frame_list)
            
            for j in range(skip_head, total_frame, step): ## div 2, so 1/2 of the video_length is overlapped
                if j + self.video_length > total_frame:
                    break
                tup = (i,j) # video index and first frame index
                self.tuple_list.append(tup)
            # print self.tuple_list;exit()
        
        self.num_examples = len(self.tuple_list)
        shuffle(self.tuple_list)

        self.list_chunk = self.chunkify(self.tuple_list, fold)
        shuffle(self.list_chunk)
        # print self.list_chunk[0];exit()
        print "length of list_chunk:", len(self.list_chunk);#exit()

        # self.training_tuple_list, self.validation_tuple_list=get_training_validation_sample_from_chunk(self.fold_index)


    def setup_video_dataset_c3d(self, overlap=0, training_example_props=0.8, skip_head=10, debug=0): ## skip those bad data in the previous of a video
        # pass
        self.tuple_list = []
        assert overlap < self.video_length, "overlap should smaller than videolength."
        step = self.video_length - overlap

        if debug==1:
            total_sample = int(0.1*len(self.video_dir_list))
        else:
            total_sample=  len(self.video_dir_list)
        for i in range(total_sample):
            video_dir = self.video_dir_list[i]
            frame_list = glob.glob(os.path.join(video_dir,'*.*'))
            total_frame = len(frame_list)
            
            for j in range(skip_head, total_frame, step): ## div 2, so 1/2 of the video_length is overlapped
                if j + self.video_length > total_frame:
                    break
                tup = (i,j) # video index and first frame index
                self.tuple_list.append(tup)

        self.num_examples = len(self.tuple_list)
        shuffle(self.tuple_list)
        self.num_training_examples = int(self.num_examples * training_example_props)

        self.training_tuple_list = self.tuple_list[:self.num_training_examples]
        self.validation_tuple_list = self.tuple_list[self.num_training_examples:]
        self.num_validation_examples = len(self.validation_tuple_list)
        print self.num_examples, "samples generated in total,",self.num_training_examples,"training samples,",self.num_validation_examples,"validation samples";#exit()

        self.num_epoch = 0
        self.index_in_training_epoch = 0
        self.index_in_validation_epoch = 0

    def setup_video_dataset_connection_c3d(self):
        # self.tuple_list = []
        # assert overlap < self.video_length, "overlap should smaller than videolength."
        # step = self.video_length - overlap
        # for i in range(len(self.video_dir_list)):
        #     video_dir = self.video_dir_list[i]
        #     frame_list = glob.glob(os.path.join(video_dir,'*.*'))
        #     total_frame = len(frame_list)
            
        #     for j in range(skip_head, total_frame, step): ## div 2, so 1/2 of the video_length is overlapped
        #         if j + self.video_length > total_frame:
        #             break
        #         tup = (i,j) # video index and first frame index
        #         self.tuple_list.append(tup)
        #     # print self.tuple_list;exit()
        
        # self.num_examples = len(self.tuple_list)
        # shuffle(self.tuple_list)
        # self.num_training_examples = int(self.num_examples * training_example_props)

        # self.training_tuple_list = self.tuple_list[:self.num_training_examples]
        # self.validation_tuple_list = self.tuple_list[self.num_training_examples:]
        # self.num_validation_examples = len(self.validation_tuple_list)
        # print self.num_examples, "samples generated in total,",self.num_training_examples,"training samples,",self.num_validation_examples,"validation samples";#exit()

        # self.num_epoch = 0
        # self.index_in_training_epoch = 0
        # self.index_in_validation_epoch = 0
        pass

    def get_frame_pair(self):
        if not self.index_in_epoch >= self.num_examples:
            tup = self.tuple_list[self.index_in_epoch]

            self.index_in_epoch += 1
        else:
            print "One epoch finished, shuffling data..."

            shuffle(self.tuple_list)
            self.index_in_epoch = 0
            self.num_epoch += 1

            tup = self.tuple_list[self.index_in_epoch]
            self.index_in_epoch += 1
        # print tup;exit()
        video_index, key_frame_index, cur_frame_index = tup
        video_dir = self.video_dir_list[video_index]
        video_name = os.path.basename(video_dir)
        # print video_dir, key_frame_index, cur_frame_index
        key_frame_name_wildcard = "frame_%d.*" % key_frame_index
        cur_frame_name_wildcard = "frame_%d.*" % cur_frame_index
        key_frame_path = glob.glob(os.path.join(video_dir, key_frame_name_wildcard))[0]
        cur_frame_path = glob.glob(os.path.join(video_dir, cur_frame_name_wildcard))[0]

        gt_density_dir = os.path.join(self.density_basedir, video_name)
        # print gt_density_dir
        cur_frame_gt_path = glob.glob(os.path.join(gt_density_dir, cur_frame_name_wildcard))[0]
        # print video_name,video_dir, key_frame_path, cur_frame_path, cur_frame_gt_path;exit()

        key_frame = cv2.resize(cv2.imread(key_frame_path), dsize=self.img_size)
        cur_frame = cv2.resize(cv2.imread(cur_frame_path), dsize=self.img_size)
        cur_frame_gt = cv2.resize(cv2.imread(cur_frame_gt_path, 0), dsize=self.img_size)

        return key_frame, cur_frame, cur_frame_gt
    
    def get_frame_stack(self, version=1):
        if not self.index_in_epoch >= self.num_examples:
            tup = self.tuple_list[self.index_in_epoch]
            self.index_in_epoch += 1
        else:
            print "One epoch finished, shuffling data..."

            shuffle(self.tuple_list)
            self.index_in_epoch = 0
            self.num_epoch += 1
            tup = self.tuple_list[self.index_in_epoch]
            self.index_in_epoch += 1
        video_index, frame_stack_tuple = tup
        video_dir = self.video_dir_list[video_index]
        video_name = os.path.basename(video_dir)
        frame_wildcard = "frame_%d.*" 

        frame_stack = []
        density_stack = []
        for i in range(len(frame_stack_tuple)):
            index = frame_stack_tuple[i]
            # print index,os.path.join(video_dir, frame_wildcard % index)
            frame_path = glob.glob(os.path.join(video_dir, frame_wildcard % index))[0]
            density_path = glob.glob(os.path.join(self.density_basedir, video_name, frame_wildcard % index))[0]

            # frame = cv2.resize(cv2.imread(frame_path), dsize=self.img_size)
            # density = cv2.resize(cv2.imread(density_path, 0), dsize=self.img_size)
            
            frame = self.pre_process_img(cv2.imread(frame_path))
            density = self.pre_process_img(cv2.imread(density_path,0), True)

            frame_stack.append(frame)
            if version==1:
                density_stack.append(density)
            elif version==2:
                if i == len(frame_stack_tuple)-1:
                    density_stack.append(density)
        # print len(frame_stack), frame_stack[0].shape;#exit()
        return np.dstack(frame_stack), np.dstack(density_stack)

    def get_frame_connection_c3d(self, mini_batch=16, phase='training', density_length='full', data_augmentation=False):
        ## 
        ## density_length : full, half, one
        if phase == 'training':
            tuple_list = self.training_tuple_list
            index_in_epoch = self.index_in_training_epoch
            self.index_in_training_epoch += mini_batch
            num_examples = self.num_training_examples

        elif phase == 'validation':
            tuple_list = self.validation_tuple_list
            index_in_epoch = self.index_in_validation_epoch
            self.index_in_validation_epoch += mini_batch
            num_examples = self.num_validation_examples
        
        frame_wildcard = "frame_%d.*"
        if not index_in_epoch >= num_examples - mini_batch:
            tup_batch = tuple_list[index_in_epoch:index_in_epoch+mini_batch]
        else:
            if phase=='validation':
                self.index_in_validation_epoch = 0
                print "Done for validation"
                return None

            print "One epoch finished, shuffling data..."

            shuffle(self.training_tuple_list)
            self.index_in_training_epoch = 0
            self.num_epoch += 1
            tup_batch = self.training_tuple_list[self.index_in_training_epoch:self.index_in_training_epoch+mini_batch]
            self.index_in_training_epoch += mini_batch

        def rand_bools_int_func(n):
            r = random.getrandbits(n)
            return ( bool((r>>i)&1) for i in xrange(n) ) 
        
        if data_augmentation:
            self.data_augmentation=True
            rdn_list = []
            for rdn in rand_bools_int_func(2):
                rdn_list.append(rdn_list)

            if rdn_list[0]:
                self.augmentation_list.append('horizontal_flip')
            if rdn_list[1]:
                self.augmentation_list.append('crop')
            if 'crop' in self.augmentation_list:
                self.crop_pos=(random.randint(0, self.size_before_crop[0]-self.img_size[0]-1), random.randint(0, self.size_before_crop[1]-self.img_size[1]-1))
            ## randomly horizontal flip
            ## randomly crop
            ## randomly downsample
            ## randomly jitter
        else:
            self.data_augmentation=False
            self.augmentation_list=[]


        density_batch = []
        frame_batch = []
        ref_density_batch = []
        
        # print tup_batch, len(tup_batch)
        for tup in tup_batch:
            # print tup
            current_frame_list = []
            current_density_list = []
            current_ref_density_batch = []


            video_index, start_frame_index=tup
            end_frame_index = start_frame_index + self.video_length
            start_ref_index = start_frame_index - self.video_length/2
            end_ref_index = start_frame_index

            video_dir = self.video_dir_list[video_index]
            video_name = os.path.basename(video_dir)
            density_dir = os.path.join(self.density_basedir, video_name)

            for i in range(start_frame_index, end_frame_index):
                frame_index = i + 1
                frame_name = frame_wildcard % frame_index
                # print frame_name
                # print os.path.join(video_dir, frame_name)
                # print os.path.join(video_dir, frame_name)
                frame_path = glob.glob(os.path.join(video_dir, frame_name))[0]
                frame = self.pre_process_img(cv2.imread(frame_path),sort='rgb')
                current_frame_list.append(frame)


            if density_length=='full':
                for i in range(start_frame_index,end_frame_index):
                    frame_index = i + 1
                    frame_name = frame_wildcard % frame_index
                    # print os.path.join(density_dir, frame_name)
                    density_path = glob.glob(os.path.join(density_dir, frame_name))[0]
                    density = self.pre_process_img(cv2.imread(density_path, 0),greyscale=True)
                    # print density.shape;exit()
                    current_density_list.append(density)                
            elif density_length=='one':
                frame_index=end_frame_index
                frame_name = frame_wildcard%frame_index
                density = self.pre_process_img(cv2.imread(glob.glob(os.path.join(density_dir, frame_name))[0], 0), True)
                current_density_list.append(density)

            ## reference density batch
            if start_ref_index < 0:# out of boundary, set blank as reference
                ref_number = end_ref_index - start_ref_index + 1

                for i in range(ref_number):
                    current_ref_density_batch.append(np.zeros(self.img_size[0], self.img_size[1], 1))
            else:
                for i in range(start_ref_index, end_ref_index):
                    frame_index =i + 1
                    frame_name = frame_wildcard % frame_index
                    try:
                        density_path = glob.glob(os.path.join(density_dir, frame_name))[0]
                    except:
                        print os.path.join(density_dir, frame_name)
                    density = self.pre_process_img(cv2.imread(density_path, 0), greyscale=True)
                    current_ref_density_batch.append(density)

            frame_batch.append(np.array(current_frame_list))
            density_batch.append(np.array(current_density_list))
            ref_density_batch.append(np.array(current_ref_density_batch))
        # print np.array(frame_batch).shape,np.array(density_batch).shape;#(50, 16, 112, 112, 3)
        # print np.array(frame_batch).shape
        return np.transpose(np.array(frame_batch),(0,4,1,2,3)),np.transpose(np.array(density_batch),(0,4,1,2,3)), np.transpose(np.array(ref_density_batch), (0,4,1,2,3))


    def get_frame_c3d(self, mini_batch=16, phase='training', density_length='full', data_augmentation=False):
        ## 
        ## density_length : full, half, one
        if phase == 'training':
            tuple_list = self.training_tuple_list
            index_in_epoch = self.index_in_training_epoch
            self.index_in_training_epoch += mini_batch
            num_examples = self.num_training_examples

        elif phase == 'validation':
            tuple_list = self.validation_tuple_list
            index_in_epoch = self.index_in_validation_epoch
            self.index_in_validation_epoch += mini_batch
            num_examples = self.num_validation_examples
        
        frame_wildcard = "frame_%d.*"
        if not index_in_epoch >= num_examples - mini_batch:
            tup_batch = tuple_list[index_in_epoch:index_in_epoch+mini_batch]
        else:
            if phase=='validation':
                self.index_in_validation_epoch = 0
                print "Done for validation"
                return None

            print "One epoch finished, shuffling data..."

            shuffle(self.training_tuple_list)
            self.index_in_training_epoch = 0
            self.num_epoch += 1
            tup_batch = self.training_tuple_list[self.index_in_training_epoch:self.index_in_training_epoch+mini_batch]
            self.index_in_training_epoch += mini_batch

        def rand_bools_int_func(n):
            r = random.getrandbits(n)
            return ( bool((r>>i)&1) for i in xrange(n) ) 
        
        if data_augmentation:
            self.data_augmentation=True
            rdn_list = []
            for rdn in rand_bools_int_func(2):
                rdn_list.append(rdn_list)

            if rdn_list[0]:
                self.augmentation_list.append('horizontal_flip')
            if rdn_list[1]:
                self.augmentation_list.append('crop')
            if 'crop' in self.augmentation_list:
                self.crop_pos=(random.randint(0, self.size_before_crop[0]-self.img_size[0]-1), random.randint(0, self.size_before_crop[1]-self.img_size[1]-1))
            ## randomly horizontal flip
            ## randomly crop
            ## randomly downsample
            ## randomly jitter
        else:
            self.data_augmentation=False
            self.augmentation_list=[]


        density_batch = []
        frame_batch = []
        # print tup_batch, len(tup_batch)
        for tup in tup_batch:
            # print tup
            current_frame_list = []
            current_density_list = []

            video_index, start_frame_index=tup
            end_frame_index = start_frame_index + self.video_length
            video_dir = self.video_dir_list[video_index]
            video_name = os.path.basename(video_dir)
            density_dir = os.path.join(self.density_basedir, video_name)

            for i in range(start_frame_index, end_frame_index):
                frame_index = i + 1
                frame_name = frame_wildcard % frame_index
                # print frame_name
                # print os.path.join(video_dir, frame_name)
                # print os.path.join(video_dir, frame_name)
                frame_path = glob.glob(os.path.join(video_dir, frame_name))[0]
                frame = self.pre_process_img(cv2.imread(frame_path),sort='rgb')
                current_frame_list.append(frame)

            if density_length=='full':
                for i in range(start_frame_index,end_frame_index):
                    frame_index = i + 1
                    frame_name = frame_wildcard % frame_index
                    # print os.path.join(density_dir, frame_name)
                    density_path = glob.glob(os.path.join(density_dir, frame_name))[0]
                    density = self.pre_process_img(cv2.imread(density_path, 0),greyscale=True)
                    # print density.shape;exit()
                    current_density_list.append(density)                
            elif density_length=='one':
                frame_index=end_frame_index
                frame_name = frame_wildcard%frame_index
                density = self.pre_process_img(cv2.imread(glob.glob(os.path.join(density_dir, frame_name))[0], 0), True)
                current_density_list.append(density)

            frame_batch.append(np.array(current_frame_list))
            density_batch.append(np.array(current_density_list))
        # print np.array(frame_batch).shape,np.array(density_batch).shape;#(50, 16, 112, 112, 3)
        # print np.array(frame_batch).shape
        if len(frame_batch)==0:
            return None
        return np.transpose(np.array(frame_batch),(0,4,1,2,3)),np.transpose(np.array(density_batch),(0,4,1,2,3))

    def pre_process_img(self, image, greyscale=False, sort='rgb'):

        if self.data_augmentation:
            if 'horizontal_flip' in self.augmentation_list:
                image = cv2.flip(image, 1)
            if 'crop' in self.augmentation_list:
                image = cv2.resize(image, dsize=self.size_before_crop)
                image = image[self.crop_pos[0]:self.crop_pos[0]+self.img_size[0], self.crop_pos[1]:self.crop_pos[1]+self.img_size[1], ...]
                # print image.shape;exit()
        image = cv2.resize(image, dsize = self.img_size)
        if greyscale==False:
            if sort=='rgb':
                image = image[:, :, ::-1]
                image = image-self.MEAN_VALUE
        else:
            image = image[...,None]
            # print np.mean(image);exit()
        image = image / 255.
        return image