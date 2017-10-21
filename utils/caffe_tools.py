import numpy as np


class SimpleTransformer:

    """
    SimpleTransformer is a simple class for preprocessing and deprocessing
    images for caffe.
    """

    def __init__(self, mean=[128, 128, 128]):
        self.mean = np.array(mean, dtype=np.float32)
        self.scale = 1.0

    def set_mean(self, mean):
        """
        Set the mean to subtract for centering the data.
        """
        self.mean = mean

    def set_scale(self, scale):
        """
        Set the data scaling.
        """
        self.scale = scale

    def preprocess(self, im):
        """
        preprocess() emulate the pre-processing occurring in the vgg16 caffe
        prototxt.
        """

        im = np.float32(im)
        im = im[:, :, ::-1]  # change to BGR
        im -= self.mean
        im *= self.scale
        im = im.transpose((2, 0, 1))

        return im

    def deprocess(self, im):
        """
        inverse of preprocess()
        """
        im = im.transpose(1, 2, 0)
        im /= self.scale
        im += self.mean
        im = im[:, :, ::-1]  # change to RGB

        return np.uint8(im)


class CaffeSolver:

    """
    Caffesolver is a class for creating a solver.prototxt file. It sets default
    values and can export a solver parameter file.
    Note that all parameters are stored as strings. Strings variables are
    stored as strings in strings.
    """

    def __init__(self, trainnet_prototxt_path="train.prototxt"):

        self.sp = {}

        # critical:
        self.sp['base_lr'] = '0.01'
        self.sp['momentum'] = '0.9'

        # looks:
        self.sp['display'] = '100'
        self.sp['iter_size'] = '1'

        # learning rate policy
        self.sp['lr_policy'] = '"inv"'

        # important, but rare:
        self.sp['gamma'] = '0.0001'
        self.sp['power'] = '0.75'
        self.sp['weight_decay'] = '0.0005'
        self.sp['train_net'] = '"' + trainnet_prototxt_path + '"'

        #
        self.sp['solver_mode'] = 'GPU'
        self.sp['solver_type'] = 'ADADELTA'
        self.sp['delta'] = '1e-6'

        #snapshot
        self.sp['snapshot'] = '50000'
        self.sp['snapshot_prefix'] = '"../training_output/salicon/snapshot"'
    def add_from_file(self, filepath):
        """
        Reads a caffe solver prototxt file and updates the Caffesolver
        instance parameters.
        """
        with open(filepath, 'r') as f:
            for line in f:
                if line[0] == '#':
                    continue
                splitLine = line.split(':')
                self.sp[splitLine[0].strip()] = splitLine[1].strip()

    def write(self, filepath):
        """
        Export solver parameters to INPUT "filepath". Sorted alphabetically.
        """
        f = open(filepath, 'w')
        for key, value in sorted(self.sp.items()):
            if not(type(value) is str):
                raise TypeError('All solver parameters must be strings')
            f.write('%s: %s\n' % (key, value))
