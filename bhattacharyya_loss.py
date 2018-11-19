import numpy as np
from scipy.stats import gaussian_kde
import caffe
import numpy as np
# import warnings
import pdb
import scipy.misc as scimisc
import cv2
import matplotlib.pyplot as plt
class GBDLossLayer(caffe.Layer):
    """
    Compute the generalized Bhattacharyya Distance loss.
    """
    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs (pred, gt) to compute distance.")

    def reshape(self, bottom, top):
        # difference is shape of prediction
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output  is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        # pdb.set_trace()
        # softmax and reshaping for the ground truth heatmap
        gts = np.empty_like(bottom[0].data)
        for i in range(bottom[0].data.shape[0]):
            gt = bottom[1].data[i,...].squeeze()
            gt_exp = np.exp(gt-np.max(gt))
            gt_snorm = gt_exp/np.sum(gt_exp)
            gts[i, ...] = gt_snorm
        gt = gts
        # softmax for the predicted heatmap
        preds = np.empty_like(bottom[0].data)
        for i in range(bottom[0].data.shape[0]): # batch size
            # pdb.set_trace()
            pmap = bottom[0].data[i,...].squeeze()
            # apply softmax normalization to obtain a probability distribution
            pmap_exp = np.exp(pmap-np.max(pmap)) # range is now 0 to 1
            pmap_snorm = pmap_exp/np.sum(pmap_exp)
            # print np.min(pmap_snorm)
            preds[i,...] = pmap_snorm
        pmap = preds

        # get alpha parameter:
        alpha= 0.5

        # compute log and difference of log values
        epsilon        = np.finfo(np.float).eps # epsilon (float or float 32)
        # print np.min(pmap), np.min(gt)
        prod_alpha     = pmap ** alpha *  gt ** (1 - alpha)
        prod_alpha_sum = 0
        # for i in range(2, len(prod_alpha)):
            # prod_alpha_sum += np.sum(prod_alpha, axis=i)
        if len(prod_alpha.shape)==4:
            prod_alpha_sum = np.sum(np.sum(prod_alpha, axis=3), axis=2)
        if len(prod_alpha.shape)==5:
            prod_alpha_sum = np.sum(np.sum(np.sum(prod_alpha, axis=4), axis=3), axis=2)

        loss = -np.log(np.maximum(prod_alpha_sum,epsilon))

        # compute combined loss
        top[0].data[...] = np.mean(loss) #averaged per image in the batch

        # calculate value for bkward pass - self.diff = dl/dpk
        const     = -alpha / prod_alpha_sum
        if len(prod_alpha.shape)==4:
            const     = const[:,:,np.newaxis,np.newaxis]
            self.diff = const * (prod_alpha * (1 - pmap) - (prod_alpha_sum[:,:,np.newaxis,np.newaxis] - prod_alpha) * pmap)
        if len(prod_alpha.shape)==5:
            const     = const[:,:,np.newaxis,np.newaxis, np.newaxis]
            self.diff = const * (prod_alpha * (1 - pmap) - (prod_alpha_sum[:,:,np.newaxis,np.newaxis, np.newaxis] - prod_alpha) * pmap)
        # print len(prod_alpha.shape), self.diff.shape
        # const     = const[:,:,np.newaxis,np.newaxis]

        # print 'alpha = {:.2f}, min diff = {:.2e}, max diff = {:.2e}, range = {:.2e}'.format(alpha, np.min(self.diff), np.max(self.diff), np.max(self.diff) - np.min(self.diff))
        # fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(22,10))
        # ax1.imshow(gt[0,0,:,:])
        # ax2.imshow(pmap[0,0,:,:])
        # ax3.imshow(self.diff[0,0,:,:])
        # plt.show()

    def backward(self, top, propagate_down, bottom):    
        loss_wgt = top[0].diff
        # print loss_wgt
        for i in range(2):
            if not propagate_down[i]:
                continue
            if i == 0:
                sign = 1
            else:
                sign = -1
            bottom[i].diff[...] = sign * loss_wgt * self.diff / bottom[i].num


# class BhattacharyyaLossLayer(caffe.Layer):
#     @classmethod
#     def parse_args(cls, argsStr):
#         parser = argparse.ArgumentParser(description='python bhattacharyya loss layer')
#         parser.add_argument('--loss_weight', default=1.0, type=float)
#         args   = parser.parse_args(argsStr.split())
#         print('Using Config:')
#         pprint.pprint(args)
#         return args 

#    def get_density(x, cov_factor=0.1):
#         #Produces a continuous density function for the data in 'x'. Some benefit may be gained from adjusting the cov_factor.
#         density = gaussian_kde(x)
#         density.covariance_factor = lambda:cov_factor
#         density._compute_covariance()
#         return density

#    def setup(self, bottom, top):
#         self.param_ = BhattacharyyaLossLayer.parse_args(self.param_str)
#         assert len(bottom) == 2, 'There should be two bottom blobs'
#         predShape = bottom[0].data.shape
#         gtShape   = bottom[1].data.shape
#         for i in range(len(predShape)):
#             assert predShape[i] == gtShape[i], 'Mismatch: %d, %d' % (predShape[i], gtShape[i])
#         assert bottom[0].data.squeeze().ndim == bottom[1].data.squeeze().ndim, 'Shape Mismatch'
#     #Get the batchSz
#         self.batchSz_ = gtShape[0]
#         #Form the top
#         assert len(top)==1, 'There should be only one output blob'
#         top[0].reshape(1,1,1,1)
        
#     def forward(self, bottom, top):
#         X1 = bottom[0].data[...].squeeze()
#         X2 = bottom[1].data[...].squeeze()
#         N_STEPS = 200
#         #Get density functions:
#         d1 = get_density(X1)
#         d2 = get_density(X2)
#         # top[0].data[...] = self.param_.loss_weight * np.sum(np.abs(bottom[0].data[...].squeeze()\
#                                                      # - bottom[1].data[...].squeeze()))/float(self.batchSz_) 
#         xs = np.linspace(min(cX),max(cX),N_STEPS)
#         bht = 0
#         for x in xs:
#             p1 = d1(x)
#             p2 = d2(x)
#             bht += sqrt(p1*p2)*(max(cX)-min(cX))/N_STEPS
        
#         dist = -np.log(bht)

#         top[0].data[...] = self.param_.loss_weight * dist/float(self.batchSz_) 
#         glog.info('Loss is %f' % top[0].data[0])
    
#     def backward(self, top, propagate_down, bottom):
#         bottom[0].diff[...] = self.param_.loss_weight * np.sign(bottom[0].data[...].squeeze()\
#                                                          - bottom[1].data[...].squeeze())/float(self.batchSz_)
        
#     def reshape(self, bottom, top):
#         top[0].reshape(1,1,1,1)
#         pass