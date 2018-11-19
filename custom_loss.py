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

            #softmax norm
            # gt_exp = np.exp(gt-np.max(gt))
            # gt_snorm = gt_exp/np.sum(gt_exp)
            # 0-1 norm
            gt_snorm = gt-np.min(gt)
            gt_snorm = gt_snorm/np.max(gt_snorm)

            gts[i, ...] = gt_snorm
        gt = gts
        # softmax for the predicted heatmap
        preds = np.empty_like(bottom[0].data)
        for i in range(bottom[0].data.shape[0]): # batch size
            # pdb.set_trace()
            pmap = bottom[0].data[i,...].squeeze()
            # apply softmax normalization to obtain a probability distribution
            # pmap_exp = np.exp(pmap-np.max(pmap)) # range is now 0 to 1
            # pmap_snorm = pmap_exp/np.sum(pmap_exp)
            # 0-1 norm
            pmap_snorm = pmap-np.min(pmap)
            pmap_snorm = pmap_snorm/np.max(pmap_snorm)
            
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

class KldLossLayer(caffe.Layer):
    """
    Compute the kld loss.
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
        # softmax and reshaping for the ground truth heatmap
        gts = np.empty_like(bottom[0].data)
        for i in range(bottom[0].data.shape[0]):
            # pdb.set_trace()
            # gt = np.transpose(bottom[1].data[i,:,:,:],(1,2,0)).squeeze()
            gt = bottom[1].data[i, ...].squeeze()
            # gt = scimisc.imresize(gt,bottom[0].data.shape[2:4],interp='bilinear')
            # gt = gt/255.0
            # # softmax normalization to obtain probability distribution
            gt_exp = np.exp(gt-np.max(gt))
            gt_snorm = gt_exp/np.sum(gt_exp)
            # back into original block
            # gt = np.transpose(gt_snorm[:,:,np.newaxis,np.newaxis],(3,2,0,1))
            gts[i,...] = gt_snorm
        gt = gts
        # softmax for the predicted heatmap
        preds = np.empty_like(bottom[0].data)
        for i in range(bottom[0].data.shape[0]): # batch size
            # pdb.set_trace()
            # pmap = np.transpose(bottom[0].data[i,:,:,:],(1,2,0)).squeeze()
            pmap = bottom[0].data[i, ...].squeeze()
            # apply softmax normalization to obtain a probability distribution
            pmap_exp = np.exp(pmap-np.max(pmap)) # range is now 0 to 1
            pmap_snorm = pmap_exp/np.sum(pmap_exp)
            # back into original block
            # pmap = np.transpose(pmap_snorm[:,:,np.newaxis,np.newaxis],(3,2,0,1))
            preds[i,...] = pmap_snorm
        pmap = preds

        # compute log and difference of log values
        # pdb.set_trace()
        epsilon = np.finfo(np.float).eps # epsilon (float or float 32)
        gt_ln = np.log(np.maximum(gt,epsilon))
        pmap_ln = np.log( np.maximum(pmap,epsilon))
        loss = gt*(gt_ln-pmap_ln)
        # compute combined loss
        top[0].data[...] = np.sum(loss) / bottom[0].num

        # # calculate value for bkward pass - self.diff = dl/dpk
        # for hind in range(pmap.shape[2]):
        #   for wind in range(pmap.shape[3]):
        #       # for each pixel in the distribution - 2D map 
        #       iequalk = np.zeros(pmap.shape)
        #       inotequalk = np.ones(pmap.shape)
        #       iequalk[:,:,hind,wind] = 1  
        #       inotequalk[:,:,hind,wind] = 0
        #       self.diff[:,:,hind,wind] = -1*( np.sum(np.sum(gt*(1-pmap)*iequalk,axis=3),axis=2) - pmap[:,:,hind,wind]*np.sum(np.sum(gt*inotequalk,axis=3),axis=2) )

        self.diff = gt * pmap - (gt * (1 - pmap))

    def backward(self, top, propagate_down, bottom):    
        loss_wgt = top[0].diff
        for i in range(2):
            if not propagate_down[i]:
                continue
            if i == 0:
                sign = 1
            else:
                sign = -1
            bottom[i].diff[...] = sign * loss_wgt * self.diff / bottom[i].num

