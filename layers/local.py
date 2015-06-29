import numpy as np
import theano
import theano.tensor as T
#
from lasagne import init
from .. import nonlinearities
from .. import utils
from ..layers import Layer, InputLayer, Conv2DLayer


__all__ = [
    "LocalLayer"
]


class LocalLayer(Layer):
    def __init__(self, incoming, num_units, img_shape, local_filters,
                 W=init.GlorotUniform(), b=init.Constant(0.), mode='square',
                 nonlinearity=nonlinearities.rectify,
                 edgeprotect=True,
                 cn=False, prior=None, seed=123778, name=None):
        """
        :param local_filters: list of tuples containing (filter size, probability of selecting this size).  Probabilities must sum to 1
        """
        super(LocalLayer, self).__init__(incoming, name)

        self.num_units     = num_units
        self.cn            = cn
        self.edgeprotect   = edgeprotect
        self.mode          = mode
        self.local_filters = local_filters
        self.img_shape     = img_shape

        input_shape = self.input_layer.output_shape # batch_size, channels, width * height
        if len(input_shape) == 2:
            input_shape = [input_shape[0],1,input_shape[1]]
        self.num_inputs = int(np.prod(input_shape[2:]))

        self.nonlinearity = nonlinearity if nonlinearity is not None else nonlinearities.linear

        self.seed = seed
        np.random.seed(self.seed)

        prior                  = self._make_prior(prior)
        local_filters          = self._generate_local_filters()
        centers, prev_centers  = self._compute_centers(local_filters, num_units, prior)
        localmask              = self._create_masks(local_filters, prev_centers, centers, input_shape)
        self.centers           = centers
        self.localmask         = theano.shared(value=localmask, name='localmask', borrow=True)
        self.local_mask_counts = T.sum(self.localmask, axis=0).astype(theano.config.floatX)

        self._set_weight_params(W, b, input_shape, num_units, localmask)
        self._set_out_mask(num_units)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)

    def get_output_for(self, input, *args, **kwargs):
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)

        W = self.W
        b = self.b

        if self.cn:
            activation = self._compute_local_cn_acts(input, W*self.localmask)
        else:
            activation = T.dot(input, W*self.localmask)

        if b is not None:
            activation += b.dimshuffle('x', 0)

        return self.nonlinearity(activation)

        # TODO: Deal with eval_output_activation
        # self.eval_nonlinearity = self.nonlinearity
        # if kwargs.has_key('eval') and kwargs['eval'] and hasattr(self,'eval_output_activation') and self.eval_output_activation is not None:# and self.eval_output_activation.lower() == 'linear':
        #     self.eval_nonlinearity = self.eval_output_activation
        #     return self.eval_nonlinearity(activation)
        # else:
        #     return self.nonlinearity(activation)

    def _compute_centers(self, local_filters, num_units, prior):
        all_points = np.array(range(len(prior)))

        centers_layer = self.input_layer
        while centers_layer.__class__ not in [LocalLayer,InputLayer,Conv2DLayer]:
            centers_layer = centers_layer.input_layer
        prev_centers = centers_layer.centers if hasattr(centers_layer,'centers') else None
        if prev_centers is None:
            prev_centers = []
            for i in xrange(self.img_shape[0]):
                for j in xrange(self.img_shape[1]):
                    prev_centers.append([i,j,i*self.img_shape[1]+j])

        count = 0
        vcenters = []
        hcenters = []
        while (count < num_units):
            centersTemp = np.random.choice(all_points, 1, p=prior)
            vcenter     = int(centersTemp / self.img_shape[1])
            hcenter     = centersTemp % self.img_shape[1]
            if self._is_valid(vcenter, hcenter, local_filters[count]):
                count += 1
                vcenters.append(vcenter)
                hcenters.append(hcenter)
        centers = zip(vcenters,hcenters,xrange(num_units))
        return centers, prev_centers

    def _is_valid(self, vcenter, hcenter, filter_size):
        if not self.edgeprotect:
            return True
        if self.mode == 'square':
            radius = (filter_size - 1)/2
        elif self.mode == 'radius':
            radius = filter_size
        else:
            return True
        left  = hcenter - radius
        right = hcenter + radius
        up    = vcenter - radius
        down  = vcenter + radius
        if left < 0 or up < 0:
            return False
        if right >= self.img_shape[1]:
            return False
        if down >= self.img_shape[0]:
            return False
        return True

    def _set_weight_params(self, W, b, input_shape, num_units, localmask):
            w_params = self.add_param(W, (self.num_inputs*input_shape[1], num_units))
            w_params = w_params.get_value() * localmask
            W        = theano.shared(w_params,'W')
            self.W   = W
            self.b   = self.add_param(b, (num_units,)) if b is not None else None

    def _set_out_mask(self, num_units):
        out_mask = np.zeros((num_units,1,num_units))
        for i in range(out_mask.shape[0]):
            out_mask[i, :, i] = 1
        self.out_mask_T = theano.shared(out_mask,name='out_mask_T').astype(theano.config.floatX)

    def _make_prior(self, prior):
        prior = np.ones(self.img_shape).astype(float).flatten() if prior is None else prior.astype(float)
        assert prior.min()>=0
        prior=prior/prior.sum()
        prior=prior.flatten()
        return prior.astype(theano.config.floatX)

    def _generate_local_filters(self):
        local_filter_sizes = [l[0] for l in self.local_filters]
        local_filter_probs = [l[1] for l in self.local_filters]
        local_filters = np.random.choice(local_filter_sizes, self.num_units, p=local_filter_probs)
        return local_filters

    def _create_masks(self, local_filters, prev_centers, centers, input_shape):
        if self.mode == 'nearest':
            dists=[[((prev_cen[0]-cen[0])**2+(prev_cen[1]-cen[1])**2,prev_cen[2])
                    for prev_cen in prev_centers] for cen in centers]

            connect_list = [[a[1] for a in sorted(dist)[:n_closest]]
                            for dist, n_closest in zip(dists,local_filters)]
        elif self.mode == 'radius':
            dists=[[((prev_cen[0]-cen[0])**2+(prev_cen[1]-cen[1])**2,prev_cen[2])
                    for prev_cen in prev_centers] for cen in centers]

            connect_list = [[d[1] for d in dist if d[0]<radius**2]
                            for dist,radius in zip(dists,local_filters)]
        elif self.mode == 'square':
            local_filters = [(side_len-1)/2 for side_len in local_filters]
            connect_list = [[prev_cen[2] for prev_cen in prev_centers
                             if (prev_cen[0]>=cen[0]-side_len
                                 and prev_cen[0]<=cen[0]+side_len
                                 and prev_cen[1]>=cen[1]-side_len
                                 and prev_cen[1]<=cen[1]+side_len)]
                            for cen,side_len in zip(centers,local_filters)]
        else:
            raise RuntimeError("Mode must be one of 'nearest', 'radius', or 'square'.")

        localmask = []
        for unit in connect_list:
            unitmask=[]
            for i in xrange(self.num_inputs):
                if i in unit:
                    unitmask.append(1)
                else:
                    unitmask.append(0)
            localmask.append(unitmask)
        localmask=np.array(localmask).T
        localmask=utils.floatX(localmask)
        # it's important to use tile, not repeat!
        localmask=np.tile(A=localmask,reps=(input_shape[1],1))

        return localmask



    def _compute_local_cn_acts(self, input, W):
        # Without Scan (Faster than scan, but still way too slow)
        shuffledIn    = input.dimshuffle(0,1,'x')
        shuffledMasks = self.localmask.dimshuffle('x',0,1)

        # cubeIn = T.repeat(shuffledIn,self.localmask.shape[1],2)
        # cubeMasks = T.repeat(shuffledMasks,input.shape[0],0)

        maskedIn     = shuffledIn * shuffledMasks
        maskedInMean = T.sum(maskedIn,axis=1,keepdims=True) / T.sum(shuffledMasks,axis=1,keepdims=True)
        maskedInVar  = T.sum(T.sqr((maskedIn-maskedInMean)*shuffledMasks),axis=1,keepdims=True)/T.sum(shuffledMasks,axis=1,keepdims=True)
        maskedInSTD  = T.sqrt(maskedInVar)

        maskedInSubMean = maskedIn - maskedInMean
        maskedCN        = maskedInSubMean / maskedInSTD
        # maskedCN = maskedInSubMean

        shuffledInCN = maskedCN.dimshuffle(2,0,1)

        allOuts      = T.dot(shuffledInCN, W)

        diagMask     = T.eye(self.localmask.shape[1],self.localmask.shape[1]).dimshuffle(0,'x',1)
        diagMaskAll  = allOuts * diagMask

        activation   = T.sum(diagMaskAll,axis=0)
        return activation
        # # With Scan
        # #TODO: Get working quickly, update to work with color channels
        # masked_input=T.repeat(input.dimshuffle(0,1,'x'),self.localmask.shape[1],2)*T.repeat(self.localmask.dimshuffle('x',0,1),input.shape[0],0)
        # m=T.sum(masked_input,axis=1)/self.local_mask_counts
        # # v=T.sqr(T.sum((masked_input-m.dimshuffle(0,'x',1))*T.repeat(self.localmask.dimshuffle('x',0,1),input.shape[0],0),axis=1))/self.local_mask_counts
        # v=T.sum(T.sqr((masked_input-m.dimshuffle(0,'x',1))*T.repeat(self.localmask.dimshuffle('x',0,1),input.shape[0],0)),axis=1)/self.local_mask_counts
        # s=T.sqrt(v)
        # new_input=(input.dimshuffle(0,1,'x')-m.dimshuffle(0,'x',1))/s.dimshuffle(0,'x',1)
        # slices,updates=theano.scan(fn=lambda maskedIn, w:T.dot(maskedIn,w),outputs_info=None,
        #     sequences=[new_input.dimshuffle(2,0,1)],
        #     non_sequences=W)
        # # activation=slices.diagonal(axis1=0,axis2=2)
        # # out_mask=np.zeros((self.num_units,1,self.num_units))
        # # for i in range(out_mask.shape[0]):
        # #     out_mask[i,:,i]=1
        # # out_mask_T=theano.shared(out_mask)
        # out_mask_T=T.repeat(self.out_mask_T,input.shape[0],1)
        # mask_out=out_mask_T*slices
        # activation=T.sum(mask_out,axis=0).astype(theano.config.floatX)
        # return activation
