import numpy as np
import theano
import theano.tensor as T
#
from lasagne import init
from lasagne import nonlinearities
from lasagne import utils
from lasagne.layers.base import Layer
from lasagne.layers.input import InputLayer
from lasagne.layers.conv import Conv2DLayer


__all__ = [
    "LocalLayer"
]


# TODO: Add a flag to local layer that would ensure that filters do not fall off the edge
class LocalLayer(Layer):
    def __init__(self, input_layer, num_units, img_shape, local_filters,
                 W=init.Uniform(), b=init.Constant(0.), mode='square',
                 localmask=None, nonlinearity=nonlinearities.rectify, cn=False, prior=None, **kwargs):
        super(LocalLayer, self).__init__(input_layer, **kwargs)

        self.num_units = num_units

        input_shape = self.input_layer.output_shape # batch_size, channels, width * height
        if len(input_shape) == 2:
            input_shape = [input_shape[0],1,input_shape[1]]
        num_inputs = int(np.prod(input_shape[2:]))

        self.num_inputs=num_inputs

        if nonlinearity is None:
            self.nonlinearity = nonlinearities.linear
        else:
            self.nonlinearity = nonlinearity
        self.eval_nonlinearity = self.nonlinearity

        if localmask is None:
            if prior is None:
                prior=np.ones(img_shape).astype(float).flatten()
            prior=prior.astype(float)
            assert prior.min()>=0
            prior=prior/prior.sum()
            # if prior.sum() != 1.0:
            #     prior=T.nnet.softmax(prior).eval()[0]
            prior=prior.flatten()
            all_points=np.array(range(len(prior)))

            centers_layer=input_layer
            while centers_layer.__class__ not in [LocalLayer,InputLayer,Conv2DLayer]:
                centers_layer=centers_layer.input_layer
            prev_centers = centers_layer.centers if hasattr(centers_layer,'centers') else None
            if prev_centers is None:
                prev_centers=[]
                for i in xrange(img_shape[0]):
                    for j in xrange(img_shape[1]):
                        prev_centers.append([i,j,i*img_shape[1]+j])

            # vcenters = [np.random.randint(0,img_shape[0],1)[0] for _ in xrange(num_units)]
            # hcenters = [np.random.randint(0,img_shape[1],1)[0] for _ in xrange(num_units)]
            centersTemp=np.random.choice(all_points,num_units,p=prior)
            vcenters=[int(i/img_shape[1]) for i in centersTemp]
            hcenters=[i%img_shape[1] for i in centersTemp]
            # print zip(vcenters,hcenters)
            centers = zip(vcenters,hcenters,xrange(num_units))
            if mode == 'nearest':
                dists=[[((prev_cen[0]-cen[0])**2+(prev_cen[1]-cen[1])**2,prev_cen[2]) for prev_cen in prev_centers]for cen in centers]

                connect_list = [[a[1] for a in sorted(dist)[:n_closest]] for dist, n_closest in zip(dists,local_filters)]
            elif mode == 'radius':
                dists=[[((prev_cen[0]-cen[0])**2+(prev_cen[1]-cen[1])**2,prev_cen[2]) for prev_cen in prev_centers]for cen in centers]

                connect_list = [[d[1] for d in dist if d[0]<radius**2] for dist,radius in zip(dists,local_filters)]
            elif mode == 'square':
                local_filters = [(side_len-1)/2 for side_len in local_filters]
                connect_list = [[prev_cen[2] for prev_cen in prev_centers if (prev_cen[0]>=cen[0]-side_len and prev_cen[0]<=cen[0]+side_len and prev_cen[1]>=cen[1]-side_len and prev_cen[1]<=cen[1]+side_len)] for cen,side_len in zip(centers,local_filters)]
            else:
                raise RuntimeError("Mode must be one of 'nearest', 'radius', or 'square'.")

            localmask = []
            for unit in connect_list:
                unitmask=[]
                for i in xrange(num_inputs):
                    if i in unit:
                        unitmask.append(1)
                    else:
                        unitmask.append(0)
                localmask.append(unitmask)
            localmask=np.array(localmask).T
            self.cn=cn
            localmask=utils.floatX(localmask)
            # localmask=np.repeat(a=localmask,repeats=input_shape[1],axis=0)
            localmask=np.tile(A=localmask,reps=(input_shape[1],1))
            self.centers = centers
            self.params['localmask'] = theano.shared(value=localmask, name='localmask', borrow=True)
            self.local_mask_counts=T.sum(self.params['localmask'], axis=0).astype(theano.config.floatX)
            # self.local_mask_indices=T.nonzero(self.params['localmask'],return_matrix=True).astype(theano.config.floatX)
        else:
            self.params['localmask'] = localmask

        w_params = self.add_param(W, (num_inputs*input_shape[1], num_units))
        w_params = w_params.get_value() * localmask
        W = theano.shared(w_params,'W')
        # W_class = W.__class__
        # W_dict = W.__dict__

        # W_dict['localmask']=self.params['localmask']
        # W = W_class(**W_dict)
        self.params['W'] = W#self.add_param(W, (num_inputs, num_units)) ###########
        self.params['b'] = self.add_param(b, (num_units,)) if b is not None else None

        out_mask=np.zeros((num_units,1,num_units))
        for i in range(out_mask.shape[0]):
            out_mask[i,:,i]=1
        self.out_mask_T=theano.shared(out_mask,name='out_mask_T').astype(theano.config.floatX)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)

    def get_output_for(self, input, *args, **kwargs):
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)

        W=self.params['W']
        b=self.params['b']

        if self.cn:
            # Without Scan (Faster than scan, but still way too slow)
            shuffledIn = input.dimshuffle(0,1,'x')
            shuffledMasks = self.params['localmask'].dimshuffle('x',0,1)

            # cubeIn = T.repeat(shuffledIn,self.params['localmask'].shape[1],2)
            # cubeMasks = T.repeat(shuffledMasks,input.shape[0],0)

            maskedIn = shuffledIn * shuffledMasks
            maskedInMean = T.sum(maskedIn,axis=1,keepdims=True) / T.sum(shuffledMasks,axis=1,keepdims=True)
            maskedInVar = T.sum(T.sqr((maskedIn-maskedInMean)*shuffledMasks),axis=1,keepdims=True)/T.sum(shuffledMasks,axis=1,keepdims=True)
            maskedInSTD = T.sqrt(maskedInVar)

            maskedInSubMean = maskedIn - maskedInMean
            maskedCN = maskedInSubMean / maskedInSTD
            # maskedCN = maskedInSubMean

            shuffledInCN = maskedCN.dimshuffle(2,0,1)

            allOuts = T.dot(shuffledInCN,W)

            diagMask = T.eye(self.params['localmask'].shape[1],self.params['localmask'].shape[1]).dimshuffle(0,'x',1)
            diagMaskAll = allOuts * diagMask

            activation = T.sum(diagMaskAll,axis=0)


            # # With Scan
            # #TODO: Get working quickly, update to work with color channels
            # masked_input=T.repeat(input.dimshuffle(0,1,'x'),self.params['localmask'].shape[1],2)*T.repeat(self.params['localmask'].dimshuffle('x',0,1),input.shape[0],0)
            # m=T.sum(masked_input,axis=1)/self.local_mask_counts
            # # v=T.sqr(T.sum((masked_input-m.dimshuffle(0,'x',1))*T.repeat(self.params['localmask'].dimshuffle('x',0,1),input.shape[0],0),axis=1))/self.local_mask_counts
            # v=T.sum(T.sqr((masked_input-m.dimshuffle(0,'x',1))*T.repeat(self.params['localmask'].dimshuffle('x',0,1),input.shape[0],0)),axis=1)/self.local_mask_counts
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

        else:
            activation = T.dot(input, W)

        if b is not None:
            activation = activation + b.dimshuffle('x', 0)

        return self.nonlinearity(activation)

        # TODO: Deal with eval_output_activation
        # if kwargs.has_key('eval') and kwargs['eval'] and hasattr(self,'eval_output_activation') and self.eval_output_activation is not None:# and self.eval_output_activation.lower() == 'linear':
        #     self.eval_nonlinearity = self.eval_output_activation
        #     return self.eval_nonlinearity(activation)
        # else:
        #     return self.nonlinearity(activation)
