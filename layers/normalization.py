from ..layers import Layer
from  .. import nonlinearities
from lasagne import init
import theano.tensor as T
import numpy as np
from .. import utils


__all__ = [
    "ContrastNormLayer",
    "BatchNormLayer"
]


class ContrastNormLayer(Layer):
    """
    Implements per image contrast normalization
    """
    def __init__(self, incoming, norm_type="mean_var", name=None):
        super(ContrastNormLayer, self).__init__(incoming, name)
        if norm_type not in [None, "mean_var"]:
            raise Exception("norm_type %s Not implemented!" % norm_type)
        self.norm_type = norm_type

    def get_output_for(self, input, **kwargs):
        output_shape = input.shape
        if input.ndim > 2:
            input = T.flatten(input, 2)
        if self.norm_type == "mean_var":
            input -= T.mean(input, axis=1, keepdims=True)
            input /= T.std(input, axis=1, keepdims=True)
        input = input.reshape(output_shape)
        return input


# TODO: Consider using default_updates to clean up mean/std params. (https://gist.github.com/f0k/f1a6bd3c8585c400c190)
# TODO: Consider parameterizing moving average versus exponential average to be able to apply to test data at eval time
class BatchNormLayer(Layer):
    """
    Implements per batch input normalization
    """
    def __init__(self, incoming, eta=1e-2, alpha=.1, learn_transform=True,
                 nonlinearity=nonlinearities.linear, name=None): #window_size=10
        super(BatchNormLayer, self).__init__(incoming, name)
        if nonlinearity is None:
            self.nonlinearity  = nonlinearities.linear
        else:
            self.nonlinearity  = nonlinearity
        self.eval_nonlinearity = self.nonlinearity
        self.eta               = eta
        self.alpha             = alpha
        self.learn_transform   = learn_transform

        beta   = init.Constant(val=0)
        gamma  = init.Constant(val=1)
        means  = init.Constant(val=0)
        stdevs = init.Constant(val=1)

        input_shape = incoming.output_shape # batch_size, channels, width * height
        if len(input_shape) > 2:
            input_shape = [input_shape[0],np.prod(input_shape[1:])]

        self.beta  = self.add_param(beta, (input_shape[1],),name='beta',regularizable=False)
        self.gamma = self.add_param(gamma, (input_shape[1],),name='gamma',regularizable=False)

        self.means = utils.create_param(means,(input_shape[1],), name='means')
        self.stdevs = utils.create_param(stdevs,(input_shape[1],), name='stdevs')

        self.batch_size = incoming.output_shape[0]

    def get_output_for(self, input, deterministic=False, **kwargs):
        beta   = self.beta
        gamma  = self.gamma
        means  = self.means
        stdevs = self.stdevs

        output_shape = input.shape

        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)

        if deterministic == False:
            m = T.mean(input, axis=0, keepdims=False)
            s = T.sqrt(T.var(input, axis=0, keepdims=False) + self.eta)

            means.default_update = self.alpha * means + (1-self.alpha) * m
            Es = self.alpha * stdevs + (1-self.alpha) * s
            u  = self.batch_size / (self.batch_size - 1)
            stdevs.default_update = u * Es

        else:
            m = means
            s = stdevs

        output = input - m
        output /= s

        # transform normalized outputs based on learned shift and scale
        if self.learn_transform is True:
            output = gamma * output + beta
        output = output.reshape(output_shape)
        return self.nonlinearity(output)

        # TODO: Deal with eval_output_activation
        # activation = output
        #
        # if kwargs.has_key('eval') and kwargs['eval'] and hasattr(self,'eval_output_activation') and self.eval_output_activation is not None:# and self.eval_output_activation.lower() == 'linear':
        #     self.eval_nonlinearity = self.eval_output_activation
        #     return self.eval_nonlinearity(activation)
        # else:
        #     return self.nonlinearity(activation)


    def get_output_shape_for(self, input_shape):
        return input_shape
