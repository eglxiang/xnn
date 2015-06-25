from lasagne.layers.base import Layer
from lasagne import nonlinearities
from lasagne import init
import theano.tensor as T


__all__ = [
    "ContrastNormLayer",
    "BatchNormLayer"
]


class ContrastNormLayer(Layer):
    """
    Implements per image contrast normalization
    """
    def __init__(self, incoming, norm_type="mean_var", **kwargs):
        super(ContrastNormLayer, self).__init__(incoming, **kwargs)
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
                 nonlinearity=nonlinearities.linear, **kwargs): #window_size=10
        super(BatchNormLayer, self).__init__(incoming, **kwargs)
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.linear
        else:
            self.nonlinearity = nonlinearity
        self.eval_nonlinearity = self.nonlinearity
        self.eta = eta
        self.alpha = alpha
        self.learn_transform = learn_transform

        beta = init.Constant(val=0)
        gamma = init.Constant(val=1)
        means = init.Constant(val=0)
        stdevs = init.Constant(val=1)

        self.params['beta'] = self.add_param(beta, (incoming.output_shape[1],))
        self.params['gamma'] = self.add_param(gamma, (incoming.output_shape[1],))

        self.params['means'] = self.add_param(means, (incoming.output_shape[1],))
        self.params['stdevs'] = self.add_param(stdevs, (incoming.output_shape[1],))

        self.means_updates = []
        self.stdevs_updates = []
        self.batch_size = incoming.output_shape[0]

    def get_output_for(self, input, deterministic=False, **kwargs):
        beta = self.params['beta']
        gamma = self.params['gamma']
        means = self.params['means']
        stdevs = self.params['stdevs']
        if deterministic == False:
            m = T.mean(input, axis=0, keepdims=False)
            s = T.sqrt(T.var(input, axis=0, keepdims=False) + self.eta)

            self.means_updates = [means, self.alpha * means + (1-self.alpha) * m]
            Es = self.alpha * stdevs + (1-self.alpha) * s
            u = self.batch_size / (self.batch_size - 1)
            self.stdevs_updates = [stdevs, u * Es]

        else:
            m = means
            s = stdevs

        output = input - m
        output /= s

        # transform normalized outputs based on learned shift and scale
        if self.learn_transform is True:
            output = gamma * output + beta

        return self.nonlinearity(output)

        # TODO: Deal with eval_output_activation
        # activation = output
        #
        # if kwargs.has_key('eval') and kwargs['eval'] and hasattr(self,'eval_output_activation') and self.eval_output_activation is not None:# and self.eval_output_activation.lower() == 'linear':
        #     self.eval_nonlinearity = self.eval_output_activation
        #     return self.eval_nonlinearity(activation)
        # else:
        #     return self.nonlinearity(activation)

    def get_params(self):
        if self.learn_transform == True:
            return [self.params['gamma']] + self.get_bias_params()
        else:
            return []

    def get_bias_params(self):
        if self.learn_transform == True:
            return [self.params['beta']]
        else:
            return []

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_additional_updates(self):
        return [self.means_updates, self.stdevs_updates]
