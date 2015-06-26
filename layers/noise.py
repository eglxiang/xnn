from lasagne.layers.base import Layer
import theano
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import theano.tensor as T
import numpy as np


__all__ = [
    "GaussianDropoutLayer",
]


class GaussianDropoutLayer(Layer):
    """ Drouput with multiplicative Gaussian noise.
    See http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf
    Does not require rescaling at test time.
    """
    def __init__(self, incoming, sigma=1.0, name=None):
        super(GaussianDropoutLayer, self).__init__(incoming, name)
        self._srng = RandomStreams(np.random.randint(1, 2147462579))
        self.sigma = sigma

    def get_output_for(self, input, deterministic=False, **kwargs):
        if deterministic or self.sigma == 0:
            return input
        else:
            noise = self._srng.normal(input.shape, avg=1.0, std=self.sigma, dtype=theano.config.floatX)
            return input * noise

gaussdropout = GaussianDropoutLayer # shortcut
