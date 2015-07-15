from ..layers import Layer
from .. import utils
from lasagne import init

# import theano
# from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import theano.tensor as T
# import numpy as np


__all__ = [
    "PReLULayer",
]

# TODO: Consider using a flag to allow tied param for coef/pivot
class PReLULayer(Layer):
    def __init__(self, incoming, coef=init.Constant(-.25), pivot=init.Constant(0), learn_pivot=False, name=None):
        super(PReLULayer, self).__init__(incoming, name)

        input_shape = self.input_layer.output_shape
        self.coef   = self.add_param(coef, input_shape[1:], regularizable=False)

        if learn_pivot:
            self.pivot = self.add_param(pivot, input_shape[1:], regularizable=False)
        else:
            self.pivot = utils.create_param(pivot, input_shape[1:])

    def get_output_for(self, input, **kwargs):
        x = input
        p = self.pivot
        a = self.coef
        return T.maximum(x - p, 0) - a * (T.maximum((-(x - p)), 0))

