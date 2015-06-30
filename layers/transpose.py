from xnn.layers import DenseLayer, LocalLayer, Layer
from xnn import init
from xnn.nonlinearities import rectify
import theano.tensor as T
from copy import deepcopy


class TransposeDenseLayer(DenseLayer):
    def get_output_for(self, input, **kwargs):
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)

        activation = T.dot(input, self.W.T)
        if self.b is not None:
            activation = activation + self.b.dimshuffle('x', 0)
        return self.nonlinearity(activation)


class TransposeLocalLayer(DenseLayer):
    def __init__(self, incoming, num_units, localmask, W, b=init.Constant(0.), nonlinearity=rectify, name=None):
        super(TransposeLocalLayer, self).__init__(incoming, num_units, W, b, nonlinearity)
        self.name = name
        self.localmask = localmask

    def get_output_for(self, input, *args, **kwargs):
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)

        W = self.W
        b = self.b
        Wmasked = W*self.localmask

        activation = T.dot(input, Wmasked.T)

        if b is not None:
            activation += b.dimshuffle('x', 0)

        return self.nonlinearity(activation)
