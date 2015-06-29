from ..init import *
from ..nonlinearities import *
from copy import deepcopy
from numbers import Number
# import lasagne.layers.base

from .base import *

'''
Have:
-Dropout

Need:
-Gaussian dropout
-Gaussian noise
'''

class DropoutLayer(SingleParentLayer):
    def __init__(self, parent, p=0.5, rescale=True, **kwargs):
        if not isinstance(p,Number)\
                or p<0 or p>1:
            raise TypeError("p must be a number between 0 and 1 inclusive!")
        if not rescale in [True,False]:
            raise TypeError("rescale must be True or False!")
        super(DropoutLayer, self).__init__(parent, **kwargs)
        self.p=p
        self.rescale=rescale

    def instantiate(self,instantiated_layers, layer_ctr):
        layer_obj = lasagne.layers.noise.DropoutLayer(
            incoming=self.parent,
            p=self.p,
            rescale=self.rescale
        )
        instantiated_layers[layer_ctr] = layer_obj
        return layer_obj