from ..init import *
from ..nonlinearities import *
from copy import deepcopy
from numbers import Number
# import lasagne.layers.base

from .base import *

'''
Have:
-Concat
-Elemwise Sum

Need:
-Elemwise Mul
-Gate
'''

class ConcatLayer(MultipleParentsLayer):
    def __init__(self, parents, axis=1, **kwargs):
        if not isinstance(axis, int):
            raise TypeError("axis must be 1 or 0!")
        super(ConcatLayer, self).__init__(parents, **kwargs)
        self.axis = axis

    def instantiate(self, instantiated_layers, layer_name):
        layer_obj = lasagne.layers.merge.ConcatLayer(
            incomings=[instantiated_layers[parent] for parent in self.parents],
            axis=self.axis
        )
        instantiated_layers[layer_name] = layer_obj
        return layer_obj

class ElemwiseSumLayer(MultipleParentsLayer):
    def __init__(self, parents, coeffs=1, **kwargs):
        if not isinstance(coeffs, Number):
            raise TypeError("coeffs must be a number!")
        super(ElemwiseSumLayer, self).__init__(parents, **kwargs)
        self.coeffs = coeffs

    def instantiate(self, instantiated_layers, layer_name):
        layer_obj = lasagne.layers.merge.ElemwiseSumLayer(
            incomings=[instantiated_layers[parent] for parent in self.parents],
            nonlinearity=self.nonlinearity.instantiate(),
            coeffs=self.coeffs
        )
        instantiated_layers[layer_name] = layer_obj
        return layer_obj