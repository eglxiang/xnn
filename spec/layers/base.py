from ..init import *
from ..nonlinearities import *
from copy import deepcopy
from numbers import Number
import lasagne.layers.base
# from spec.layers.conv import *

class Layer(object):
    def __init__(self, **kwargs):
        if kwargs:
            self.additional_args = kwargs
        self.type = self.__class__.__name__
        self.has_learned_params = False

    def to_dict(self):
        properties = deepcopy(self.__dict__)
        return properties

class InputLayer(Layer):
    def __init__(self, shape=(48,48), **kwargs):
        if not isinstance(shape, (list, tuple)) \
                or any(not isinstance(x, Number) for x in shape):
            raise TypeError("shape must be a list or tuple of integers!")
        super(InputLayer, self).__init__(**kwargs)
        self.shape = shape

    def instantiate(self, instantiated_layers, layer_name):
        layer_obj = lasagne.layers.input.InputLayer(shape=self.shape)
        instantiated_layers[layer_name] = layer_obj
        return layer_obj

class SingleParentLayer(Layer):
    def __init__(self, parent, **kwargs):
        if not isinstance(parent, str):
            raise TypeError("parent must be a string.")
        super(SingleParentLayer, self).__init__(**kwargs)
        self.parent = parent

    def to_dict(self):
        properties = super(SingleParentLayer, self).to_dict()
        return properties

class MultipleParentsLayer(Layer):
    def __init__(self, parents, **kwargs):
        if not isinstance(parents, list):
            raise TypeError("parents must be a list of strings.")
        super(MultipleParentsLayer, self).__init__(**kwargs)
        self.parents = parents

    def to_dict(self):
        properties = super(MultipleParentsLayer, self).to_dict()
        return properties
