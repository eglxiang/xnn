from spec.init import *
from spec.nonlinearities import *
from copy import deepcopy
from numbers import Number
import lasagne.layers.base

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
        layer_obj = lasagne.layers.base.InputLayer(shape=self.shape)
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

class DenseLayer(SingleParentLayer):
    def __init__(self, parent, num_units, Winit=Uniform(range=(0,1)),
                 binit=Constant(val=0), nonlinearity=linear(), **kwargs):
        if not isinstance(Winit, Initializer):
            raise TypeError("Winit must be an object of type Initializer!")
        if not isinstance(binit, Initializer):
            raise TypeError("binit must be an object of type Initializer!")
        if not isinstance(nonlinearity, Activation):
            raise TypeError("nonlinearity must be an object of type Activation!")

        super(DenseLayer, self).__init__(parent, **kwargs)
        self.num_units = num_units
        self.Winit = Winit
        self.binit = binit
        self.nonlinearity = nonlinearity
        self.has_learned_params = True

    def to_dict(self):
        outdict = super(DenseLayer, self).to_dict()
        outdict['Winit'] = self.Winit.to_dict()
        outdict['binit'] = self.binit.to_dict()
        outdict['nonlinearity'] = self.nonlinearity.to_dict()
        return outdict

    def instantiate(self, instantiated_layers, layer_name):
        layer_obj = lasagne.layers.base.DenseLayer(
            input_layer=instantiated_layers[self.parent],
            num_units=self.num_units,
            W=self.Winit.instantiate(),
            b=self.binit.instantiate(),
            nonlinearity=self.nonlinearity.instantiate()
        )
        instantiated_layers[layer_name] = layer_obj
        return layer_obj

class MultipleParentsLayer(Layer):
    def __init__(self, parents, **kwargs):
        if not isinstance(parents, list):
            raise TypeError("parents must be a list of strings.")
        super(MultipleParentsLayer, self).__init__(**kwargs)
        self.parents = parents

    def to_dict(self):
        properties = super(MultipleParentsLayer, self).to_dict()
        return properties

class ConcatLayer(MultipleParentsLayer):
    def __init__(self, parents, axis=1, **kwargs):
        if not isinstance(axis, int):
            raise TypeError("axis must be 1 or 0!")
        super(ConcatLayer, self).__init__(parents, **kwargs)
        self.axis = axis

    def instantiate(self, instantiated_layers, layer_name):
        layer_obj = lasagne.layers.base.ConcatLayer(
            input_layers=[instantiated_layers[parent] for parent in self.parents],
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
        layer_obj = lasagne.layers.base.ElemwiseSumLayer(
            input_layers=[instantiated_layers[parent] for parent in self.parents],
            nonlinearity=self.nonlinearity.instantiate(),
            coeffs=self.coeffs
        )
        instantiated_layers[layer_name] = layer_obj
        return layer_obj


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

    def to_dict(self):
        outdict = super(DropoutLayer, self).to_dict()
        return outdict

    def instantiate(self,instantiated_layers, layer_name):
        layer_obj = lasagne.layers.base.DropoutLayer(
            self.parent,
            p=self.p,
            rescale=self.rescale
        )
        return layer_obj