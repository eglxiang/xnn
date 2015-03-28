from spec.initializerspec import *
from spec.activationspec import *
from copy import deepcopy
from numbers import Number
import lasagne.layers.base as layerbase

class layerSpec(object):
    def __init__(self, **kwargs):
        if kwargs:
            self.additional_args = kwargs
        self.type = self.__class__.__name__

    def to_dict(self):
        properties = deepcopy(self.__dict__)
        return properties

class inputLayerSpec(layerSpec):
    def __init__(self, shape=(48,48), **kwargs):
        if not isinstance(shape, (list, tuple)) \
                or any(not isinstance(x, Number) for x in shape):
            raise TypeError("shape must be a list or tuple of integers!")
        super(inputLayerSpec, self).__init__(**kwargs)
        self.shape = shape

    def instantiate(self, instantiated_layers, layer_name):
        layer_obj = layerbase.InputLayer(shape=self.shape)
        instantiated_layers[layer_name] = layer_obj
        return layer_obj

class singleParentLayerSpec(layerSpec):
    def __init__(self, parent, **kwargs):
        if not isinstance(parent, str):
            raise TypeError("parent for singleParentLayerSpec type layers must be "
                            "a single string specifying the name of a layerSpec object.")
        super(singleParentLayerSpec, self).__init__(**kwargs)
        self.parent = parent

    def to_dict(self):
        properties = super(singleParentLayerSpec, self).to_dict()
        return properties

class denseLayerSpec(singleParentLayerSpec):
    def __init__(self, parent, num_units, Winit=UniformSpec(range=(0,1)), binit=ConstantSpec(val=0), nonlinearity=linearSpec(), **kwargs):
        if not isinstance(Winit, initializerSpec):
            raise TypeError("Winit must be an object of type initializerSpec!")
        if not isinstance(binit, initializerSpec):
            raise TypeError("binit must be an object of type initializerSpec!")
        if not isinstance(nonlinearity, activationSpec):
            raise TypeError("nonlinearity must be an object of type activationSpec!")

        super(denseLayerSpec, self).__init__(parent, **kwargs)
        self.num_units = num_units
        self.Winit = Winit
        self.binit = binit
        self.nonlinearity = nonlinearity

    def to_dict(self):
        outdict = super(denseLayerSpec, self).to_dict()
        outdict['Winit'] = self.Winit.to_dict()
        outdict['binit'] = self.binit.to_dict()
        outdict['nonlinearity'] = self.nonlinearity.to_dict()
        return outdict

    def instantiate(self, instantiated_layers, layer_name):
        layer_obj = layerbase.DenseLayer(
            input_layer=instantiated_layers[self.parent],
            num_units=self.num_units,
            W=self.Winit.instantiate(),
            b=self.binit.instantiate(),
            nonlinearity=self.nonlinearity.instantiate()
        )
        instantiated_layers[layer_name] = layer_obj
        return layer_obj

class MultipleParentsLayerSpec(layerSpec):
    def __init__(self, parents, **kwargs):
        if not isinstance(parents, list):
            raise TypeError("parents for MultipleParentsLayerSpec type layers must be "
                            "a list of strings each specifying the name of "
                            "a layerSpec object.")
        super(MultipleParentsLayerSpec, self).__init__(**kwargs)
        self.parents = parents

    def to_dict(self):
        properties = super(MultipleParentsLayerSpec, self).to_dict()
        return properties

class concatLayerSpec(MultipleParentsLayerSpec):
    def __init__(self, parents, axis=1, **kwargs):
        if not isinstance(axis, int):
            raise TypeError("axis must be 1 or 0!")
        super(concatLayerSpec, self).__init__(parents, **kwargs)
        self.axis = axis

    def instantiate(self, instantiated_layers, layer_name):
        layer_obj = layerbase.ConcatLayer(
            input_layers=[instantiated_layers[parent] for parent in self.parents],
            axis=self.axis
        )
        instantiated_layers[layer_name] = layer_obj
        return layer_obj

class ElemwiseSumLayerSpec(MultipleParentsLayerSpec):
    def __init__(self, parents, coeffs=1, **kwargs):
        if not isinstance(coeffs, Number):
            raise TypeError("coeffs must be a number!")
        super(ElemwiseSumLayerSpec, self).__init__(parents, **kwargs)
        self.coeffs = coeffs

    def instantiate(self, instantiated_layers, layer_name):
        layer_obj = layerbase.ElemwiseSumLayer(
            input_layers=[instantiated_layers[parent] for parent in self.parents],
            nonlinearity=self.nonlinearity.instantiate(),
            coeffs=self.coeffs
        )
        instantiated_layers[layer_name] = layer_obj
        return layer_obj
