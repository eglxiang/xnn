from spec.initializerspec import *
from spec.activationspec import *
from copy import deepcopy
from numbers import Number

class layerSpec(object):
    def __init__(self, name, **kwargs):
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        if kwargs:
            self.additional_args = kwargs
        self.name = name
        self.type = self.__class__.__name__

    def to_dict(self):
        properties = deepcopy(self.__dict__)
        del(properties['name'])
        del(properties['type'])
        outdict = dict(
            name=self.name,
            type=self.type,
            properties=properties
        )
        return outdict

class inputLayerSpec(layerSpec):
    def __init__(self, name, shape=(48,48), **kwargs):
        if not isinstance(shape, (list, tuple)) \
                or any(not isinstance(x, Number) for x in shape):
            raise TypeError("shape must be a list or tuple of integers!")
        super(inputLayerSpec, self).__init__(name=name, **kwargs)
        self.shape = shape

class singleParentLayerSpec(layerSpec):
    def __init__(self, name, parent, **kwargs):
        if not isinstance(parent, str):
            raise TypeError("parent for singleParentLayerSpec type layers"
                            " must be a single string referring to a layer name.")
        super(singleParentLayerSpec, self).__init__(name, **kwargs)
        self.parent = parent

class denseLayerSpec(singleParentLayerSpec):
    def __init__(self, name, parent, num_units, Winit=UniformSpec(), binit=ConstantSpec(), nonlinearity=linearSpec(), **kwargs):
        if not isinstance(Winit, initializerSpec):
            raise TypeError("Winit must be an object of type initializerSpec!")

        if not isinstance(binit, initializerSpec):
            raise TypeError("binit must be an object of type initializerSpec!")

        if not isinstance(nonlinearity, activationSpec):
            raise TypeError("nonlinearity must be an object of type activationSpec!")

        super(denseLayerSpec, self).__init__(name, parent, **kwargs)
        self.num_units = num_units
        self.Winit = Winit
        self.binit = binit
        self.nonlinearity = nonlinearity

    def to_dict(self):
        outdict = super(denseLayerSpec, self).to_dict()
        outdict['properties']['Winit'] = self.Winit.to_dict()
        outdict['properties']['binit'] = self.binit.to_dict()
        outdict['properties']['nonlinearity'] = self.nonlinearity.to_dict()
        return outdict

class MultipleParentsLayerSpec(layerSpec):
    def __init__(self, name, parents, **kwargs):
        if not isinstance(parents, list):
            raise TypeError("parents for MultipleParentsLayerSpec type layers"
                            " must be a list of strings referring to layer names.")
        super(MultipleParentsLayerSpec, self).__init__(name, **kwargs)
        self.parents = parents

class concatLayerSpec(MultipleParentsLayerSpec):
    def __init__(self, name, parents, axis=1, **kwargs):
        if not isinstance(axis, int):
            raise TypeError("axis must be 1 or 0!")
        super(concatLayerSpec, self).__init__(name, parents, **kwargs)
        self.axis = axis

class ElemwiseSumLayerSpec(MultipleParentsLayerSpec):
    def __init__(self, name, parents, coeffs=1, **kwargs):
        if not isinstance(coeffs, Number):
            raise TypeError("coeffs must be a number!")
        super(ElemwiseSumLayerSpec, self).__init__(name, parents, **kwargs)
        self.coeffs = coeffs
