from spec.layerspec import *
from copy import deepcopy
import sys

class modelSpec(object):
    def __init__(self, name="", **kwargs):
        if kwargs:
            self.additional_args = kwargs
        self.name = name
        self.type = self.__class__.__name__
        self.layers = []

    def to_dict(self):
        modeldict = deepcopy(self.__dict__)
        for i in range(len(modeldict['layers'])):
            modeldict['layers'][i] = modeldict['layers'][i].to_dict()
        return modeldict

    def add(self, layer_spec):
        if not isinstance(layer_spec, layerSpec):
            raise TypeError("layer_spec must be an object of type layerSpec!")
        if len(self.layers)==0:
            if not isinstance(layer_spec, inputLayerSpec):
                raise TypeError("The first layer_spec added to the model must be an object of type inputLayerSpec!")
        layer_names = self.get_layer_names()
        if layer_spec.name in layer_names:
            raise RuntimeError("Layer name %s is already used by the model. "
                               "Each layer must have a unique name!" % layer_spec.name)
        self.layers.append(layer_spec)

    def get_layer_names(self):
        layer_names = [layer.name for layer in self.layers]
        return layer_names

    def get_layer(self, name):
        layer_names = self.get_layer_names()
        if name not in layer_names:
            print >>sys.stderr, "Name '%s' is not in layer_names" % name
            return None
        else:
            return self.layers[layer_names.index(name)]
