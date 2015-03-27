from spec.layerspec import *
from copy import deepcopy
import sys

class modelSpec(object):
    def __init__(self, **kwargs):
        if kwargs:
            self.additional_args = kwargs
        self.layers = []

    def to_dict(self):
        modeldict = deepcopy(self.__dict__)
        for i in range(len(modeldict['layers'])):
            layer_name = modeldict['layers'][i]['name']
            layer_type = modeldict['layers'][i]['type']
            layer = modeldict['layers'][i]['layer']
            modeldict['layers'][i] = dict(
                name=layer_name,
                type=layer_type,
                properties=layer.to_dict()
            )
        return modeldict

    def add(self, layer_spec, name=None):
        if not isinstance(layer_spec, layerSpec):
            raise TypeError("layer_spec must be an object of type layerSpec!")
        if len(self.layers)==0:
            if not isinstance(layer_spec, inputLayerSpec):
                raise TypeError("The first layer_spec added to the model must be an object of type inputLayerSpec!")
        if name is not None and not isinstance(name, str):
            raise TypeError("name is an optional argument that must be a string or None. "
                            "Otherwise name is auto-generated.")

        # auto generate layer name if not passed in
        if name is None:
            layernum = len(self.layers)
            name = layer_spec.__class__.__name__.replace('Spec', '') + '_%d' % layernum
        # make sure name is unique
        layer_names = self.get_layer_names()
        if name in layer_names:
            raise RuntimeError("Layer name %s is already used by the model. "
                               "Each layer must have a unique name!" % name)

        # create layer dict with extra information used to keep track of layers in a graph
        layerdict = dict(
            layer=layer_spec,
            name=name,
            type=layer_spec.__class__.__name__
        )

        self.layers.append(layerdict)

        return layerdict

    def first(self):
        return self.layers[0]

    def last(self):
        return self.layers[-1]

    def get_layer_names(self):
        layer_names = [layer['name'] for layer in self.layers]
        return layer_names

    def get_layer(self, name):
        layer_names = self.get_layer_names()
        if name not in layer_names:
            print >>sys.stderr, "Name '%s' is not in layer_names" % name
            return None
        else:
            return self.layers[layer_names.index(name)]
