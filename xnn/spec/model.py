from .output import *
from .layers import base
from .settings import *
from copy import deepcopy
import sys

class LayerContainer():
    def __init__(self, name, layer, type, output_settings=None, update_settings=None):
        self.name = name
        self.layer = layer
        self.type = type
        self.output_settings = output_settings
        self.update_settings = update_settings

class Model(object):

    def __init__(self, **kwargs):
        if kwargs:
            self.additional_args = kwargs
        self.layers = []
        self.instantiated_layers = dict()
        self.channelsets = []

    def to_dict(self):
        modeldict = deepcopy(self.__dict__)
        if 'instantiated_layers' in modeldict:
            del(modeldict['instantiated_layers'])
        modeldict['outputs'] = []
        modeldict['updates'] = []
        modeldict['channelsets'] = [cs.to_dict() for cs in self.channelsets]
        for i in range(len(modeldict['layers'])):
            layer_name = modeldict['layers'][i].name
            layer = modeldict['layers'][i].layer
            output_settings = modeldict['layers'][i].output_settings
            update_settings = modeldict['layers'][i].update_settings
            modeldict['layers'][i] = layer.to_dict()
            modeldict['layers'][i]['name'] = layer_name
            if output_settings:
                output_settings = output_settings.to_dict()
                output_settings['layer_name'] = layer_name
                modeldict['outputs'].append(output_settings)
            if update_settings:
                update_settings = update_settings.to_dict()
                update_settings['layer_name'] = layer_name
                modeldict['updates'].append(update_settings)
        return modeldict

    def bind_output(self, layername, settings=Output()):
        if not isinstance(settings, Output):
            raise TypeError("settings must be an object of type Output.")
        layerctr = self.get_layer(layername)
        layerctr.output_settings = settings

    def bind_param_update_settings(self, layername, settings=ParamUpdateSettings()):
        if not isinstance(settings, ParamUpdateSettings):
            raise TypeError("settings must be an object of type ParamUpdateSettings")
        layerctr = self.get_layer(layername)
        if layerctr.layer.has_learned_params:
            layerctr.update_settings = settings
        else:
            layertype = layerctr.layer.type
            raise RuntimeError("Cannot bind param update settings to a layer of type %s!" % layertype)

    def add_channel_set(self, channelset):
        if not isinstance(channelset, ChannelSet):
            raise TypeError("channelset must be an object of type ChannelSet.")
        self.channelsets.append(channelset)

    def add(self, layer_spec, name=None):
        if not isinstance(layer_spec, base.Layer):
            raise TypeError("layer_spec must be an object of type Layer!")
        if len(self.layers)==0:
            if not isinstance(layer_spec, base.InputLayer):
                raise TypeError("The first layer_spec added to the model must be an object of type InputLayer!")
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

        # create layer container with extra information used to keep track of layers in a graph
        layercontainer = LayerContainer(
            layer=layer_spec,
            name=name,
            type=layer_spec.__class__.__name__
        )
        self.layers.append(layercontainer)
        return layercontainer

    def first(self):
        return self.layers[0]

    def last(self):
        return self.layers[-1]

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

    def instantiate(self):
        layers = []
        for layerctr in self.layers:
            layername = layerctr.name
            layerobj = layerctr.layer.instantiate(self.instantiated_layers, layername)
            layerdata = dict(
                layer=layerobj,
                name=layername
            )
            if layerctr.output_settings:
                outputobj = layerctr.output_settings.instantiate(self.instantiated_layers)
                layerdata['output_settings']=outputobj
            if layerctr.update_settings:
                # TODO: does this need to change?
                updateobj = layerctr.update_settings#.instantiate(self.instantiated_layers)
                layerdata['update_settings']=updateobj
            layers.append(layerdata)
        return layers
