from .objectives import *
from .nonlinearities import *
from .data.channels import *
from .layers.base import LayerContainer

# from lasagne.utils import Separator

from copy import deepcopy

class Target(object):
    def to_dict(self):
        outdict = deepcopy(self.__dict__)
        outdict['type'] = self.__class__.__name__
        return outdict

class ChannelsTarget(Target):
    def __init__(self, channelsets):#, separator=None):
        if not isinstance(channelsets, list):
            raise TypeError("channelsets must be a list of objects of type ChannelSet.")
        # Separator(channels=CHANNELS,logic=any,separate=[['Group','2']],include=False)
        # if separator and not isinstance(separator, Separator):
        #     raise TypeError("settings for ChannelsTarget must be a Separator object or None.")
        self.channelsets = channelsets
        # self.separator = separator

    def to_dict(self):
        outdict = super(ChannelsTarget, self).to_dict()
        outdict['channelsets'] = [channelset.to_dict() for channelset in outdict['channelsets']]
        return outdict

class ReconstructionTarget(Target):
    def __init__(self, layerctr):
        if not isinstance(layerctr, LayerContainer):
            raise TypeError("layer for ReconstructionTarget must be a LayerContainer.")
        self.layerctr = layerctr

    def to_dict(self):
        outdict = super(ReconstructionTarget, self).to_dict()
        outdict['layerctr'] = outdict['layerctr'].name
        return outdict

class Output(object):
    # TODO: Allow for scheduled scale
    def __init__(self, loss=crossentropy(), runtime_nonlinearity=linear(), scale=1.0, target=None, **kwargs):
        if kwargs:
            self.additional_args = kwargs
        self.loss = loss
        self.runtime_nonlinearity = runtime_nonlinearity
        self.scale = scale
        self.target = target

    def to_dict(self):
        outdict = deepcopy(self.__dict__)
        outdict['loss'] = outdict['loss'].to_dict()
        outdict['runtime_nonlinearity'] = outdict['runtime_nonlinearity'].to_dict()
        outdict['target'] = outdict['target'].to_dict()
        if isinstance(self.target, ChannelsTarget):
            outdict['target']['channelsets'] = [cs['name'] for cs in outdict['target']['channelsets']]
        return outdict

    def instantiate(self, instantiated_layers):
        target_obj = None
        if isinstance(self.target, ChannelsTarget):
            target_obj = self.target.channelsets
        elif isinstance(self.target, ReconstructionTarget):
            target_obj = instantiated_layers[self.target.layerctr]
        output_settings = dict(
            loss=self.loss.instantiate(),
            runtime_nonlinearity=self.runtime_nonlinearity.instantiate(),
            scale=self.scale,
            target=target_obj
        )
        return output_settings