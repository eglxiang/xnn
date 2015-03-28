from spec.activationspec import *
from spec.objectivespec import *
from spec.layerspec import *

from lasagne.utils import Separator

from copy import deepcopy

class TargetSpec(object):
    def to_dict(self):
        outdict = deepcopy(self.__dict__)
        outdict['type'] = self.__class__.__name__
        return outdict

class ChannelsTargetSpec(TargetSpec):
    def __init__(self, separator=None):
        # Separator(channels=CHANNELS,logic=any,separate=[['Group','2']],include=False)
        if separator and not isinstance(separator, Separator):
            raise TypeError("settings for ChannelsTargetSpec must be a Separator object or None.")
        self.separator = separator

class ReconstructionTargetSpec(TargetSpec):
    def __init__(self, layer):
        if not isinstance(layer, dict):
            raise TypeError("parent for ReconstructionTargetSpec must be a single dict "
                            "containing a key='layer' with value specifying a layerSpec object.")
        self.layer = layer

    def to_dict(self):
        outdict = super(ReconstructionTargetSpec, self).to_dict()
        outdict['layer'] = outdict['layer']['name']
        return outdict

class outputSpec(object):
    # TODO: Allow for scheduled scale
    def __init__(self, loss=crossentropySpec(), eval_output_activation=linearSpec(), scale=1.0, target=ChannelsTargetSpec(), **kwargs):
        if kwargs:
            self.additional_args = kwargs
        self.loss = loss
        self.eval_output_activation = eval_output_activation
        self.scale = scale
        self.target = target
        self.type = self.__class__.__name__

    def to_dict(self):
        outdict = deepcopy(self.__dict__)
        outdict['loss'] = outdict['loss'].to_dict()
        outdict['eval_output_activation'] = outdict['eval_output_activation'].to_dict()
        outdict['target'] = outdict['target'].to_dict()

        return outdict
