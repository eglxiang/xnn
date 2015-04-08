from adjusters import *
from copy import deepcopy

import lasagne.regularization

class Regularizer(object):
    def __init__(self, **kwargs):
        if kwargs:
            self.additional_args = kwargs
        self.type = self.__class__.__name__

    def to_dict(self):
        properties = deepcopy(self.__dict__)
        return properties

    def __call__(self, layer, epoch, **kwargs):
        return None

class L2(Regularizer):
    def __init__(self, scale=ConstantVal(0), include_biases=False, **kwargs):
        super(L2, self).__init__(**kwargs)
        self.scale          = scale
        self.include_biases = include_biases

    def to_dict(self):
        properties = super(L2, self).to_dict()
        properties['scale'] = properties['scale'].to_dict()
        return properties

    def __call__(self, layer, epoch, **kwargs):
        return lasagne.regularization.l2(layer=layer, include_biases=self.include_biases)
