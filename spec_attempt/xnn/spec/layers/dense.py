from ..init import *
from ..nonlinearities import *
from copy import deepcopy
from numbers import Number
# import lasagne.layers.base

from .base import *
'''
Have:
-Dense

Need:
-Local
-NIN
-elemwise
'''

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
        self.num_units          = num_units
        self.Winit              = Winit
        self.binit              = binit
        self.nonlinearity       = nonlinearity
        self.has_learned_params = True

    def to_dict(self):
        outdict = super(DenseLayer, self).to_dict()
        outdict['Winit'] = self.Winit.to_dict()
        outdict['binit'] = self.binit.to_dict()
        outdict['nonlinearity'] = self.nonlinearity.to_dict()
        return outdict

    def instantiate(self, instantiated_layers, layer_ctr):
        layer_obj = lasagne.layers.dense.DenseLayer(
            incoming=instantiated_layers[self.parent],
            num_units=self.num_units,
            W=self.Winit.instantiate(),
            b=self.binit.instantiate(),
            nonlinearity=self.nonlinearity.instantiate()
        )
        instantiated_layers[layer_ctr] = layer_obj
        return layer_obj