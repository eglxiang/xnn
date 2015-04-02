from ..init import *
from ..nonlinearities import *
from copy import deepcopy
from numbers import Number
# import lasagne.layers.base
import lasagne.theano_extensions

from .base import *

'''
Have:
-Conv1DLayer
-Conv2DLayer

Need:
'''

class Convolution(object):
    def __init__(self):
        self.type = self.__class__.__name__

    def to_dict(self):
        properties = deepcopy(self.__dict__)
        return properties

    def instantiate(self):
        raise NotImplementedError

class conv1d_sc(Convolution):
    def __init__(self):
        super(conv1d_sc,self).__init__()

    def instantiate(self):
        return theano_extensions.conv1d_sc

class conv1d_mc0(Convolution):
    def __init__(self):
        super(conv1d_mc0,self).__init__()

    def instantiate(self):
        return theano_extensions.conv1d_mc0

class conv1d_mc1(Convolution):
    def __init__(self):
        super(conv1d_mc1,self).__init__()

    def instantiate(self):
        return theano_extensions.conv1d_mc1

class conv1d_unstrided(Convolution):
    def __init__(self):
        super(conv1d_unstrided,self).__init__()

    def instantiate(self):
        return theano_extensions.conv1d_unstrided

class conv1d_sd(Convolution):
    def __init__(self):
        super(conv1d_sd,self).__init__()

    def instantiate(self):
        return theano_extensions.conv1d_sd

class conv1d_md(Convolution):
    def __init__(self):
        super(conv1d_md,self).__init__()

    def instantiate(self):
        return theano_extensions.conv1d_md

class conv2d(Convolution):
    def __init__(self):
        super(conv1d_md,self).__init__()

    def instantiate(self):
        import theano.tensor as T
        return T.nnet.conv2d




class ConvLayer(SingleParentLayer):
    def __init__(self, parent, num_filters, filter_length,
                 border_mode, untie_biases, Winit,
                 binit, nonlinearity,
                 convolution, **kwargs):
        if not isinstance(num_filters, Number):
            raise TypeError("num_filters must be an integer!")
        if not isinstance(filter_length, Number):
            raise TypeError("filter_length must be an integer!")
        if not border_mode in ['valid','full','same']:
            raise TypeError("border_mode must be 'valid', 'full', or 'same'!")
        if not untie_biases in [True,False]:
            raise TypeError("untie_biases must be True or False!")
        if not isinstance(Winit, Initializer):
            raise TypeError("Winit must be an object of type Initializer!")
        if not isinstance(binit, Initializer):
            raise TypeError("binit must be an object of type Initializer!")
        if not isinstance(nonlinearity, Activation):
            raise TypeError("nonlinearity must be an object of type Activation!")
        if not isinstance(convolution, Convolution):
            raise TypeError("convolution must be an object of type Convolution!")

        super(ConvLayer, self).__init__(parent, **kwargs)
        self.num_filters        = num_filters
        self.filter_length      = filter_length
        self.border_mode        = border_mode
        self.untie_biases       = untie_biases
        self.Winit              = Winit
        self.binit              = binit
        self.nonlinearity       = nonlinearity
        self.convolution        = convolution
        self.has_learned_params = True

    def to_dict(self):
        outdict = super(ConvLayer, self).to_dict()
        outdict['Winit'] = self.Winit.to_dict()
        outdict['binit'] = self.binit.to_dict()
        outdict['nonlinearity'] = self.nonlinearity.to_dict()
        outdict['convolution'] = self.convolution.to_dict()
        return outdict

    def instantiate(self, instantiated_layers, layer_name):
        raise NotImplementedError

class Conv1DLayer(ConvLayer):
    def __init__(self, parent, num_filters, filter_length, stride=1,
                 border_mode="valid", untie_biases=False, Winit=Uniform(range=(0,1)),
                 binit=Constant(val=0), nonlinearity=linear(),
                 convolution=conv1d_mc0, **kwargs):
        if not isinstance(stride, Number):
            raise TypeError("stride must be an object of type Initializer!")
        self.stride=stride
        super(Conv1DLayer, self).__init__(parent, num_filters, filter_length,
                 border_mode, untie_biases, Winit,
                 binit, nonlinearity,
                 convolution, **kwargs)

    def instantiate(self, instantiated_layers, layer_name):
        layer_obj = lasagne.layers.conv.Conv1DLayer(
            incoming=instantiated_layers[self.parent],
            num_filters=self.num_filters,
            filter_length=self.filter_length,
            stride=self.stride,
            border_mode=self.border_mode,
            untie_biases=self.untie_biases,
            W=self.Winit.instantiate(),
            b=self.binit.instantiate(),
            nonlinearity=self.nonlinearity.instantiate(),
            convolution=self.convolution.instantiate()
        )
        instantiated_layers[layer_name] = layer_obj
        return layer_obj

class Conv2DLayer(ConvLayer):
    def __init__(self, parent, num_filters, filter_length, strides=(1,1),
                 border_mode="valid", untie_biases=False, Winit=Uniform(range=(0,1)),
                 binit=Constant(val=0), nonlinearity=linear(),
                 convolution=conv2d, **kwargs):
        if not isinstance(strides, tuple) or len(strides)!=2:
            raise TypeError("strides must be a tuple of length 2!")
        self.strides=strides
        super(Conv2DLayer, self).__init__(parent, num_filters, filter_length,
                 border_mode, untie_biases, Winit,
                 binit, nonlinearity,
                 convolution, **kwargs)

    def instantiate(self, instantiated_layers, layer_name):
        layer_obj = lasagne.layers.conv.Conv2DLayer(
            incoming=instantiated_layers[self.parent],
            num_filters=self.num_filters,
            filter_length=self.filter_length,
            strides=self.strides,
            border_mode=self.border_mode,
            untie_biases=self.untie_biases,
            W=self.Winit.instantiate(),
            b=self.binit.instantiate(),
            nonlinearity=self.nonlinearity.instantiate(),
            convolution=self.convolution.instantiate()
        )
        instantiated_layers[layer_name] = layer_obj
        return layer_obj

