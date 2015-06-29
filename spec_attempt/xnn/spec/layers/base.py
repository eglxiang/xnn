from ..init import *
from ..nonlinearities import *
from copy import deepcopy
from numbers import Number
import lasagne.layers.base
# from spec.layers.conv import *

class LayerContainer():
    """
    A struct for keeping track of auxiliary information for a layer spec.
    """
    def __init__(self, name, layer, type, output_settings=None, update_settings=None):
        self.name = name
        self.layer = layer
        self.type = type
        self.output_settings = output_settings
        self.update_settings = update_settings

class Layer(object):
    """Base class for layer specification."""
    def __init__(self, **kwargs):
        """A Layer object will store any arguments that are not specifically
        expected for that layer in the additional_args field for custom purposes.
        The Layer object will also store its specific type in the type field.
        If a layer contains learned parameters its has_learned_params field will be True.
        """
        if kwargs:
            self.additional_args = kwargs
        self.type = self.__class__.__name__
        self.has_learned_params = False

    def to_dict(self):
        """The to_dict method serializes a Layer object to a vanilla python dictionary
        whose contents are directly portable to formats such as JSON.
        :return: dict containing object properties in standard readable python format.
        """
        properties = deepcopy(self.__dict__)
        return properties

class InputLayer(Layer):
    """The InputLayer class is meant for specifying input layers.
    There is no parent layer for input layers.
    This class handles property validation.
    """
    def __init__(self, shape=(48,48), **kwargs):
        """Requires a valid shape tuple. Throws an error if shape is not a tuple of numbers.
        :param shape: tuple of integers specifying input dimensions.
        """
        if not isinstance(shape, (list, tuple)) \
                or any(not isinstance(x, Number) for x in shape):
            raise TypeError("shape must be a list or tuple of integers!")
        super(InputLayer, self).__init__(**kwargs)
        self.shape = shape

    def instantiate(self, instantiated_layers, layer_ctr):
        """Instantiates a Lasagne InputLayer object with specified properties.
        :param instantiated_layers: A dictionary of <layer name>:<instantiated layer> objects.
        :param layer_ctr: A LayerContainer object.
        :return: Returns a Lasagne InputLayer object.
        """
        layer_obj = lasagne.layers.input.InputLayer(shape=self.shape)
        instantiated_layers[layer_ctr] = layer_obj
        return layer_obj

class SingleParentLayer(Layer):
    """The SingleParentLayer class is an interface for layer spec types
    who take one and only one parent as input, such as a DenseLayer.
    """
    def __init__(self, parent, **kwargs):
        """Throws an error if the parent is not a LayerContainer object.
        :param parent: LayerContainer object referring to this layer's parent.
        """
        if not isinstance(parent, LayerContainer):
            raise TypeError("parent must be a LayerContainer.")
        super(SingleParentLayer, self).__init__(**kwargs)
        self.parent = parent

    def to_dict(self):
        properties = super(SingleParentLayer, self).to_dict()
        properties['parent'] = self.parent.name
        return properties

class MultipleParentsLayer(Layer):
    """The MultipleParentsLayer class is an interface for layer spec types
    who take a list of parents as inputs, such as a ConcatLayer.
    """
    def __init__(self, parents, **kwargs):
        """Throws an error if the parents variable is not a list of LayerContainer objects.
        :param parents: list of LayerContainer objects referring to this layer's parents.
        """
        if not isinstance(parents, list) \
                or any(not isinstance(x, LayerContainer) for x in parents):
            raise TypeError("parents must be a list of LayerContainer objects.")
        super(MultipleParentsLayer, self).__init__(**kwargs)
        self.parents = parents

    def to_dict(self):
        properties = super(MultipleParentsLayer, self).to_dict()
        properties['parents'] = [parent.name for parent in self.parents]
        return properties
