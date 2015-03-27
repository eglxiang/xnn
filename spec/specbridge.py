from spec.initializerspec import *
from spec.activationspec import *
from spec.layerspec import *

from lasagne.init import *
from lasagne.nonlinearities import *
from lasagne.layers.base import *

spec_to_map = {
    # Initializers
    'UniformSpec': Uniform,
    'ConstantSpec': Constant,

    # Nonlinearities
    'linearSpec': linear,
    'tanhSpec': tanh,
    'leakyRectifySpec': None,

    # Layers
    'inputLayerSpec': InputLayer,
    'denseLayerSpec': DenseLayer,
    'concatLayerSpec': ConcatLayer,
    'ElemwiseSumLayerSpec': ElemwiseSumLayer
}