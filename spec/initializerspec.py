from numbers import Number
from copy import deepcopy
import lasagne.init as init

class initializerSpec(object):
    def __init__(self, **kwargs):
        if kwargs:
            self.additional_args = kwargs
        self.type = self.__class__.__name__

    def to_dict(self):
        properties = deepcopy(self.__dict__)
        return properties

class UniformSpec(initializerSpec):
    def __init__(self, range=None, **kwargs):
        if range and not isinstance(range, (list, tuple)):
            raise TypeError("range must be a list or tuple of numbers or None.")
        elif range and any(not isinstance(x, Number) for x in range):
            raise TypeError("range must be a list or tuple of numbers or None.")
        super(UniformSpec, self).__init__(**kwargs)
        self.range = range

    def instantiate(self):
        return init.Uniform(range=self.range)

class ConstantSpec(initializerSpec):
    def __init__(self, val=None, **kwargs):
        if val and not isinstance(val, Number):
            raise TypeError("val must be a number or None.")
        super(ConstantSpec, self).__init__(**kwargs)
        self.val = val

    def instantiate(self):
        return init.Constant(val=self.val)
