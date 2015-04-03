from numbers import Number
from copy import deepcopy
import lasagne.init

class Initializer(object):
    def __init__(self, **kwargs):
        if kwargs:
            self.additional_args = kwargs
        self.type = self.__class__.__name__

    def to_dict(self):
        properties = deepcopy(self.__dict__)
        return properties

class Uniform(Initializer):
    def __init__(self, range=None, **kwargs):
        if range and not isinstance(range, (list, tuple)):
            raise TypeError("range must be a list or tuple of numbers or None.")
        elif range and any(not isinstance(x, Number) for x in range):
            raise TypeError("range must be a list or tuple of numbers or None.")
        super(Uniform, self).__init__(**kwargs)
        self.range = range

    def instantiate(self):
        return lasagne.init.Uniform(range=self.range)

class Constant(Initializer):
    def __init__(self, val=None, **kwargs):
        if val and not isinstance(val, Number):
            raise TypeError("val must be a number or None.")
        super(Constant, self).__init__(**kwargs)
        self.val = val

    def instantiate(self):
        return lasagne.init.Constant(val=self.val)

class Normal(Initializer):
    def __init__(self, std=0.01, avg=0.0, **kwargs):
        if not isinstance(std, Number):
            raise TypeError("std must be a number.")
        if not isinstance(avg, Number):
            raise TypeError("avg must be a number.")
        super(Normal, self).__init__(**kwargs)
        self.std = std
        self.avg = avg

    def instantiate(self):
        return lasagne.init.Normal(std=self.std, avg=self.avg)

class Sparse(Initializer):
    def __init__(self, std=0.01, sparsity=0.1, **kwargs):
        if not isinstance(std, Number):
            raise TypeError("std must be a number.")
        if not isinstance(sparsity, Number):
            raise TypeError("sparsity must be a number.")
        super(Sparse, self).__init__(**kwargs)
        self.std = std
        self.sparsity = sparsity

    def instantiate(self):
        return lasagne.init.Sparse(std=self.std, avg=self.avg)

class Orthogonal(Initializer):
    def __init__(self, gain=1.0, **kwargs):
        if not isinstance(gain, Number) and gain!='relu':
            raise TypeError("Gain must be a number or 'relu'.")

        super(Orthogonal, self).__init__(**kwargs)
        self.gain = gain

    def instantiate(self):
        return lasagne.init.Orthogonal(gain=self.gain)
