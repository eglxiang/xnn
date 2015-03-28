from copy import deepcopy
import lasagne.nonlinearities as nonlinearities

class Activation(object):
    def __init__(self, **kwargs):
        if kwargs:
            self.additional_args = kwargs
        self.type = self.__class__.__name__

    def to_dict(self):
        properties = deepcopy(self.__dict__)
        return properties

class linear(Activation):
    def __init__(self, **kwargs):
        super(linear, self).__init__(**kwargs)

    def instantiate(self):
        return nonlinearities.linear

class tanh(Activation):
    def __init__(self, **kwargs):
        super(tanh, self).__init__(**kwargs)

    def instantiate(self):
        return nonlinearities.tanh

class sigmoid(Activation):
    def __init__(self, **kwargs):
        super(sigmoid, self).__init__(**kwargs)

    def instantiate(self):
        return nonlinearities.sigmoid

class rectify(Activation):
    def __init__(self, **kwargs):
        super(rectify, self).__init__(**kwargs)

    def instantiate(self):
        return nonlinearities.rectify

class softmax(Activation):
    def __init__(self, **kwargs):
        super(softmax, self).__init__(**kwargs)

    def instantiate(self):
        return nonlinearities.softmax

class LeakyRectify(Activation):
    def __init__(self, leakiness=0.01, **kwargs):
        super(LeakyRectify, self).__init__(**kwargs)
        self.leakiness = leakiness

    def instantiate(self):
        return None
