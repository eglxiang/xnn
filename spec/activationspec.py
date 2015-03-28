from copy import deepcopy
import lasagne.nonlinearities as nonlinearities

class activationSpec(object):
    def __init__(self, **kwargs):
        if kwargs:
            self.additional_args = kwargs
        self.type = self.__class__.__name__

    def to_dict(self):
        properties = deepcopy(self.__dict__)
        return properties

class linearSpec(activationSpec):
    def __init__(self, **kwargs):
        super(linearSpec, self).__init__(**kwargs)

    def instantiate(self):
        return nonlinearities.linear

class tanhSpec(activationSpec):
    def __init__(self, **kwargs):
        super(tanhSpec, self).__init__(**kwargs)

    def instantiate(self):
        return nonlinearities.tanh

class LeakyRectifySpec(activationSpec):
    def __init__(self, leakiness=0.01, **kwargs):
        super(LeakyRectifySpec, self).__init__(**kwargs)
        self.leakiness = leakiness

    def instantiate(self):
        return None
