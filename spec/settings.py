from spec.adjusters import *
from spec.updates import *
from spec.regularization import *
from copy import deepcopy

class Settings(object):
    def __init__(self, **kwargs):
        if kwargs:
            self.additional_args = kwargs
        self.type = self.__class__.__name__

    def to_dict(self):
        return deepcopy(self.__dict__)

class ParamUpdateSettings(Settings):
    def __init__(self,
                 learning_rate=ConstantVal(0.01),
                 momentum=ConstantVal(0.5),
                 weightdecay=L2(),
                 **kwargs):
        super(ParamUpdateSettings, self).__init__(**kwargs)
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weightdecay = weightdecay

    def to_dict(self):
        properties = super(ParamUpdateSettings, self).to_dict()
        properties['learning_rate'] = properties['learning_rate'].to_dict()
        properties['momentum'] = properties['momentum'].to_dict()
        properties['weightdecay'] = properties['weightdecay'].to_dict()
        return properties

class TrainerSettings(ParamUpdateSettings):
    def __init__(self,
                 learning_rate=ConstantVal(0.01),
                 momentum=ConstantVal(0.5),
                 update=NesterovMomentum,
                 weightdecay=L2(),
                 epochs=100,
                 **kwargs):
        super(TrainerSettings, self).__init__(learning_rate, momentum, weightdecay, **kwargs)
        self.update = update
        self.epochs = epochs

    def to_dict(self):
        properties = deepcopy(self.__dict__)
        properties['update'] = properties['update'].__name__
        return properties