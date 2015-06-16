from .adjusters import *
from .updates import *
from .regularization import *
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
                 lr_scale=1.0,
                 mom_scale=1.0,
                 wc_scale=1.0,
                 **kwargs):
        super(ParamUpdateSettings, self).__init__(**kwargs)
        self.lr_scale = lr_scale
        self.mom_scale = mom_scale
        self.wc_scale = wc_scale

    def to_dict(self):
        properties = super(ParamUpdateSettings, self).to_dict()
        return properties

class TrainerSettings(Settings):
    def __init__(self,
                 update=NesterovMomentum(learning_rate=LR_DEFAULT, momentum=MOM_DEFAULT),
                 weightcost=L2(),
                 max_epochs=100,
                 **kwargs):
        super(TrainerSettings, self).__init__(**kwargs)
        self.update = update
        self.weightcost = weightcost
        self.max_epochs = max_epochs

    def to_dict(self):
        properties = super(TrainerSettings, self).to_dict()
        properties['update'] = properties['update'].to_dict()
        properties['weightcost'] = properties['weightcost'].to_dict()
        return properties