from ..settings import *
# from spec.regularization import *
from ..data.manager import *

from copy import deepcopy


class Trainer(object):
    def __init__(self, model, data_manager=DataManager(), default_settings=Settings()):
        self.model = model
        self.data_manager = data_manager
        self.default_settings = default_settings

    def to_dict(self):
        properties = deepcopy(self.__dict__)
        properties['model'] = properties['model'].to_dict()
        properties['data_manager'] = properties['data_manager'].to_dict()
        properties['default_settings'] = properties['default_settings'].to_dict()
        return properties