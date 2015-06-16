from ..output import *
from copy import deepcopy

class DataManager(object):
    __defaults__ = dict(input_names=None,
                        channel_sets=None,
                        batch_size=128,
                        shuffle_batches=False)
    def __init__(self, input_names=__defaults__['input_names'],
                 channel_sets=__defaults__['channel_sets'],
                 batch_size=__defaults__['batch_size'],
                 shuffle_batches=__defaults__['shuffle_batches']):
        self.batch_size = batch_size
        self.shuffle_batches = shuffle_batches
        self.input_names = input_names if input_names else []
        self.channel_sets = channel_sets if channel_sets else []

    def add_channel_set(self, channel_set):
        if not isinstance(channel_set, ChannelSet):
            raise TypeError("channel_set must be an object of type ChannelSet")
        self.channel_sets.append(channel_set)

    def add_input_name(self, input_name):
        if not isinstance(input_name, str):
            raise TypeError("input_name must be a string.")
        self.input_names.append(input_name)

    def to_dict(self):
        properties = deepcopy(self.__dict__)
        for key in self.__defaults__.keys():
            if properties[key] == self.__defaults__[key]:
                del(properties[key])
        properties['channel_sets'] = [cs.to_dict() for cs in properties['channel_sets']]
        return properties
