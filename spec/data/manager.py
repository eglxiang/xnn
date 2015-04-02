from ..output import *
from copy import deepcopy

class DataManager(object):
    def __init__(self, input_names=None, channel_sets=None, batch_size=128, shuffle_batches=False):
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
        properties['channel_sets'] = [cs.to_dict() for cs in properties['channel_sets']]
        return properties
