from copy import deepcopy
from numbers import Number

class Channel(object):
    __defaults__ = dict(channel_weight=1.0)
    def __init__(self, name, channel_weight=__defaults__['channel_weight']):#negative_weight=1.0, ):
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        if not isinstance(channel_weight, Number):
            raise TypeError("channel_weight must be a number")
        self.name = name
        self.type = self.__class__.__name__
        # self.negative_weight = negative_weight
        self.channel_weight = channel_weight

    def to_dict(self):
        properties = deepcopy(self.__dict__)
        for key in self.__defaults__.keys():
            if properties[key] == self.__defaults__[key]:
                del(properties[key])
        return properties

class Binary(Channel):
    __defaults__ = dict(channel_weight=1.0, negative_weight=1.0)
    def __init__(self, name,
                 channel_weight=__defaults__['channel_weight'],
                 negative_weight=__defaults__['negative_weight']):
        if not isinstance(negative_weight, Number):
            raise TypeError("negative_weight must be a number")
        super(Binary, self).__init__(name, channel_weight)
        self.negative_weight = negative_weight

class Real(Channel):
    __defaults__ = dict(channel_weight=1.0, bins_to_weight=None)
    def __init__(self, name, channel_weight=__defaults__['channel_weight'],
                 bins_to_weight=__defaults__['bins_to_weight']):
        super(Real, self).__init__(name, channel_weight)
        if bins_to_weight:
            if not isinstance(bins_to_weight, list) \
                    or any(not isinstance(x, Number) for x in bins_to_weight):
                raise TypeError("bins_to_weight must be a list of right bin edges")
        self.bins_to_weight = bins_to_weight

class ChannelSet(object):
    __defaults__ = dict(set_weight=1.0)
    def __init__(self, name, type, set_weight=__defaults__['set_weight']):
        self.channels = []
        self.name = name
        self.type = type
        self.set_weight = set_weight

    def add(self, channel):
        if not isinstance(channel, Channel):
            raise TypeError("channel must be an object of type Channel.")
        self.channels.append(channel)

    def get_channel_names(self):
        return [channel.name for channel in self.channels]

    def get_channel(self, name):
        ind = self.get_channel_names().index(name)
        return self.channels[ind]

    def size(self):
        return len(self.channels)

    def to_dict(self):
        properties = deepcopy(self.__dict__)
        for key in self.__defaults__.keys():
            if properties[key] == self.__defaults__[key]:
                del(properties[key])
        properties['channels'] = [channel.to_dict() for channel in properties['channels']]
        return properties
