from copy import deepcopy

class Channel(object):
    def __init__(self, name, negative_weight=1.0, channel_weight=1.0):
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        self.name = name
        self.negative_weight = negative_weight
        self.channel_weight = channel_weight

    def to_dict(self):
        return deepcopy(self.__dict__)

class ChannelSet(object):
    def __init__(self, name, type, type_weight=1.0):
        self.channels = []
        self.name = name
        self.type = type
        self.type_weight = type_weight

    def add(self, channel):
        self.channels.append(channel)

    def get_channel_names(self):
        return [channel.name for channel in self.channels]

    def get_channel(self, name):
        ind = self.get_channel_names().index(name)
        return self.channels[ind]

    def size(self):
        return len(self.channels)

    def to_dict(self):
        outdict = deepcopy(self.__dict__)
        outdict['channels'] = [channel.to_dict() for channel in outdict['channels']]
        return outdict
