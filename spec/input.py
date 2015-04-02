from copy import deepcopy

class Input(object):
    def __init__(self, input_name, **kwargs):
        if not isinstance(input_name, str):
            raise TypeError("input_name must be a string.")
        if kwargs:
            self.additional_args = kwargs
        self.input_name = input_name
    #
    # def to_dict(self):
    #     indict = deepcopy(self.__dict__)
    #     return indict
    #
    # def instantiate(self, instantiated_layers):
    #     input_settings = dict(
    #         input_variable=
    #     )
    #     return input_settings
    #
    #
    #
    # def instantiate(self, instantiated_layers):
    #     target_obj = None
    #     if isinstance(self.target, ChannelsTarget):
    #         target_obj = self.target.channelsets
    #     elif isinstance(self.target, ReconstructionTarget):
    #         target_obj = instantiated_layers[self.target.layer]
    #     output_settings = dict(
    #         loss=self.loss.instantiate(),
    #         runtime_nonlinearity=self.runtime_nonlinearity.instantiate(),
    #         scale=self.scale,
    #         target=target_obj
    #     )
    #     return output_settings