class activationSpec(object):
    def __init__(self, **kwargs):
        if kwargs:
            self.additional_args = kwargs

    def to_dict(self):
        return self.__dict__

class linearSpec(activationSpec):
    def __init__(self, **kwargs):
        super(linearSpec, self).__init__(**kwargs)

class tanhSpec(activationSpec):
    def __init__(self, **kwargs):
        super(tanhSpec, self).__init__(**kwargs)

class LeakyRectifySpec(activationSpec):
    def __init__(self, leakiness=0.01, **kwargs):
        super(LeakyRectifySpec, self).__init__(**kwargs)
        self.leakiness = leakiness
