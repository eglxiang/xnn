from copy import deepcopy
import math

class Adjuster(object):
    def __init__(self, **kwargs):
        if kwargs:
            self.additional_args = kwargs
        self.type = self.__class__.__name__

    def to_dict(self):
        properties = deepcopy(self.__dict__)
        return properties

    def __call__(self, epoch):
        return None

'''
TODO: Add following adjusters:

-annealed (1/t)
-monitoring (function based)
    -eg: if validation error plateaus, decrease learning rate

NOTE:
-want to be able to adjust per epoch, per gradient update, ...
-include start/stop epochs
'''

class ConstantVal(Adjuster):
    def __init__(self, start=0.9, **kwargs):
        super(ConstantVal, self).__init__(**kwargs)
        self.start = start

    def __call__(self, epoch):
        return self.start

class LinearChange(Adjuster):
    def __init__(self, start=0.9, stop=0.99, interval=500, **kwargs):
        super(LinearChange, self).__init__(**kwargs)
        self.start    = start
        self.stop     = stop
        self.interval = interval

    def __call__(self, epoch):
        if epoch < self.interval:
            return epoch * (self.stop - self.start)/self.interval + self.start
        else:
            return self.stop

class ExponentialChange(Adjuster):
    def __init__(self, start=0.9, change=0.99, interval=None, **kwargs):
        super(ExponentialChange, self).__init__(**kwargs)
        self.start    = start
        self.change   = change
        self.interval = interval

    def __call__(self, epoch):
        if self.interval is None:
            return self.start * (self.change ** epoch)
        else:
            return self.start * (self.change ** min(self.interval,epoch))

class StepChange(Adjuster):
    def __init__(self, start=0.9, interval=500, base=10, steps=5, **kwargs):
        super(StepChange, self).__init__(**kwargs)
        self.start    = start
        self.interval = interval
        self.base     = base
        self.steps    = steps

    def __call__(self, epoch):
        return self.start / self.base ** \
            (int(min(epoch,self.interval)/(math.ceil(self.interval/self.steps))))


class Anneal(Adjuster):
    def __init__(self, start=0.9, anneal_coef=1, interval=None, **kwargs):
        super(Anneal, self).__init__(**kwargs)
        self.start       = start
        self.anneal_coef = anneal_coef
        self.interval    = interval

    def __call__(self, epoch):
        if self.interval is None:
            return self.start/(self.anneal_coef*epoch)
        else:
            return self.start/(self.anneal_coef*min(self.interval,epoch))

# This adjuster specifies precise epochs to change the parameter by precise amounts
# e.g. changepoints = [(250, .5), (500, .1)] would mean change the param to .5 at epoch 250
#      and then change it to .1 at 500
class ChangePoints(Adjuster):
    def __init__(self, start=0.9, changepoints=None, **kwargs):
        super(ChangePoints, self).__init__(**kwargs)
        self.start = start
        self.changepoints=changepoints

    def __call__(self, epoch):
        value = self.start
        for changepoint in sorted(self.changepoints):
            if epoch >= changepoint[0]:
                value = changepoint[1]
        return value
