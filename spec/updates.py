from spec.adjusters import *
import lasagne.updates

LR_DEFAULT = ConstantVal(start=0.01)
MOM_DEFAULT = ConstantVal(start=0.5)

class Updater(object):
    def __init__(self, **kwargs):
        if kwargs:
            self.additional_args = kwargs
        self.type = self.__class__.__name__

    def to_dict(self):
        properties = deepcopy(self.__dict__)
        return properties

class SGD(Updater):
    def __init__(self, learning_rate=LR_DEFAULT, **kwargs):
        if not isinstance(learning_rate, Adjuster):
            raise TypeError("learning_rate must be an object of type Adjuster.")
        super(SGD, self).__init__(**kwargs)
        self.learning_rate = learning_rate

    def to_dict(self):
        properties = super(SGD, self).to_dict()
        properties['learning_rate'] = properties['learning_rate'].to_dict()
        return properties

    def __call__(self, loss, params, epoch, **kwargs):
        masks = None # TODO: Deal with masks!
        learning_rate = self.learning_rate(epoch)
        return lasagne.updates.sgd(loss, params, masks, learning_rate, **kwargs)

class Momentum(SGD):
    def __init__(self, learning_rate=LR_DEFAULT, momentum=MOM_DEFAULT, **kwargs):
        if not isinstance(momentum, Adjuster):
            raise TypeError("momentum must be an object of type Adjuster.")
        super(Momentum, self).__init__(learning_rate=learning_rate, **kwargs)
        self.momentum = momentum

    def to_dict(self):
        properties = super(Momentum, self).to_dict()
        properties['momentum'] = properties['momentum'].to_dict()
        return properties

    def __call__(self, loss, params, epoch, **kwargs):
        masks = None # TODO: Deal with masks!
        learning_rate = self.learning_rate(epoch)
        momentum = self.momentum(epoch)
        return lasagne.updates.momentum(loss, params, masks, learning_rate, momentum, **kwargs)

class NesterovMomentum(Momentum):
    def __init__(self, learning_rate=LR_DEFAULT, momentum=MOM_DEFAULT, **kwargs):
        super(NesterovMomentum, self).__init__(
            learning_rate=learning_rate, momentum=momentum, **kwargs)

    def __call__(self, loss, params, epoch, **kwargs):
        masks = None # TODO: Deal with masks!
        activation_params = None # TODO: Deal with activation_params!
        learning_rate = self.learning_rate(epoch)
        momentum = self.momentum(epoch)
        return lasagne.updates.nesterov_momentum(loss,
                                                 params,
                                                 activation_params,
                                                 masks,
                                                 learning_rate,
                                                 momentum, **kwargs)
