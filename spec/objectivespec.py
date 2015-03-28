from copy import deepcopy
import lasagne.objectives as objectives

class objectiveSpec(object):
    def __init__(self, **kwargs):
        if kwargs:
            self.additional_args = kwargs
        self.type = self.__class__.__name__

    def to_dict(self):
        properties = deepcopy(self.__dict__)
        return properties

class mseSpec(objectiveSpec):
    def __init__(self, **kwargs):
        super(mseSpec, self).__init__(**kwargs)

    def instantiate(self):
        return objectives.Objective(loss_function=objectives.mse)

class crossentropySpec(objectiveSpec):
    def __init__(self, **kwargs):
        super(crossentropySpec, self).__init__(**kwargs)

    def instantiate(self):
        return objectives.Objective(loss_function=objectives.crossentropy)

class categorical_crossentropySpec(objectiveSpec):
    def __init__(self, **kwargs):
        super(categorical_crossentropySpec, self).__init__(**kwargs)

    def instantiate(self):
        return objectives.Objective(loss_function=objectives.categorical_crossentropy)

class hinge_lossSpec(objectiveSpec):
    def __init__(self, threshold=0.0, **kwargs):
        super(hinge_lossSpec, self).__init__(**kwargs)
        self.threshold = threshold

    def instantiate(self):
        return objectives.Objective(loss_function=objectives.hinge_loss, threshold=self.threshold)

class squared_hinge_lossSpec(objectiveSpec):
    def __init__(self, gamma=2.0, **kwargs):
        super(squared_hinge_lossSpec, self).__init__(**kwargs)
        self.gamma = gamma

    def instantiate(self):
        return objectives.Objective(loss_function=objectives.squared_hinge_loss, gamma=self.gamma)

class kl_divergenceSpec(objectiveSpec):
    def __init__(self, eps=1e-08, **kwargs):
        super(kl_divergenceSpec, self).__init__(**kwargs)
        self.eps = eps

    def instantiate(self):
        return objectives.Objective(loss_function=objectives.kl_divergence, eps=self.eps)
