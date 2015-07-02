import theano
import theano.tensor as T
import xnn
from ..utils import typechecker
from lasagne.objectives import binary_crossentropy,categorical_crossentropy


__all__ = [
    "absolute_error",
    "kl_divergence",
    "hinge_loss",
    "from_dict",
    "squared_hinge_loss",
    "binary_crossentropy",
    "categorical_crossentropy",
    "cross_covariance",
    "ScaledObjective"
]


binary_crossentropy = typechecker(binary_crossentropy)
categorical_crossentropy = typechecker(categorical_crossentropy)

def absolute_error(a, b):
    """Computes the element-wise absolute difference between two tensors.
    .. math:: L = abs(p - t)
    Parameters
    ----------
    a, b : Theano tensor
        The tensors to compute the absolute difference between.
    Returns
    -------
    Theano tensor
        An expression for the item-wise absolute difference.
    """
    return abs(a - b)

@typechecker
def kl_divergence(predictions, targets, eps=1e-08):
    """Computes the kl-divergence between predictions and targets
    (for categorical variables).
    Parameters
    ----------
    predictions : Theano 2D tensor
        Predictions from softmax output of a neural network,
        with data points in rows and class probabilities in columns.
    targets : Theano 2D tensor or 1D tensor
        Either targets in [0, 1] matching the layout of `predictions`, or
        a vector of int giving the correct class index per data point.
    eps : A constant for ensuring that the kl_divergence doesn't blow up.
    Returns
    -------
    Theano 1D tensor
        An expression for the item-wise kl-divergence.
    """

    lograt = T.log(targets+eps) - T.log(predictions+eps)
    return T.sum(targets * lograt, axis=1)


class hinge_loss(object):
    # TODO: Add to_dict and from_dict functionality to hinge_loss
    def __init__(self, threshold=0.0):
        self.threshold = threshold
    @typechecker
    def __call__(self, x, t):
        t_ = T.switch(T.eq(t, 0), -1, 1)
        scores = 1 - (t_ * x)
        return T.maximum(0, scores - self.threshold)
    def to_dict(self):
        outdict = self.__dict__.copy()
        outdict['name'] = 'hinge_loss'
        return outdict
    def from_dict(self,d):
        self.threshold = d['threshold']
# convenience for typical hinge loss
regular_hinge_loss = hinge_loss(threshold=0)


class squared_hinge_loss(object):
    # TODO: Add to_dict and from_dict functionality to squared_hinge_loss
    def __init__(self, threshold=0.0, gamma=2.0):
        self.threshold = threshold
        self.gamma = gamma

    def __call__(self, x, t):
        hinge = hinge_loss(threshold=self.threshold)
        loss  = hinge(x, t)
        return 1.0/(2.0 * self.gamma) * loss**2
    
    def to_dict(self):
        outdict = self.__dict__.copy()
        outdict['name'] = 'squared_hinge_loss'
        return outdict
    
    def from_dict(self,d):
        self.threshold = d['threshold']
        self.gamma = d['gamma']
# convenience for typical squared hinge loss
regular_squared_hinge_loss = squared_hinge_loss(threshold=0, gamma=2.0)


class cross_covariance(object):
    def __init__(self, groups=None, mode='min'):
        """
        :param groups: a list of lists containing indices into data that define each group to compute cross-covariance
        :param mode: 'min' or 'max' to minimize or maximize the cross-covariance
        """
        if mode not in ['min', 'max']:
            raise Exception("mode must be either 'minimize' or 'maximize'")
        self.groups = groups
        self.mode = mode

    @typechecker
    def __call__(self, x, t):
        allg = set(range(len(self.groups)))
        ccov = T.zeros((1,), dtype=theano.config.floatX)
        for g1 in range(len(self.groups)-1):
            subsetg2 = allg.copy()
            subsetg2.remove(g1)
            for g2 in subsetg2:
                group1 = self.groups[g1]
                group2 = self.groups[g2]
                x_ = x[:, group1]
                t_ = t[:, group2]
                xmean = x_.mean(axis=0, keepdims=True)
                tmean = t_.mean(axis=0, keepdims=True)
                xdiff = x_ - xmean
                tdiff = t_ - tmean
                xcov = (1./x_.shape[0]) * T.dot(xdiff.T, tdiff)
                cmat = T.sqr(xcov)
                ccov += .5 * cmat.sum()
        if self.mode == 'min':
            return ccov
        elif self.mode == 'max':
            return -ccov

    def to_dict(self):
        outdict = self.__dict__.copy()
        outdict['name'] = 'cross_covariance'
        return outdict

    def from_dict(self,d):
        self.groups = d['groups']
        self.mode = d['mode']


#TODO:  fill in from_dict so that model can load objective objects from their serialized representation
def from_dict(objdict):
    name = objdict['name']
    o = getattr(xnn.objectives,name)()
    o.from_dict(objdict)
    return o


class ScaledObjective(object):
    def __init__(self, objective=absolute_error, scale=1.0):
        self.objective = objective
        self.scale = scale

    def __call__(self, x, t):
        return self.scale * self.objective(x, t)

    def to_dict(self):
        outdict = self.__dict__.copy()
        outdict['name']='ScaledObjective'
        outdict['objective'] = self.objective.__name__
        return outdict

    def from_dict(self,d):
        self.objective = getattr(xnn.objectives,d['objective'])
        self.scale = d['scale']

