import theano.tensor as T


__all__ = [
    "absolute_error",
    "kl_divergence",
    "hinge_loss",
    "squared_hinge_loss"
]


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
    return T.abs_(a - b)


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


class hinge_loss():
    # TODO: Add to_dict and from_dict functionality to hinge_loss
    def __init__(self, threshold=0.0):
        self.threshold = threshold

    def __call__(self, x, t):
        t_ = T.switch(T.eq(t, 0), -1, 1)
        scores = 1 - (t_ * x)
        return T.maximum(0, scores - self.threshold)
# convenience for typical hinge loss
regular_hinge_loss = hinge_loss(threshold=0)


class squared_hinge_loss():
    # TODO: Add to_dict and from_dict functionality to squared_hinge_loss
    def __init__(self, threshold=0.0, gamma=2.0):
        self.threshold = threshold
        self.gamma = gamma

    def __call__(self, x, t):
        hinge = hinge_loss(threshold=self.threshold)
        loss = hinge(x, t)
        return 1.0/(2.0 * self.gamma) * loss**2
# convenience for typical squared hinge loss
regular_squared_hinge_loss = squared_hinge_loss(threshold=0, gamma=2.0)

