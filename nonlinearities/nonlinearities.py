import theano.tensor.nnet
import theano.tensor as T
import numpy as np
import lasagne.nonlinearities


__all__ = [
    "hard_sigmoid", "scale", "sigmoid_evidence", "softmax_evidence"
]


LOG10E = float(np.log10(np.exp(1)))

def hard_sigmoid(x):
    """
    Piecewise linear approximation to a Sigmoid activation function
    Approximation in 3 parts: 0, scaled linear, 1.
    The scaled linear section has slope = 0.2 and shift = 0.5.

    Parameters
    ----------
    x : theano tensor
        The tensor hard sigmoid will be applied to

    Returns
    -------
    theano tensor in [0, 1]
        The output of the sigmoid function applied to x.
    """
    return theano.tensor.nnet.hard_sigmoid(x)

class scale(object):
    """
    Object which, when called on a value, returns the scaled value

    Parameters
    ----------
    s : float, optional, for initialization
        The amount by which to scale. Default is 1.0
    x : scalar, numpy array, or theano tensor, for call
        Value to scale

    Returns
    -------
    x scaled by s. Same type as x.
    """
    def __init__(self, s=1.0):
        """Specify a floating point number to scale"""
        self.s = s

    def __call__(self, x):
        """Scale the x's by a constant"""
        return self.s * x

def sigmoid_evidence(x):
    """Output log10 evidence (log-likelihood ratio) for an independent sigmoid model"""
    return LOG10E * x

def softmax_evidence(x):
    """
    Output log10 evidence (log-likelihood ratio) for a softmax model.
    Assumes the prior for class i is equal to the prior for not class i.
    This corresponds to comparing probability for class i to the average probability for not class i.
    """
    num_other_classes = T.cast(x.shape[1] - 1, 'float32')
    p                 = lasagne.nonlinearities.softmax(x)
    q                 = (1-p)/num_other_classes
    part              = T.log(p) - T.log(q)
    return part * LOG10E
