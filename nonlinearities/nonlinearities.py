import theano.tensor.nnet
import theano.tensor as T
import numpy as np
import lasagne.nonlinearities


__all__ = [
    "hardsigmoid", "scale", "sigmoid_evidence", "softmax_evidence"
]


LOG10E = float(np.log10(np.exp(1)))

def hardsigmoid(x):
    """Piecewise linear approximation to a Sigmoid activation function
    Approx in 3 parts: 0, scaled linear, 1
        slope = 0.2
        shift = 0.5
    Removing the slope and shift does not make it faster.
    Parameters
    ----------
    x : float32
        The activation (the summed, weighted input of a neuron).
    Returns
    -------
    float32 in [0, 1]
        The output of the sigmoid function applied to the activation.
    """
    return theano.tensor.nnet.hard_sigmoid(x)

class scale(object):
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
