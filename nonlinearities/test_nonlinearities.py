from nonlinearities import *
import theano
import theano.tensor as T
import numpy as np


def test_hard_sigmoid():
    # test that the hard sigmoid computes the proper values for particular anchor cases
    x        = T.vector('x')
    y        = hard_sigmoid(x)
    inputs   = np.array([-2.5, -1.25, 0., 1.25, 2.5]).astype(theano.config.floatX)
    expected = np.array([0, .25, .5, .75, 1]).astype(theano.config.floatX)
    actual   = y.eval({x:inputs})
    assert np.allclose(expected, actual)

def test_scale():
    # test that scaling the outputs by 10 is correct
    x        = T.vector('x')
    y        = scale(s=10)(x)
    inputs   = np.array([-2,-1,0,1,2]).astype(theano.config.floatX)
    expected = np.array([-20, -10, 0, 10, 20]).astype(theano.config.floatX)
    actual   = y.eval({x:inputs})
    assert np.allclose(expected, actual)

def test_sigmoid_evidence():
    # test that the log10 odds are correct
    x        = T.vector('x')
    y        = sigmoid_evidence(x)
    inputs   = np.log([100/1., 10/1., 1/1., 1/10., 1/100.]).astype(theano.config.floatX)
    expected = np.array([2, 1, 0, -1, -2]).astype(theano.config.floatX)
    actual   = y.eval({x:inputs})
    assert np.allclose(expected, actual)

def test_softmax_evidence():
    # test that base-10 sigmoid recovers proper probabilities from softmax evidence
    # for some cases
    x = T.matrix('x')
    y = softmax_evidence(x)
    f = theano.function([x], y)

    inputs   = np.log(np.float32([[1/2., 1/2., 0.], [1/3., 1/3., 1/3.]]))
    expected = np.float32([[2/3., 2/3., 0], [.5, .5, .5]])
    actual   = f(inputs)
    actual   = 1 / (1 + np.power(10, -actual))
    assert np.allclose(expected, actual)

if __name__ == '__main__':
    test_hard_sigmoid()
    test_scale()
    test_sigmoid_evidence()
    test_softmax_evidence()
