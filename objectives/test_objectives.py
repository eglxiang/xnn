from xnn.objectives import *

# from objectives import aggregate
import theano
import theano.tensor as T
import numpy as np


def test_absolute_error():
    x = T.vector()
    t = T.vector()

    # first check that the absolute error works as expected
    y = absolute_error(x, t)
    f = theano.function([x,t], y)

    predictions = np.arange(-3,4).astype(theano.config.floatX)
    targets = predictions[::-1]

    expected = np.abs(predictions-targets)
    actual = f(predictions, targets)
    assert np.all(actual == expected)

    # now check that aggregated mean absolute error works as expected
    y = aggregate(y,mode='mean')
    f = theano.function([x,t], y)

    expected = expected.mean()
    actual = f(predictions, targets)
    assert np.allclose(actual, expected)

    return True


# def test_kl_divergence():
#     x = T.matrix()
#     t = T.matrix()
#
#     y = kl_divergence(x, t)
#     f = theano.function([x,t], y)
#
#     predictions = np.float32([ [1/10., 1/10., 8/10.], [1/4., 1/4., 1/2.], [1/3., 1/3., 1/3.] ])
#     targets = np.float32([ [0, 0, 1], [1/4., 1/4., 1/2.], [1/2., 1/2., 0] ])
#
#     lograt = np.where()
#     lograt = np.log(targets/predictions)
#     expected = np.sum(targets * lograt, axis=1)
#     actual = f(predictions, targets)
#     print actual
#     print expected
#     return True

if __name__ == '__main__':
    print 'test_absolute_error:', test_absolute_error()
    # print 'test_kl_divergence:', test_kl_divergence()
