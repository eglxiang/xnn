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

    pred = np.arange(-3,4).astype(theano.config.floatX)
    targ = pred[::-1]

    expected = np.abs(pred-targ)
    actual = f(pred, targ)
    assert np.all(actual == expected)

    # now check that aggregated mean absolute error works as expected
    y = aggregate(y,mode='mean')
    f = theano.function([x,t], y)

    expected = expected.mean()
    actual = f(pred, targ)
    assert np.allclose(actual, expected)

    return True


def test_kl_divergence():
    # test that kl divergence comes out same as stable numpy equivalent
    x = T.matrix()
    t = T.matrix()

    y = kl_divergence(x, t)
    f = theano.function([x,t], y)

    pred = np.float32([ [1/10., 1/10., 8/10.], [1/4., 1/4., 1/2.], [1/3., 1/3., 1/3.] ])
    targ = np.float32([ [0., 0., 1.], [1/4., 1/4., 1/2.], [1/2., 1/2., 0.] ])

    expected = np.sum(
        np.where(targ != 0,(targ) * np.log(targ / pred), 0), axis=1)
    actual = f(pred, targ)

    assert np.allclose(actual, expected, rtol=1.e-4, atol=1.e-4)
    return True


def test_hinge_loss():
    # TODO: Finish test for hinge loss
    # Currently just ensures that it runs without error
    x = T.vector()
    t = T.vector()

    y = hinge_loss()(x, t)
    f = theano.function([x,t], y)

    pred = np.arange(-3,4).astype(theano.config.floatX)
    targ = (pred[::-1] > 0).astype(theano.config.floatX)

    # expected =
    actual = f(pred, targ)

    # print pred
    # print targ
    # print pred-targ
    # print actual

    return True


def test_squared_hinge_loss():
    # TODO: Finish test for squared hinge loss
    # Currently just ensures that it runs without error
    x = T.vector()
    t = T.vector()

    y = squared_hinge_loss()(x, t)
    f = theano.function([x,t], y)

    pred = np.arange(-3,4).astype(theano.config.floatX)
    targ = (pred[::-1] > 0).astype(theano.config.floatX)

    actual = f(pred, targ)

    return True



if __name__ == '__main__':
    print 'test_absolute_error:', test_absolute_error()
    print 'test_kl_divergence:', test_kl_divergence()
    print 'test_hinge_loss:', test_hinge_loss()
    print 'test_squared_hinge_loss:', test_squared_hinge_loss()
