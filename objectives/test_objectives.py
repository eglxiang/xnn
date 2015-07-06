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
    actual   = f(pred, targ)
    decorated = absolute_error(pred,targ)

    assert np.all(actual == expected)
    assert np.all(actual == decorated)

    # now check that aggregated mean absolute error works as expected
    y = aggregate(y,mode='mean')
    f = theano.function([x,t], y)

    expected = expected.mean()
    actual   = f(pred, targ)

    assert np.allclose(actual, expected)

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
    decorated = kl_divergence(pred,targ)

    assert np.allclose(actual, expected, rtol=1.e-4, atol=1.e-4)
    assert np.allclose(actual, decorated, rtol=1.e-4, atol=1.e-4)

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
    decorated = hinge_loss()(pred,targ)

    assert np.allclose(actual, decorated, rtol=1.e-4, atol=1.e-4)
    # print pred
    # print targ
    # print pred-targ
    # print actual

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
    decorated = squared_hinge_loss()(pred,targ)

    assert np.allclose(actual, decorated, rtol=1.e-4, atol=1.e-4)

def test_cross_covariance():
    x = T.matrix()
    t = T.matrix()

    groups = [ [0,1,2], [3,4,5,6] ]
    y = cross_covariance(groups, 'min')(x, t)

    f = theano.function([x,t], y)

    a = np.random.randn(100, 3).astype(theano.config.floatX)
    b = np.random.randn(100, 4).astype(theano.config.floatX)
    c = np.concatenate((a,b), axis=1)

    xcov = np.cov(a, b, bias=1, rowvar=0)
    xcov = xcov[0:3, 3:]
    cmat = xcov**2
    expected = .5 * cmat.sum()
    actual = f(c, c)[0]
    assert np.isclose(actual, expected)

    decorated = cross_covariance(groups, 'min')(c, c)
    assert np.allclose(actual, decorated, rtol=1.e-4, atol=1.e-4)

if __name__ == '__main__':
    test_absolute_error()
    test_kl_divergence()
    test_hinge_loss()
    test_squared_hinge_loss()
    test_cross_covariance()
