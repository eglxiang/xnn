import xnn
import numpy as np
import theano

# collection of tests for XNN layers to check correctness
# Run with nosetests

def test_contrast_normalization():
    np.random.seed(100)
    batchsize = 3
    numchannels = 1
    height = 10
    width = 5
    numdims = numchannels*height*width

    # flat
    l_in = xnn.layers.InputLayer(shape=(batchsize, numdims))
    l_cn = xnn.layers.ContrastNormLayer(l_in, norm_type="mean_var")
    pix_norm_flat = l_cn.get_output_for(l_in.input_var)
    f_flat = theano.function([l_in.input_var], pix_norm_flat)

    # image
    l_in = xnn.layers.InputLayer(shape=(batchsize, numchannels, height, width))
    l_cn = xnn.layers.ContrastNormLayer(l_in, norm_type="mean_var")
    pix_norm = l_cn.get_output_for(l_in.input_var)
    f_image = theano.function([l_in.input_var], pix_norm)


    pixels_flat = np.random.rand(batchsize, numdims).astype(theano.config.floatX)
    pixels = np.random.rand(batchsize, numchannels, height, width).astype(theano.config.floatX)

    outs_flat = f_flat(pixels_flat)
    outs_image = f_image(pixels)

    m_flat = outs_flat.mean(axis=1)
    s_flat = outs_flat.std(axis=1)
    m_image = outs_image.reshape(batchsize, numdims).mean(axis=1)
    s_image = outs_image.reshape(batchsize, numdims).std(axis=1)

    # means of flattened image should be close to 0
    assert np.allclose(m_flat,0,rtol=1e-05, atol=1e-05)

    # stdevs of flattened image should be close to 0
    assert np.allclose(s_flat,1,rtol=1e-05, atol=1e-05)

    # means of image should be close to 0
    assert np.allclose(m_image,0,rtol=1e-05, atol=1e-05)

    # stdevs of image should be close to 0
    assert np.allclose(s_image,1,rtol=1e-05, atol=1e-05)

    return True

def test_batch_normalization():
    np.random.seed(100)
    batchsize = 100
    numdims = 10
    # flat
    l_in = xnn.layers.InputLayer(shape=(batchsize, numdims))
    l_bn = xnn.layers.BatchNormLayer(l_in, learn_transform=True, eta=0)

    normed = l_bn.get_output_for(l_in.input_var)
    f = theano.function([l_in.input_var], normed)

    inputs = np.random.rand(batchsize, numdims).astype(theano.config.floatX)
    outs = f(inputs)

    # make sure batch-normalized means are close to 0
    assert np.allclose(outs.mean(axis=0), 0, rtol=1e-05, atol=1e-05)

    # make sure batch-normalized stdevs are close to 1
    assert np.allclose(outs.std(axis=0), 1, rtol=1e-05, atol=1e-05)

    return True

def test_gaussian_dropout():
    np.random.seed(100)
    batch_size = 10000
    input_vec = np.array([-100,-10,0,10,100]).astype(theano.config.floatX)
    l_in = xnn.layers.InputLayer(shape=(batch_size, input_vec.shape[0]))
    l_gd = xnn.layers.GaussianDropoutLayer(l_in, sigma=1.0)

    corrupted = l_gd.get_output_for(l_in.input_var)
    f = theano.function([l_in.input_var], corrupted)

    inputs = input_vec[np.newaxis,:].repeat(batch_size, axis=0)
    outs = f(inputs)

    # if input vector is corrupted N times, the mean corrupted vector should
    # approximately equal the input vector
    expected = np.round(input_vec/10)
    actual = np.round(outs.mean(axis=0)/10)
    assert np.all(expected-actual==0)

    # if input vector is corrupted N times, the stdev of the corrupted samples
    # should approximately equal the magnitude of the input vector
    expected = np.abs(np.round(input_vec/10))
    actual = np.round(outs.std(axis=0)/10)
    assert np.all(expected-actual==0)

    return True

def test_local():
    # TODO: Add tests for radius, edgeprotect=False, and multiple local filters with different probs
    # TODO: Consider tests for stacks of local layers
    np.random.seed(100)
    batchsize = 1
    numchannels = 1
    height = 100
    width = 100
    numdims = numchannels*height*width
    numunits = 10
    side = 9

    # First test results for a single side length with edge protection
    l_in = xnn.layers.InputLayer(shape=(batchsize, numdims))
    l_ll = xnn.layers.LocalLayer(l_in, num_units=numunits,
                                   img_shape=(height, width),
                                   local_filters=[(side, 1)],
                                   edgeprotect=True,
                                   mode='square')

    localmask = l_ll.params['localmask'].get_value()

    # Make sure that the localmask for a given hidden unit is side * side (use max for now)
    assert np.alltrue(localmask.sum(axis=0) == side**2)

    acts = l_ll.get_output_for(l_in.input_var)
    f = theano.function([l_in.input_var], acts)

    inputs = np.random.rand(batchsize, numdims).astype(theano.config.floatX)

    # This just makes sure the forward function computes without error
    outs = f(inputs)

    return True



if __name__ == '__main__':
    print 'test_contrast_normalization:', test_contrast_normalization()
    print 'test_batch_normalization:', test_batch_normalization()
    print 'test_gaussian_dropout:', test_gaussian_dropout()
    print 'test_local:', test_local()
