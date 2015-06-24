import lasagne
import layers.normalization
import numpy as np
import theano

def test_contrast_normalization():
    batchsize = 3
    numchannels = 1
    height = 10
    width = 5
    numdims = numchannels*height*width

    # flat
    l_in = lasagne.layers.InputLayer(shape=(batchsize, numdims))
    l_cn = layers.normalization.ContrastNormLayer(l_in, norm_type="mean_var")
    pix_norm_flat = l_cn.get_output_for(l_in.input_var)
    f_flat = theano.function([l_in.input_var], pix_norm_flat)

    # image
    l_in = lasagne.layers.InputLayer(shape=(batchsize, numchannels, height, width))
    l_cn = layers.normalization.ContrastNormLayer(l_in, norm_type="mean_var")
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
