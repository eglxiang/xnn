import xnn
from xnn.model import Model
import numpy as np
import pprint
import time
import theano
import tempfile
import shutil
import os.path


def test_build_model():
    m = Model('test model')
    l_in  = m.add_layer(xnn.layers.InputLayer(shape=(10,200)), name="l_in")
    l_h1  = m.add_layer(xnn.layers.DenseLayer(l_in, 100), name="l_h1")
    l_out = m.add_layer(xnn.layers.DenseLayer(l_h1, 200), name="l_out")

    m.bind_input(l_in, "pixels")
    m.bind_output(l_h1, xnn.objectives.categorical_crossentropy, "emotions", "label", "mean")
    m.bind_output(l_out, xnn.objectives.mse, "l_in", "recon", "mean")

def test_convenience_build():
    m2 = _build_model()

def test_serialization():
    m2 = _build_model()
    serialized_old = m2.to_dict()
    print "original model"
    pprint.pprint(serialized_old)

    m3 = Model('test serialize')
    m3.from_dict(serialized_old)
    print "model loaded from dict"
    pprint.pprint(m3.to_dict())
    data = dict(pixels=np.random.rand(10,1,20,10).astype(theano.config.floatX))

    dirpath = tempfile.mkdtemp()
    filepath = os.path.join(dirpath, 'testmodelout')

    m3.save_model(filepath)
    out3 = m3.predict(data,['out','out2'])

    m4 = Model('test convenience')
    m4.load_model(filepath)

    shutil.rmtree(dirpath)

    assert serialized_old == m4.to_dict()

    assert np.allclose(m4.layers['DenseLayer_2'].W.get_value(),m3.layers['DenseLayer_2'].W.get_value())
    assert ~np.allclose(m4.layers['DenseLayer_2'].W.get_value(),m2.layers['DenseLayer_2'].W.get_value())


    out4 = m4.predict(data,['out','out2'])

    print 'out pre serialization'
    print out3['out']
    print 'out post serialization'
    print out4['out']

    print 'out2 pre serialization'
    print out3['out2']
    print 'out2 post serialization'
    print out4['out2']
    assert np.allclose(out4['out'],out3['out'])
    assert np.allclose(out4['out2'],out3['out2'])

def _build_model():
    r = xnn.nonlinearities.rectify
    s = xnn.nonlinearities.scale
    m2    = Model('test convenience')
    l_in  = m2.make_bound_input_layer((10,1,20,10),'pixels')
    l_in2 = m2.make_bound_input_layer((10,1,20,10),'pixels')
    l_conv = m2.add_layer(xnn.layers.Conv2DLayer(l_in2,3,4),name='l_conv')
    l_den = m2.make_dense_drop_stack(l_in,[60,3,2],[.6,.4,.3],drop_type_list=['gauss','gauss','standard'],nonlin_list=[r,s(2.),r])
    l_reshp = m2.add_layer(xnn.layers.ReshapeLayer(l_conv,(10,-1)))
    l_mer = xnn.layers.ConcatLayer([l_reshp, l_den])
    m2.add_layer(l_mer,name='merger')
    l_bn = xnn.layers.BatchNormLayer(l_in2,nonlinearity=xnn.nonlinearities.sigmoid)
    m2.add_layer(l_bn)
    l_loc = xnn.layers.LocalLayer(l_bn,num_units=32,img_shape=(20,10),local_filters=[(2,1)],seed=121212)
    np.random.seed(int(time.time()*10000%10000))
    m2.add_layer(l_loc)
    l_out= m2.add_layer(xnn.layers.DenseLayer(l_loc,num_units=2,nonlinearity=xnn.nonlinearities.linear),name='out')
    l_pr = m2.add_layer(xnn.layers.PReLULayer(l_out,xnn.init.Constant(.25)))
    m2.bind_output(l_pr, xnn.objectives.squared_error, 'age', 'label', 'mean',is_eval_output=True,scale=2)
    l_out2= m2.add_layer(xnn.layers.DenseLayer(l_loc,num_units=2,nonlinearity=xnn.nonlinearities.softmax),name='out2')
    m2.bind_output(l_out2, xnn.objectives.hinge_loss(threshold=2), 'age','label','mean',is_eval_output=False)
    return m2

if __name__ == "__main__":
    test_build_model()
    test_convenience_build()
    test_serialization()

