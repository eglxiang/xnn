import xnn
from xnn.model import Model
import numpy as np
import pprint


def test_build_model():
    m = Model('test model')
    l_in  = m.add_layer(xnn.layers.InputLayer(shape=(10,200)), name="l_in")
    l_h1  = m.add_layer(xnn.layers.DenseLayer(l_in, 100), name="l_h1")
    l_out = m.add_layer(xnn.layers.DenseLayer(l_h1, 200), name="l_out")

    m.bind_input(l_in, "pixels")
    m.bind_output(l_h1, xnn.objectives.categorical_crossentropy, "emotions", "label", "mean")
    m.bind_output(l_out, xnn.objectives.mse, "l_in", "recon", "mean")
    return True


def _build_model():
    m2    = Model('test convenience')
    l_in  = m2.make_bound_input_layer((10,200),'pixels')
    l_out = m2.make_dense_drop_stack(l_in,[60,3,2],[.6,.4,.3])
    l_mer = xnn.layers.MergeLayer([l_in, l_out])
    m2.add_layer(l_mer,name='merger')
    m2.bind_output(l_out, xnn.objectives.squared_error, 'age', 'label', 'mean')
    return m2

def test_convenience_build():
    m2 = _build_model()
    return True

def test_serialization():
    m2 = _build_model()
    serialized = m2.to_dict()
    pprint.pprint(serialized)

    m3 = Model('test serialize')
    m3.from_dict(serialized)
    pprint.pprint(m3.to_dict())
    m3.save_model('testmodelout')

    m4 = Model('test load')
    m4.load_model('testmodelout')

    assert np.allclose(m4.layers['l__dense_2'].W.get_value(),m3.layers['l__dense_2'].W.get_value())
    assert ~np.allclose(m4.layers['l__dense_2'].W.get_value(),m2.layers['l__dense_2'].W.get_value())

    data = dict(
        pixels=np.random.rand(10,200),
            )

    out4 = m4.predict(data,['l__drop_2'])
    out3 = m3.predict(data,['l__drop_2'])

    print out3['l__drop_2']
    print out4['l__drop_2']

    assert np.allclose(out4['l__drop_2'],out3['l__drop_2'])

    return True


if __name__ == "__main__":
    print "test_build",test_build_model()
    print "test_convenience_build",test_convenience_build()
    print "test_serialization",test_serialization()

