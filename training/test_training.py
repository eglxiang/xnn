# import xnn
from xnn.model import Model
from xnn import layers
from xnn.training.trainer import *
import xnn
import numpy as np
import lasagne
import tempfile
import shutil
import os.path

def test_train():
    batch_size = 128
    img_size = 10
    num_hid = 100

    m = _build_model(batch_size,img_size,num_hid)

    global_update_settings = ParamUpdateSettings(update=lasagne.updates.nesterov_momentum, learning_rate=0.1, momentum=0.5)

    trainer = Trainer(m,global_update_settings)

    pixels = np.random.rand(batch_size,img_size).astype(theano.config.floatX)
    emotions = np.random.rand(batch_size,num_hid).astype(theano.config.floatX)

    batch_dict = dict(
        # learning_rate_default=0.1,
        # momentum_default=0.5,
        pixels=pixels,
        emotions=emotions
    )
    outs = trainer.train_step(batch_dict)
    trainer.bind_update(m.layers['l_h1'],ParamUpdateSettings(learning_rate=0.01, momentum=0.6))
    outs = trainer.train_step(batch_dict)
    trainer.bind_update([m.layers['l_in'],'l_out'],ParamUpdateSettings(update=lambda loss,params,learning_rate,momentum=0.9: lasagne.updates.nesterov_momentum(loss,params,learning_rate,momentum),learning_rate=0.02,momentum=0.65))
    outs = trainer.train_step(batch_dict)
    
    print "Data on cpu succeeded"

    num_batches = 5

    pixels = np.random.rand(batch_size*num_batches,img_size).astype(theano.config.floatX)
    emotions = np.random.rand(batch_size*num_batches,num_hid).astype(theano.config.floatX)
    pixelsT = theano.shared(pixels)
    emotionsT = theano.shared(emotions)
    dataDict = dict(
        pixels=pixelsT,
        emotions=emotionsT
    )
    trainer = Trainer(m,global_update_settings,dataDict)
    batch_dict=dict(batch_index=0,batch_size=batch_size)
    outs = trainer.train_step(batch_dict)
    trainer.bind_update(m.layers['l_h1'],ParamUpdateSettings(learning_rate=0.01, momentum=0.6))
    outs = trainer.train_step(batch_dict)
    trainer.bind_update([m.layers['l_in'],'l_out'],ParamUpdateSettings(update=lambda loss,params,learning_rate,momentum=0.9: lasagne.updates.nesterov_momentum(loss,params,learning_rate,momentum),learning_rate=0.02,momentum=0.65))
    outs = trainer.train_step(batch_dict)

    print "Data on gpu succeeded"

def test_bind_global_update():
    batch_size = 128
    img_size = 10
    num_hid = 100

    m = _build_model(batch_size,img_size,num_hid)

    global_update_settings1 = ParamUpdateSettings(update=lasagne.updates.nesterov_momentum, learning_rate=0.1, momentum=0.5)
    global_update_settings2 = ParamUpdateSettings(update=lasagne.updates.adadelta, learning_rate=1.1, rho=0.9)

    trainer = Trainer(m,global_update_settings1)

    pixels = np.random.rand(batch_size,img_size).astype(theano.config.floatX)
    emotions = np.random.rand(batch_size,num_hid).astype(theano.config.floatX)

    batch_dict = dict(
        # learning_rate_default=0.1,
        # momentum_default=0.5,
        pixels=pixels,
        emotions=emotions
    )
    outs = trainer.train_step(batch_dict)
    trainer.bind_global_update(update_settings=global_update_settings2)
    outs = trainer.train_step(batch_dict)
    trainer.bind_update([m.layers['l_in'],'l_out'],ParamUpdateSettings(update=lambda loss,params,learning_rate,momentum=0.9: lasagne.updates.nesterov_momentum(loss,params,learning_rate,momentum),learning_rate=0.02,momentum=0.65))
    trainer.bind_global_update(update_settings=global_update_settings1,overwrite=False)
    outs = trainer.train_step(batch_dict)
    trainer.bind_global_update(update_settings=global_update_settings2,overwrite=True)
    outs = trainer.train_step(batch_dict)

def test_regularization():
    batch_size = 2
    img_size = 10
    num_hid = 10
    m = _build_model(batch_size,img_size,num_hid)
    m2 = _build_model(batch_size,img_size,num_hid)
    global_update_settings = ParamUpdateSettings(update=lasagne.updates.nesterov_momentum,learning_rate=0.1, momentum=0.2)
    trainer = Trainer(m,global_update_settings)
    trainer2 = Trainer(m2,global_update_settings)
    trainer.bind_regularization(xnn.regularization.l2, ['l_h1',('l_out',.1)])
    trainer.bind_regularization(xnn.regularization.l2,.5)
    
    pixels = np.random.rand(batch_size,img_size).astype(theano.config.floatX)
    emotions = np.random.rand(batch_size,num_hid).astype(theano.config.floatX)

    batch_dict = dict(
        # learning_rate_default=0.1,
        # momentum_default=0.5,
        pixels=pixels,
        emotions=emotions
    )

    trainer2.train_step(batch_dict)
    trainer.train_step(batch_dict)
    trainer2.train_step(batch_dict)
    trainer.train_step(batch_dict)
    lh_r= np.linalg.norm(m.layers['l_h1'].W.get_value())
    lh_n= np.linalg.norm(m2.layers['l_h1'].W.get_value())
    lo_r= np.linalg.norm(m.layers['l_out'].W.get_value())
    lo_n= np.linalg.norm(m2.layers['l_out'].W.get_value())
    assert lh_r<lh_n
    assert lo_r<lo_n

def test_trained_model_serialization():
    batch_size = 2
    img_size = 10
    num_hid = 10
    m = _build_model(batch_size,img_size,num_hid)
    global_update_settings = ParamUpdateSettings(update=lasagne.updates.nesterov_momentum,learning_rate=0.1, momentum=0.2)
    trainer = Trainer(m,global_update_settings)

    pixels = np.random.rand(batch_size,img_size).astype(theano.config.floatX)
    emotions = np.random.rand(batch_size,num_hid).astype(theano.config.floatX)

    batch_dict = dict(
        # learning_rate_default=0.1,
        # momentum_default=0.5,
        pixels=pixels,
        emotions=emotions
    )
    outs = trainer.train_step(batch_dict)
    preds = m.predict(batch_dict)

    dirpath = tempfile.mkdtemp()
    filepath = os.path.join(dirpath, 'testtrainerout')
    m.save_model(filepath)
    m2 = Model('load')
    m2.load_model(filepath)
    shutil.rmtree(dirpath)

    global_update_settings = ParamUpdateSettings(update=lasagne.updates.nesterov_momentum,learning_rate=0.1, momentum=0.2)
    trainer2 = Trainer(m2,global_update_settings)
    preds2 = m.predict(batch_dict)
    outs  = trainer.train_step(batch_dict)
    outs2 = trainer2.train_step(batch_dict)
   
    assert m.outputs.keys()==m2.outputs.keys()

    for p,p2 in zip(preds.values(),preds2.values()):
        assert np.allclose(p,p2)

    for o,o2 in zip(outs,outs2):
        assert np.allclose(o,o2)

def test_aggregation():
    batch_size = 128
    img_size = 10
    num_out = 3

    m = Model('test model cpu')
    l_in = m.add_layer(layers.InputLayer(shape=(batch_size,img_size)), name="l_in")
    l_out_mean = m.add_layer(layers.DenseLayer(l_in, num_out), name="l_out_mean")
    l_out_sum = m.add_layer(layers.DenseLayer(l_in, num_out), name="l_out_sum")

    m.bind_input(l_in, "pixels")
    m.bind_output(l_out_mean, lasagne.objectives.squared_error, "emotions", "label", "mean")
    m.bind_output(l_out_sum, lasagne.objectives.squared_error, "emotions", "label", "sum")

    from pprint import pprint
    pprint(m.to_dict())
    global_update_settings = ParamUpdateSettings(update=lasagne.updates.nesterov_momentum,learning_rate=0.1, momentum=0.5)

    trainer = Trainer(m, global_update_settings)

    pixels = np.random.rand(batch_size,img_size).astype(theano.config.floatX)
    emotions = np.random.rand(batch_size,num_out).astype(theano.config.floatX)

    batch_dict = dict(
        # learning_rate_default=0.1,
        # momentum_default=0.5,
        pixels=pixels,
        emotions=emotions,
        exmotions=emotions.copy()
    )
    outs = trainer.train_step(batch_dict)

    print "Aggregation test succeeded"

def test_trainer_serialization():
    batch_size = 128
    img_size = 10
    num_hid = 100

    m = _build_model(batch_size,img_size,num_hid)

    global_update_settings = ParamUpdateSettings(update=lasagne.updates.nesterov_momentum, learning_rate=0.1, momentum=0.5)

    trainer = Trainer(m,global_update_settings)
    print trainer.to_dict()

def _build_model(batch_size,img_size,num_hid):
    m = Model('test model cpu')
    l_in = m.add_layer(layers.InputLayer(shape=(batch_size,img_size)), name="l_in")
    l_loc = m.add_layer(layers.LocalLayer(l_in,num_units=3,img_shape=(2,5),local_filters=[(2,1)]))
    l_h1 = m.add_layer(layers.DenseLayer(l_loc, num_hid), name="l_h1")
    l_out = m.add_layer(layers.DenseLayer(l_h1, img_size), name="l_out")

    m.bind_input(l_in, "pixels")
    m.bind_output(l_h1, xnn.objectives.kl_divergence, "emotions", "label", "mean")
    m.bind_output(l_out, xnn.objectives.squared_error, "l_in", "recon", "mean")
    return m

if __name__ == '__main__':
    test_train()
    test_aggregation()
    test_bind_global_update()
    test_regularization()
    test_trained_model_serialization()
    test_trainer_serialization()
