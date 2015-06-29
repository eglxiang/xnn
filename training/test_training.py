# import xnn
from xnn.model import Model
from xnn import layers
from xnn.training.trainer import *
import numpy as np
import lasagne


def test_train():
    batch_size = 128
    img_size = 10
    num_hid = 100


    m = Model('test model cpu')
    l_in = m.add_layer(layers.InputLayer(shape=(batch_size,img_size)), name="l_in")
    l_h1 = m.add_layer(layers.DenseLayer(l_in, num_hid), name="l_h1")
    l_out = m.add_layer(layers.DenseLayer(l_h1, img_size), name="l_out")

    m.bind_input(l_in, "pixels")
    m.bind_output(l_h1, lasagne.objectives.categorical_crossentropy, "emotions", "label", "mean")
    m.bind_output(l_out, lasagne.objectives.squared_error, "l_in", "recon", "mean")

    global_update_settings = ParamUpdateSettings(learning_rate=0.1, momentum=0.5)

    trainer_settings = TrainerSettings(global_update_settings=global_update_settings)
    trainer = Trainer(m,trainer_settings)

    pixels = np.random.rand(batch_size,img_size).astype(theano.config.floatX)
    emotions = np.random.rand(batch_size,num_hid).astype(theano.config.floatX)

    batch_dict = dict(
        # learning_rate_default=0.1,
        # momentum_default=0.5,
        pixels=pixels,
        emotions=emotions
    )
    outs = trainer.train_step(batch_dict)
    trainer.bindUpdate(l_h1,ParamUpdateSettings(learning_rate=0.01, momentum=0.6))
    outs = trainer.train_step(batch_dict)
    trainer.bindUpdate([l_in,'l_out'],ParamUpdateSettings(update=lambda *args,**kwargs: lasagne.updates.nesterov_momentum(*args,**kwargs)))
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
    trainer_settings = TrainerSettings(global_update_settings=global_update_settings,dataSharedVarDict=dataDict)
    trainer = Trainer(m,trainer_settings)
    batch_dict=dict(batch_index=0)
    outs = trainer.train_step(batch_dict)
    trainer.bindUpdate(l_h1,ParamUpdateSettings(learning_rate=0.01, momentum=0.6))
    outs = trainer.train_step(batch_dict)
    trainer.bindUpdate([l_in,'l_out'],ParamUpdateSettings(update=lambda *args,**kwargs: lasagne.updates.nesterov_momentum(*args,**kwargs)))
    outs = trainer.train_step(batch_dict)

    print "Data on gpu succeeded"
    return True

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
    global_update_settings = ParamUpdateSettings(learning_rate=0.1, momentum=0.5)

    trainer_settings = TrainerSettings(global_update_settings=global_update_settings)
    trainer = Trainer(m, trainer_settings)

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
    return True

if __name__ == '__main__':
    print test_train()
    print test_aggregation()
