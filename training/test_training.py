# import xnn
from ..model import Model
from .. import layers
from .trainer import *
import numpy as np
import lasagne


def test_train():
    # from model.Model import Model

    batch_size = 128
    img_size = 10
    num_hid = 100


    m = Model('test model cpu')
    l_in = m.addLayer(layers.InputLayer(shape=(batch_size,img_size)), name="l_in")
    l_h1 = m.addLayer(layers.DenseLayer(l_in, num_hid), name="l_h1")
    l_out = m.addLayer(layers.DenseLayer(l_h1, img_size), name="l_out")

    m.bindInput(l_in, "pixels")
    m.bindOutput(l_h1, lasagne.objectives.categorical_crossentropy, "emotions", "label", "mean")
    m.bindOutput(l_out, lasagne.objectives.squared_error, "l_in", "recon", "mean")

    global_update_settings = ParamUpdateSettings(learning_rate=0.1, momentum=0.5)

    trainer_settings = TrainerSettings(update_settings=global_update_settings)
    trainer = Trainer(trainer_settings,m)

    pixels = np.random.rand(batch_size,img_size).astype(theano.config.floatX)
    emotions = np.random.rand(batch_size,num_hid).astype(theano.config.floatX)

    batch_dict = dict(
        # learning_rate_default=0.1,
        # momentum_default=0.5,
        pixels=pixels,
        emotions=emotions
    )
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
    trainer_settings = TrainerSettings(update_settings=global_update_settings,dataSharedVarDict=dataDict)
    trainer = Trainer(trainer_settings,m)
    batch_dict=dict(batch_index=0)
    outs = trainer.train_step(batch_dict)

    print "Data on gpu succeeded"
