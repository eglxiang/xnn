#!/usr/bin/env python

"""
Modified usage example from Lasagne for digit recognition using the MNIST dataset.
"""

import sys
import os
import numpy as np

import xnn
from xnn.model import *
from xnn.layers import *
from xnn.nonlinearities import *
from xnn.objectives import *
from xnn.training import *
from xnn.metrics import *
from lasagne.updates import *
from bokeh.plotting import *
import argparse
from mnist_loader import *


BATCHSIZE = 500
NUMEPOCHS = 500


# make a generator to yield a batch of data for training/validating
class iterate_minibatches():
    def __init__(self, dataset, batchsize, partition='train'):
        self.dataset = dataset
        self.batchsize = batchsize
        self.partition = partition

    def __call__(self):
        inputs = self.dataset['X_'+self.partition]
        targets = self.dataset['y_'+self.partition]
        for start_idx in range(0, len(inputs) - self.batchsize + 1, self.batchsize):
            excerpt = slice(start_idx, start_idx + self.batchsize)
            batchdata = dict(
                X=inputs[excerpt],
                y=targets[excerpt]
            )
            yield batchdata


# ##################### Build the neural network model #######################
# We define a function that takes a Theano variable representing the input and returns
# the output layer of a neural network model.
def build_mlp(input_var=None):
    m = Model("MLP")
    # This creates an MLP of two hidden layers of 800 units each, followed by
    # a softmax output layer of 10 units. It applies 20% dropout to the input
    # data and 50% dropout to the hidden layers.

    # Input layer, specifying the expected input shape of the network
    # (unspecified batchsize, 1 channel, 28 rows and 28 columns) and
    # linking it to the given Theano variable `input_var`, if any:

    lin = m.make_bound_input_layer((None, 1, 28, 28), 'X', input_var=input_var)

    # Apply 20% dropout to the input data:
    l_in_drop = m.make_dropout_layer(lin, p=0.2)

    # Add a stack of fully-connected layers of 800 units each with dropout
    l_stacktop = m.make_dense_drop_stack(l_in_drop, [800, 800], drop_p_list=[.5, .5])

    # Finally, we'll add the fully-connected output layer, of 10 softmax units:
    l_out = m.add_layer(DenseLayer(l_stacktop, num_units=10, nonlinearity=softmax), "l_out")

    m.bind_output(l_out, categorical_crossentropy, 'y')
    return m


# ##################### Setup a trainer #######################
def set_up_trainer(m):
    global_update_settings = ParamUpdateSettings(update=nesterov_momentum, learning_rate=0.01, momentum=0.9)
    trainer = Trainer(m, global_update_settings)
    return trainer


# Train an MLP on MNIST
def main(ploturl=None, savefilenamecsv=None, savemodelnamebase=None):
    dataset = load_dataset()
    model = build_mlp()

    modelgraphimg = xnn.utils.draw_to_file(model, '/tmp/modelgraph.png')
    modelgraphimg.show()

    trainer = set_up_trainer(model)

    # TODO: track accuracy and crossentropy loss like in the lasagne example
    metrics = [
        ('l_out', Metric(computeCategoricalCrossentropy, "y", aggregation_type="mean"), 'min'),
        ('l_out', Metric(computeOneHotAccuracy, "y", aggregation_type="none"), 'max')
    ]

    trainbatchit = iterate_minibatches(dataset, BATCHSIZE, 'train')
    validbatchit = iterate_minibatches(dataset, BATCHSIZE, 'valid')

    loop = Loop(trainer, trainbatchit, validbatchit, metrics,
                plotmetricmean=False, url=ploturl,
                savefilenamecsv=savefilenamecsv, savemodelnamebase=savemodelnamebase)
    loop(NUMEPOCHS)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='mnist training example in XNN.')
    parser.add_argument('-p', dest='ploturl', type=str,
                   help='url to bokeh plot server (e.g. http://127.0.0.1:5006')
    parser.add_argument('-sf', dest='savefilenamecsv', type=str,
                   help='path to csv file containing epoch by epoch training progress')
    parser.add_argument('-sm', dest='savemodelnamebase', type=str,
                   help='path to prefix for saved model output (in pkl form)')
    args = parser.parse_args()

    main(args.ploturl, args.savefilenamecsv, args.savemodelnamebase)