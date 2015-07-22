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

BATCHSIZE = 500
NUMEPOCHS = 500


# ################## Download and prepare the MNIST dataset ##################
# This is just some way of getting the MNIST dataset from an online location
# and loading it into numpy arrays. It doesn't involve Lasagne or XNN at all.
def load_dataset():
    # We first define some helper functions for supporting both Python 2 and 3.
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
        import cPickle as pickle

        def pickle_load(f, encoding):
            return pickle.load(f)
    else:
        from urllib.request import urlretrieve
        import pickle

        def pickle_load(f, encoding):
            return pickle.load(f, encoding=encoding)

    # We'll now download the MNIST dataset if it is not yet available.
    url = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
    filename = 'mnist.pkl.gz'
    if not os.path.exists(filename):
        print("Downloading MNIST dataset...")
        urlretrieve(url, filename)

    # We'll then load and unpickle the file.
    import gzip
    with gzip.open(filename, 'rb') as f:
        data = pickle_load(f, encoding='latin-1')

    # The MNIST dataset we have here consists of six numpy arrays:
    # Inputs and targets for the training set, validation set and test set.
    X_train, y_train = data[0]
    X_val, y_val = data[1]
    X_test, y_test = data[2]

    # The inputs come as vectors, we reshape them to monochrome 2D images,
    # according to the shape convention: (examples, channels, rows, columns)
    X_train = X_train.reshape((-1, 1, 28, 28))
    X_val = X_val.reshape((-1, 1, 28, 28))
    X_test = X_test.reshape((-1, 1, 28, 28))

    # The targets are int64, we cast them to int8 for GPU compatibility.
    y_train = y_train.astype(np.uint8)
    y_val = y_val.astype(np.uint8)
    y_test = y_test.astype(np.uint8)

    y_train = xnn.utils.numpy_one_hot(y_train, 10)
    y_val = xnn.utils.numpy_one_hot(y_val, 10)
    y_test = xnn.utils.numpy_one_hot(y_test, 10)

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    dataset = dict(
        X_train=X_train,
        y_train=y_train,
        X_valid=X_val,
        y_valid=y_val,
        X_test=X_test,
        y_test=y_test
    )
    return dataset


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
    trainer_settings = TrainerSettings(global_update_settings=global_update_settings)
    trainer = Trainer(m, trainer_settings)
    return trainer


# Train the
def main():
    dataset = load_dataset()
    model = build_mlp()
    trainer = set_up_trainer(model)

    # TODO: track accuracy and crossentropy loss like in the lasagne example
    metrics = [
        ('l_out', Metric(computeCategoricalCrossentropy, "y", aggregation_type="mean"), 'min'),
        ('l_out', Metric(computeOneHotAccuracy, "y", aggregation_type="none"), 'max')
    ]

    trainbatchit = iterate_minibatches(dataset, BATCHSIZE, 'train')
    validbatchit = iterate_minibatches(dataset, BATCHSIZE, 'valid')

    loop = Loop(trainer, trainbatchit, validbatchit, metrics, plotmetricmean=False)
    loop(NUMEPOCHS)


if __name__ == '__main__':
    main()