import sys
import pickle
import gzip
import os
import numpy as np

import theano
import lasagne


PY2 = sys.version_info[0] == 2

if PY2:
    from urllib import urlretrieve

    def pickle_load(f, encoding):
        return pickle.load(f)
else:
    from urllib.request import urlretrieve

    def pickle_load(f, encoding):
        return pickle.load(f, encoding=encoding)

DATA_URL = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
DATA_FILENAME = 'data/mnist.pkl.gz'


def _load_data(url=DATA_URL, filename=DATA_FILENAME):
    """Load data from `url` and store the result in `filename`."""
    if not os.path.exists(filename):
        print("Downloading MNIST dataset")
        urlretrieve(url, filename)

    with gzip.open(filename, 'rb') as f:
        return pickle_load(f, encoding='latin-1')

def delete_some_labels(labels, missing_pct):
    mask = np.random.rand(labels.shape[0], labels.shape[1]) < missing_pct
    labels_with_missing = np.where(mask, -12345, labels)
    return labels_with_missing

def load_data(missing_pct=None, random_seed=100):
    """Get data with labels, split into training, validation and test set."""
    data = _load_data()
    X_train, y_train_ = data[0]
    X_valid, y_valid_ = data[1]
    X_test, y_test_ = data[2]

    numclasses = 10

    y_train = np.zeros((X_train.shape[0], numclasses), dtype=theano.config.floatX)
    y_valid = np.zeros((X_valid.shape[0], numclasses), dtype=theano.config.floatX)
    y_test = np.zeros((X_test.shape[0], numclasses), dtype=theano.config.floatX)

    for i in range(numclasses):
        y_train[y_train_==i,i] = 1
        y_valid[y_valid_==i,i] = 1
        y_test[y_test_==i,i] = 1

    np.random.seed(random_seed)

    if missing_pct is not None:
        y_train = delete_some_labels(y_train, missing_pct)

    return dict(
        X_train=theano.shared(lasagne.utils.floatX(X_train)),
        y_train=theano.shared(lasagne.utils.floatX(y_train)),
        X_valid=theano.shared(lasagne.utils.floatX(X_valid)),
        y_valid=theano.shared(lasagne.utils.floatX(y_valid)),
        X_test=theano.shared(lasagne.utils.floatX(X_test)),
        y_test=theano.shared(lasagne.utils.floatX(y_test)),
        num_examples_train=X_train.shape[0],
        num_examples_valid=X_valid.shape[0],
        num_examples_test=X_test.shape[0],
        input_dim=X_train.shape[1],
        output_dim=numclasses,
    )

