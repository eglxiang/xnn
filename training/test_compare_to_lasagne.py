from xnn.model import Model
from xnn.layers import *
from xnn.nonlinearities import *
from xnn.objectives import *
from xnn.training import *
from lasagne.updates import *
import theano
import theano.tensor as T
import numpy as np


def build_lr_net(batch_size, in_size, out_size):
    np.random.seed(100)
    m = Model()
    l_in = m.add_layer(InputLayer(shape=(batch_size,in_size)))
    l_out = m.add_layer(DenseLayer(l_in, out_size, nonlinearity=softmax))
    return m, l_in, l_out


def build_iter_train(batch_size, l_out, dataset):
    x = T.matrix()
    y = T.matrix()
    outs = get_output(l_out, x)

    batch_index = T.iscalar('batch_index')
    batch_slice = slice(batch_index * batch_size,
                        (batch_index + 1) * batch_size)

    loss = T.mean(categorical_crossentropy(outs, y))

    params = get_all_params(l_out)

    # The authors mention that they use adadelta, let's do the same
    updates = adadelta(loss, params)

    iter_train = theano.function(
        [batch_index], loss,
        updates=updates,
        givens={
            x: dataset['inputs'][batch_slice],
            y: dataset['labels'][batch_slice]
        }
    )

    return iter_train


def test_logistic_regression_trainer():
    batch_size = 32
    in_size = 10
    out_size = 5
    num_batches = 5

    m, l_in, l_out = build_lr_net(batch_size, in_size, out_size)

    m.bind_input(l_in, "inputs")
    m.bind_output(l_out, categorical_crossentropy, "labels", "label", "mean")

    global_update_settings = ParamUpdateSettings(update=adadelta)

    inputs = np.random.rand(batch_size * num_batches, in_size).astype(theano.config.floatX)
    labels = np.random.rand(batch_size * num_batches, out_size).astype(theano.config.floatX)
    labels = (labels == labels.max(axis=1, keepdims=True)).astype(theano.config.floatX)
    dataset = dict(
        inputs=theano.shared(inputs, borrow=True),
        labels=theano.shared(labels, borrow=True)
    )

    trainer_settings = TrainerSettings(global_update_settings=global_update_settings, dataSharedVarDict=dataset)
    trainer = Trainer(m, trainer_settings)

    trainer_batch_losses = np.zeros(num_batches)
    for b in range(num_batches):
        batch_dict = dict(batch_index=b)
        outs = trainer.train_step(batch_dict)
        trainer_batch_losses[b] = outs[-1]

    m, l_in, l_out = build_lr_net(batch_size, in_size, out_size)

    iter_train = build_iter_train(batch_size, l_out, dataset)

    lasagne_batch_losses = np.zeros(num_batches)
    for b in range(num_batches):
        batch_train_loss = iter_train(b)
        lasagne_batch_losses[b] = batch_train_loss

    print trainer_batch_losses
    print lasagne_batch_losses
    assert np.allclose(trainer_batch_losses, lasagne_batch_losses, rtol=1.e-4, atol=1.e-4)
    return True

if __name__ == '__main__':
    print test_logistic_regression_trainer()

