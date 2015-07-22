from xnn.model.model import Model
from xnn.training.trainer import ParamUpdateSettings
from xnn.training.trainer import Trainer
from xnn.layers import *
from xnn.objectives import *
from experiment import *
import theano
import lasagne

import numpy as np
import pprint

from nose.tools import *

class _Condition(ExperimentCondition):
    def __init__(self):
        self.batchsize = 10
        self.numpix = 3
        self.numhid = 100
        self.lr = 0.1
        self.mom = 0.5
        self.loss1 = cross_covariance()
        self.loss2 = categorical_crossentropy

def _make_experiment():
    expt = Experiment("test experiment", _Condition())
    expt.add_factor("numhid", [100])
    expt.add_factor("lr", [0.01, 0.05, 0.1, 0.5])
    expt.add_factor("mom", [0.01, 0.5, 0.9])
    expt.add_factor("loss1", [categorical_crossentropy, cross_covariance()])
    expt.add_factor("loss2", [cross_covariance(), categorical_crossentropy])
    return expt

def test_experiment():
    # Currently just a run test (just to make sure it doesn't break)

    def build_trainer(cond=_Condition()):
        m = Model('test model')
        l_in = m.add_layer(InputLayer(shape=(cond.batchsize, cond.numpix)), name="l_in")
        l_h1 = m.add_layer(DenseLayer(l_in, cond.numhid), name="l_h1")
        l_out = m.add_layer(DenseLayer(l_h1, cond.numpix), name="l_out")

        m.bind_input(l_in, "pixels")
        m.bind_output(l_h1, categorical_crossentropy, "emotions", "label", "mean")
        m.bind_output(l_out, squared_error, "l_in", "recon", "mean")

        global_update_settings = ParamUpdateSettings(update=lasagne.updates.nesterov_momentum,learning_rate=cond.lr, momentum=cond.mom)

        trainer = Trainer(m,global_update_settings)

        return trainer

    batch_dict = dict(
        # learning_rate_default=0.1,
        # momentum_default=0.5,
        pixels=np.random.rand(10,3).astype(theano.config.floatX),
        emotions=np.random.rand(10,100).astype(theano.config.floatX)
    )

    expt = _make_experiment()

    print "expt.to_dict():"
    pprint.pprint(expt.to_dict())

    print '\nnum conditions:', expt.get_num_conditions()
    print 'conditions:'
    pprint.pprint(expt.get_all_conditions_changes())

    print '\ncondition(2) trainer:'
    cond = expt.get_nth_condition(2)
    trainer = build_trainer(cond=cond)
    pprint.pprint(trainer.to_dict())

    _ = trainer.train_step(batch_dict)

    print '\nget_conditions_iterator:'
    iter = expt.get_conditions_iterator(start=2, stop=5)
    for cond in iter:
        print cond.to_dict()

    print '\nget_conditions_slice_iterator:'
    iter = expt.get_conditions_slice_iterator(['lr'], dict(mom=.5))
    for cond in iter:
        print cond.to_dict()

def test_experiment_serialization():
    expt = _make_experiment()
    expected_dict = dict(
        default_condition=dict(
            numhid=100,
            batchsize=10,
            lr=0.1,
            mom=0.5,
            numpix=3,
            loss1=dict(mode='min', groups=None, name='cross_covariance'),
            loss2='categorical_crossentropy'
        ),
        factors=dict(
            numhid=[100],
            lr=[0.01, 0.05, 0.1, 0.5],
            mom=[0.01, 0.5, 0.9],
            loss1=['categorical_crossentropy', {'mode': 'min', 'groups': None, 'name': 'cross_covariance'}],
            loss2=[{'mode': 'min', 'groups': None, 'name': 'cross_covariance'}, 'categorical_crossentropy']
        )
    )
    actual_dict = expt.to_dict()
    pprint.pprint(actual_dict)
    assert expected_dict == actual_dict

@raises(RuntimeError)
def test_add_invalid_factor_name():
    expt = Experiment("test experiment", _Condition())
    expt.add_factor("invalid", [])

if __name__ == "__main__":
    test_experiment()
    test_experiment_serialization()
    test_add_invalid_factor_name()