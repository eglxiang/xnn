from xnn.model.model import Model
from training.trainer import ParamUpdateSettings
from training.trainer import TrainerSettings
from training.trainer import Trainer
from xnn.layers import *
from xnn.objectives import *
from experiment import *
import theano

import numpy as np
import pprint

##############################
import warnings
warnings.filterwarnings('ignore', '.*topo.*')
warnings.filterwarnings('ignore', '.*Glorot.*')
##############################


def test_experiment():

    class Condition(ExperimentCondition):
        def __init__(self):
            self.batchsize = 10
            self.numpix = 3
            self.numhid = 100
            self.lr = 0.1
            self.mom = 0.5

    def build_trainer(cond=Condition()):
        m = Model('test model')
        l_in = m.add_layer(InputLayer(shape=(cond.batchsize, cond.numpix)), name="l_in")
        l_h1 = m.add_layer(DenseLayer(l_in, cond.numhid), name="l_h1")
        l_out = m.add_layer(DenseLayer(l_h1, cond.numpix), name="l_out")

        m.bind_input(l_in, "pixels")
        m.bind_output(l_h1, categorical_crossentropy, "emotions", "label", "mean")
        m.bind_output(l_out, squared_error, "l_in", "recon", "mean")

        global_update_settings = ParamUpdateSettings(learning_rate=cond.lr, momentum=cond.mom)

        trainer_settings = TrainerSettings(update_settings=global_update_settings)
        trainer = Trainer(m,trainer_settings)

        return trainer

    batch_dict = dict(
        # learning_rate_default=0.1,
        # momentum_default=0.5,
        pixels=np.random.rand(10,3).astype(theano.config.floatX),
        emotions=np.random.rand(10,100).astype(theano.config.floatX)
    )

    expt = Experiment("test experiment", Condition())
    expt.add_factor("numhid", [100])
    expt.add_factor("lr", [0.01, 0.05, 0.1, 0.5])
    expt.add_factor("mom", [0.01, 0.5, 0.9])

    print "expt.to_dict():"
    pprint.pprint(expt.to_dict())

    print '\nnum conditions:', expt.get_num_conditions()
    print 'conditions:'
    pprint.pprint(expt.get_all_conditions_changes())

    print '\ncondition(2) trainer:'
    cond = expt.get_nth_condition(2)
    trainer = build_trainer(cond=cond)
    pprint.pprint(trainer.to_dict())

    outs = trainer.train_step(batch_dict)

    print '\nget_conditions_slice_iterator:'
    iter = expt.get_conditions_slice_iterator(['lr'], dict(mom=.5))
    for cond in iter:
        print cond.to_dict()

if __name__ == "__main__":
    test_experiment()
