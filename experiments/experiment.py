from itertools import product, islice
from operator import mul
from copy import deepcopy

# TODO: Add ability to slice into experimental conditions (e.g. get all variants of learning rate for fixed momentum)

class ExperimentCondition(object):
    def to_dict(self):
        properties = deepcopy(self.__dict__)
        for key in properties:
            if hasattr(properties[key], 'to_dict'):
                properties[key] = properties[key].to_dict()
        return properties

    def get_property_names(self):
        return self.__dict__.keys()

class Experiment(object):
    def __init__(self, id, defaultcondition):
        self.id = id
        self.factors = dict()
        self.defaultcondition = defaultcondition
        self.conditions = None

    def add_factor(self, name, levels):
        if name not in self.defaultcondition.__dict__:
            errstr = "name %s is not a valid property. Must be one of these:\n\t%s" \
                % (name, '\n\t'.join(self.defaultcondition.to_dict().keys()))
            raise RuntimeError(errstr)
        self.factors[name] = zip([name]*len(levels), levels)

    def generate_conditions(self):
        self.conditions = product(*self.factors.itervalues())

    def get_num_conditions(self):
        return reduce(mul, [len(self.factors[factor]) for factor in self.factors], 1)

    def get_nth_condition_changes(self, n, default=None):
        if not self.conditions:
            self.generate_conditions()
        condition = next(islice(self.conditions, n, None), default)
        condition = dict(condition)
        self.generate_conditions()
        return condition

    def get_nth_condition(self, n, default=None):
        condition_changes = self.get_nth_condition_changes(n, default)
        condition = deepcopy(self.defaultcondition)
        for key in condition_changes:
            condition.__setattr__(key, condition_changes[key])
        return condition

    def get_all_conditions(self):
        conditions = [self.get_nth_condition(i) for i in range(self.get_num_conditions())]
        return conditions

    def get_all_conditions_changes(self):
        conditions = dict()
        for i in range(self.get_num_conditions()):
            condition_changes = self.get_nth_condition_changes(i)
            condition = ExperimentCondition()
            for key in condition_changes:
                condition.__dict__[key] = condition_changes[key]
            conditions[i] = condition.to_dict()
        return conditions

def experiment_test():
    import lasagne
    import theano
    from model.Model import Model
    from training.trainer import ParamUpdateSettings
    from training.trainer import TrainerSettings
    from training.trainer import Trainer
    import numpy as np
    import pprint

    class Condition(ExperimentCondition):
        def __init__(self):
            self.batchsize = 10
            self.numpix = 3
            self.numhid = 100
            self.lr = 0.1
            self.mom = 0.5

    def build_trainer(cond=Condition()):
        m = Model('test model')
        l_in = m.addLayer(lasagne.layers.InputLayer(shape=(cond.batchsize, cond.numpix)), name="l_in")
        l_h1 = m.addLayer(lasagne.layers.DenseLayer(l_in, cond.numhid), name="l_h1")
        l_out = m.addLayer(lasagne.layers.DenseLayer(l_h1, cond.numpix), name="l_out")

        m.bindInput(l_in, "pixels")
        m.bindOutput(l_h1, lasagne.objectives.categorical_crossentropy, "emotions", "label", "mean")
        m.bindOutput(l_out, lasagne.objectives.squared_error, "l_in", "recon", "mean")

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
    expt.add_factor("lr", [0.01, 0.05, 0.1])
    expt.add_factor("mom", [0.01, 0.5, 0.9])

    print 'num conditions:', expt.get_num_conditions()
    print 'conditions:'
    pprint.pprint(expt.get_all_conditions_changes())

    print 'condition(2)'
    cond = expt.get_nth_condition(2)
    trainer = build_trainer(cond=cond)
    pprint.pprint(trainer.to_dict())

    outs = trainer.train_step(batch_dict)

if __name__ == "__main__":
    experiment_test()
