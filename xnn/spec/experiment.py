from itertools import product, islice
from operator import mul
from copy import deepcopy

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
