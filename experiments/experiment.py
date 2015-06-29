from itertools import product, islice
from operator import mul
from copy import deepcopy


# Don't allow user to return more than this number of conditions
# in a single call to get_all_conditions_changes()
MAXCONDITIONSINMEMORY = 10000


class ExperimentCondition(object):
    def to_dict(self):
        properties = deepcopy(self.__dict__)
        for key in properties:
            if hasattr(properties[key], 'to_dict'):
                properties[key] = properties[key].to_dict()
        return properties


class Experiment(object):
    def __init__(self, id, default_condition):
        self.id = id
        self.factors = dict()
        self.default_condition = default_condition
        self.conditions = None

    def to_dict(self):
        return dict(
            factors=self.factors,
            default_condition=self.default_condition.to_dict()
        )

    def add_factor(self, name, levels):
        if name not in self.default_condition.__dict__:
            errstr = "name %s is not a valid property. Must be one of these:\n\t%s" \
                % (name, '\n\t'.join(self.default_condition.to_dict().keys()))
            raise RuntimeError(errstr)
        self.factors[name] = levels
        self._generate_conditions()

    def get_num_conditions(self):
        if self.factors:
            return reduce(mul, [len(self.factors[factor]) for factor in self.factors], 1)
        else:
            return 0

    def get_nth_condition_changes(self, n, default=None, conditions=None):
        conditions_ = self.conditions if conditions is None else conditions
        condition = next(islice(conditions_, n, None), default)
        # print n
        condition = dict(condition)
        # TODO: allow maintaining position in generator rather than resetting
        self._generate_conditions()
        return condition

    def get_nth_condition(self, n, default=None, conditions=None):
        conditions_ = self.conditions if conditions is None else conditions
        cond = next(islice(conditions_, n, None), default)
        return self._patch_condition(cond)

    def get_conditions_iterator(self, start=0, stop=None):
        stop = self.get_num_conditions() if stop is None else stop
        for i in range(start, stop):
            yield self.get_nth_condition(i)

    def get_conditions_slice_iterator(self, variable_keys, fixed_dict):
        factors = dict()
        for variable_key in variable_keys:
            factors[variable_key] = self.factors[variable_key]
        for fixed_key in fixed_dict:
            factors[fixed_key] = [(fixed_key, fixed_dict[fixed_key])]
        conditions = self._expand_factors(factors)
        n = sum([len(factors[variable_key]) for variable_key in variable_keys])
        for i in range(n):
            cond = conditions.next()
            yield self._patch_condition(cond)

    def get_all_conditions_changes(self):
        self._check_max_to_return()
        conditions = dict()
        for i in range(self.get_num_conditions()):
            condition_changes = self.get_nth_condition_changes(i)
            condition = ExperimentCondition()
            for key in condition_changes:
                condition.__dict__[key] = condition_changes[key]
            conditions[i] = condition.to_dict()
        return conditions

    def _generate_conditions(self):
        if not self.factors:
            raise Exception("You must add at least one factor to the experiment!")
        self.conditions = self._expand_factors(self.factors)

    def _check_max_to_return(self):
        n = self.get_num_conditions()
        if n > MAXCONDITIONSINMEMORY:
            raise Exception("You cannot get all conditions because"
                            " the number requested %d is greater than max=%d"
                            % (n, MAXCONDITIONSINMEMORY))

    def _patch_condition(self, cond):
        condition = deepcopy(self.default_condition)
        for item in cond:
            key = item[0]
            val = item[1]
            condition.__setattr__(key, val)
        return condition

    def _expand_factors(self, factors):
        factors_ = deepcopy(factors)
        for name in factors_:
            levels = factors_[name]
            factors_[name] = zip([name]*len(levels), levels)
        return product(*factors_.itervalues())
