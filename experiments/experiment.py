from itertools import product, islice
from operator import mul
from copy import deepcopy
from collections import OrderedDict


class ExperimentCondition(object):
    """
    The :class:`ExperimentCondition` class is a base-class for representing
    various experimental variables and their default values.
    A given experiment should create a subclass with specific members
    representing the variables for the experiment.
    """

    def to_dict(self):
        """
        Returns a JSON-serializable python dictionary storing experiment variable names and values.

        Returns
        -------
        dictionary holding native python types
            A dictionary where keys are experiment variable names and values are the particular values
            that define the condition.
        """
        properties = deepcopy(self.__dict__)
        for key in properties:
            properties[key] = _convert_property_value(properties[key])
        return properties


class ExperimentGroup(object):
    """
    The :class:`ExperimentGroup` class manages a particular nested level of an experiment design.

    It is intended to handle nested (hierarchical) experiment designs.
    """

    def __init__(self, name, parent=None):
        self.parent = parent
        self.name = name
        self.local_factors = dict()
        self.children = []
        self.cross_product = None
        if parent:
            self.parent.children.append(self)

    def to_dict(self):
        """
        Returns a JSON-serializable python dictionary representing this experiment group.

        Returns
        -------
        dictionary of native python types
            A dictionary containing the groupname, the parent groupname, and the dictionary of factors.
        """
        factors = dict()
        for factorkey in self.local_factors:
            for item in self.local_factors[factorkey]:
                factors.setdefault(factorkey, [])
                converted = _convert_property_value(item)
                factors[factorkey].append(converted)
        outdict = dict(
            groupname=self.name,
            parentname=self.parent.name if self.parent is not None else None,
            factors=factors
        )
        return outdict

    def add_factor(self, name, values):
        if not isinstance(values, list):
            values = [values]
        self.local_factors[name] = values
        chained_factors = self.get_chained_factors()
        self.cross_product = self._get_factor_crossproduct(chained_factors)

    def get_chained_factors(self):
        groups = self._get_chain('root')
        chained_factors = dict()
        # children override parent factors of the same name
        for group in groups:
            for key in group.local_factors:
                chained_factors[key] = group.local_factors[key]
        return chained_factors

    def get_num_conditions(self):
        chained_factors = self.get_chained_factors()
        n = reduce(mul, [len(chained_factors[factor]) for factor in chained_factors], 1) if chained_factors else 0
        return n

    def get_nth_condition(self, n, base_condition=None):
        if not self.cross_product:
            chained_factors = self.get_chained_factors()
            self.cross_product = self._get_factor_crossproduct(chained_factors)
        changes = next(islice(self.cross_product, n, None), {})
        chained_factors = self.get_chained_factors()
        self.cross_product = self._get_factor_crossproduct(chained_factors)
        if base_condition:
            return self._patch_condition(base_condition, changes)
        else:
            return dict(changes)

    def get_condition_iterator(self, base_condition=None, start=0, stop=None):
        stop = self.get_num_conditions() if stop is None else stop
        for i in range(start, stop):
            yield self.get_nth_condition(i, base_condition=base_condition)

    def get_conditions_slice_iterator(self, variable_keys, fixed_dict, base_condition=None):
        chained_factors = self.get_chained_factors()
        slice_factors = dict()
        for variable_key in variable_keys:
            slice_factors[variable_key] = chained_factors[variable_key]
        for fixed_key in fixed_dict:
            slice_factors[fixed_key] = [(fixed_key, fixed_dict[fixed_key])]
        cross_product = self._get_factor_crossproduct(slice_factors)
        for changes in cross_product:
            if base_condition:
                yield self._patch_condition(base_condition, changes)
            else:
                yield changes

    def _get_chain(self, start='root'):
        if start not in ['root', 'leaf']:
            raise Exception("start must be root or leaf")
        chain = [self]
        root = self
        while root is not None:
            if root.parent:
                root = root.parent
                chain.append(root)
            else:
                break
        if start == 'root':
            return chain[::-1]
        else:
            return chain

    def _get_factor_crossproduct(self, factors):
        factors_ = deepcopy(factors)
        for name in factors_:
            levels = factors_[name]
            factors_[name] = zip([name]*len(levels), levels)
        return product(*factors_.itervalues())

    def _patch_condition(self, cond, cond_changes):
        cond_patched = deepcopy(cond)
        for item in cond_changes:
            key = item[0]
            val = item[1]
            cond_patched.__setattr__(key, val)
        return cond_patched


class Experiment(object):
    def __init__(self, name, default_condition):
        self.name = name
        self.default_condition = default_condition
        self.groups = dict(
            base=ExperimentGroup('base')
        )
        self.leaves = OrderedDict(base=self.groups['base'])

    def to_dict(self):
        groupsdict = {groupname: self.groups[groupname].to_dict() for groupname in self.groups}
        outdict = dict(
            name=self.name,
            default_condition=self.default_condition.to_dict(),
            groups=groupsdict
        )
        return outdict

    def add_group(self, groupname, parentname='base'):
        # TODO: check if groupname in groups and fail if so
        parent = self.groups[parentname]
        self.groups[groupname] = ExperimentGroup(groupname, parent)
        if parentname in self.leaves:
            del(self.leaves[parentname])
        self.leaves[groupname] = self.groups[groupname]

    def add_factor(self, name, values, groupname='base'):
        group = self.groups[groupname]
        group.add_factor(name, values)

    def get_num_conditions(self, groupname=None):
        num_conditions = 0
        groupnames = self.leaves if groupname is None else [groupname]
        for groupname in groupnames:
            num_conditions += self.groups[groupname].get_num_conditions()
        return num_conditions

    def get_group_nth_condition(self, n, groupname, changes_only=False):
        cond = None if changes_only else self.default_condition
        return self.groups[groupname].get_nth_condition(n, cond)

    def get_nth_condition(self, n, changes_only=False):
        cond = None if changes_only else self.default_condition
        n_sofar = 0
        for groupname in self.leaves:
            n_ingroup = self.get_num_conditions(groupname)
            n_offset = n - n_sofar
            n_sofar += n_ingroup
            if (n_offset >= 0) and (n_offset < n_ingroup):
                items = dict(groupname=groupname)
                if changes_only:
                    items['changes'] = self.groups[groupname].get_nth_condition(n_offset, cond)
                else:
                    items['condition'] = self.groups[groupname].get_nth_condition(n_offset, cond)
                return items

    def get_group_condition_iterator(self, groupname, changes_only=False, start=0, stop=None):
        cond = None if changes_only else self.default_condition
        return self.groups[groupname].get_condition_iterator(cond, start, stop)

    def get_all_condition_iterator(self, changes_only=False, start=0, stop=None):
        stop = self.get_num_conditions() if stop is None else stop
        for i in range(start, stop):
            yield self.get_nth_condition(i, changes_only=changes_only)

    def get_all_condition_changes_dict(self):
        return dict([(i, item) for i, item in enumerate(
            expt.get_all_condition_iterator(changes_only=True))])


class MySchema(ExperimentCondition):
    def __init__(self):
        self.servings = None
        self.preparation = None
        self.ripeness = None
        self.type = None
        self.mode = None
        self.seeds = None


expt = Experiment("food", default_condition=MySchema())
expt.add_group('fruit')
expt.add_group('veggies')
expt.add_group('banana', parentname='fruit')
expt.add_group('orange', parentname='fruit')
expt.add_group('zucchini', parentname='veggies')

expt.add_factor('servings', [1, 2, 3])
expt.add_factor('mode', ['real', 'fake'])
expt.add_factor('preparation', ['juice', 'whole', 'sliced'], groupname='fruit')
expt.add_factor('preparation', ['uncooked', 'grilled', 'boiled'], groupname='veggies')
expt.add_factor('ripeness', ['raw', 'ripe', 'overripe'], groupname='banana')
expt.add_factor('type', ['naval', 'blood', 'valencia'], groupname='orange')
expt.add_factor('seeds', True, groupname='zucchini')

# class Experiment(object):
#     """
#     The :class:`Experiment` class manages specifying and iterating through an experiment design.
#
#     It is intended to specify factors whose combinations of values define models
#     (e.g. neural nets) and to manage iterating, querying, and reporting results.
#     """
#
#     def __init__(self, id, default_condition):
#         """
#         Instantiates the experiment.
#
#         Parameters
#         ----------
#         id : a string or None
#             An optional experiment ID.
#         default_condition : a :class:`ExperimentCondition` instance
#             The experiment condition object defining experiment variables.
#         """
#         self.id = id
#         self.factors = dict()
#         self.default_condition = default_condition
#         self.conditions = None
#
#     def to_dict(self):
#         """
#         Returns a JSON-serializable python dictionary representing the experiment specification.
#
#         Returns
#         -------
#         dictionary of native python types
#             A dictionary with keys "factors" and "default_condition" holding the specified
#             factors and condition information.
#         """
#         factors = dict()
#         for factorkey in self.factors:
#             for item in self.factors[factorkey]:
#                 factors.setdefault(factorkey, [])
#                 converted = _convert_property_value(item)
#                 factors[factorkey].append(converted)
#         return dict(
#             factors=factors,
#             default_condition=self.default_condition.to_dict()
#         )
#
#     def add_factor(self, name, levels):
#         """
#         Adds a factor to the experiment specifying levels of a particular condition to vary.
#
#         Parameters
#         ----------
#         name : a string that must match one of the keys in the default_condition member variable
#             the name of a key in default_condition whose values are to be varied.
#         levels : a list of appropriate types (not type-safe)
#             the values that define the levels of the factor
#         """
#         if name not in self.default_condition.__dict__:
#             errstr = "name %s is not a valid property. Must be one of these:\n\t%s" \
#                 % (name, '\n\t'.join(self.default_condition.to_dict().keys()))
#             raise RuntimeError(errstr)
#         self.factors[name] = levels
#         self._generate_conditions()
#
#     def get_num_conditions(self):
#         """
#         Returns the number of experiment conditions (the cross-product of factor levels)
#
#         Returns
#         -------
#         int
#             An integer specifying the number of experiment conditions in the experiment.
#         """
#         if self.factors:
#             return reduce(mul, [len(self.factors[factor]) for factor in self.factors], 1)
#         else:
#             return 0
#
#     def get_nth_condition_changes(self, n, conditions=None):
#         """
#         Returns a dictionary containing the experiment variables for a particular condition
#         that differ from the set of values specified in default_condition.
#
#         Parameters
#         ----------
#         n : an int
#             the condition number to query
#         conditions : an optional itertools.product instance
#             cartesian product of experiment factors.
#             If not specified uses the conditions generated for the specified experiment.
#         Returns
#         -------
#         dict
#             A dictionary where keys/values are experiment condition variable names and values
#             for those items that are based on specified factors.
#         """
#         conditions_ = self.conditions if conditions is None else conditions
#         condition = next(islice(conditions_, n, None), {})
#         # print n
#         condition = dict(condition) if condition is not None else {}
#         # TODO: allow maintaining position in generator rather than resetting
#         self._generate_conditions()
#         return condition
#
#     def get_nth_condition(self, n, conditions=None):
#         """
#         Returns a :class:`ExperimentCondition` instance containing the experiment variables for a particular condition.
#
#         Parameters
#         ----------
#         n : an int
#             the condition number to query
#         conditions : an optional itertools.product instance
#             cartesian product of experiment factors.
#             If not specified uses the conditions generated for the specified experiment.
#
#         Returns
#         -------
#         A :class:`ExperimentCondition` instance
#             Contains the experiment variables for a particular condition.
#         """
#         conditions_ = self.conditions if conditions is None else conditions
#         cond = next(islice(conditions_, n, None), {})
#         self._generate_conditions()
#         return self._patch_condition(cond)
#
#     def get_conditions_iterator(self, start=0, stop=None):
#         """
#         Returns an iterator over experiment conditions between start and end condition indices.
#
#         Parameters
#         ----------
#         start : an int (default 0)
#             A condition index to start the iterator.
#         stop : an int or None (Default None)
#             A condition index to end the iterator (None means continue through all conditions).
#
#         Returns
#         -------
#         An iterator which iterates over :class:`ExperimentCondition` instances
#         defined by the cross-product of experiment factors.
#         """
#         stop = self.get_num_conditions() if stop is None else stop
#         for i in range(start, stop):
#             yield self.get_nth_condition(i)
#
#     def get_conditions_slice_iterator(self, variable_keys, fixed_dict):
#         """
#         Returns an iterator over a slice of experiment conditions defined by specified factor(s)
#         to vary and specified fixed values.
#
#         Parameters
#         ----------
#         variable_keys : a list of strings where each string must match a particular factor key
#             Specifies the list of factor names to iterate over their values
#         fixed_dict : a dict where key/value pairs define experiment variables to hold fixed
#             Fix these factors at a particular level
#
#         Returns
#         -------
#         An iterator which iterates over :class:`ExperimentCondition` instances
#         defined by the specified fixed and variable factors specified.
#         """
#         factors = dict()
#         for variable_key in variable_keys:
#             factors[variable_key] = self.factors[variable_key]
#         for fixed_key in fixed_dict:
#             factors[fixed_key] = [(fixed_key, fixed_dict[fixed_key])]
#         conditions = self._expand_factors(factors)
#         n = sum([len(factors[variable_key]) for variable_key in variable_keys])
#         for i in range(n):
#             cond = conditions.next()
#             yield self._patch_condition(cond)
#
#     def get_all_conditions_changes(self):
#         """
#         Returns a dictionary of dictionaries containing the experiment variables for each
#         condition that differ from the set of values specified in default_condition.
#
#         Returns
#         -------
#         dict of dicts
#             A dictionary where key is condition id and value is a dictionary of
#             key/value pairs listing the factor values specific to each condition
#         """
#         self._check_max_to_return()
#         conditions = dict()
#         for i in range(self.get_num_conditions()):
#             condition_changes = self.get_nth_condition_changes(i)
#             condition = ExperimentCondition()
#             for key in condition_changes:
#                 condition.__dict__[key] = condition_changes[key]
#             conditions[i] = condition.to_dict()
#         return conditions
#
#     def _generate_conditions(self):
#         if not self.factors:
#             raise Exception("You must add at least one factor to the experiment!")
#         self.conditions = self._expand_factors(self.factors)
#
#     def _check_max_to_return(self):
#         n = self.get_num_conditions()
#         if n > MAXCONDITIONSINMEMORY:
#             raise Exception("You cannot get all conditions because"
#                             " the number requested %d is greater than max=%d"
#                             % (n, MAXCONDITIONSINMEMORY))
#
#     def _patch_condition(self, cond):
#         condition = deepcopy(self.default_condition)
#         for item in cond:
#             key = item[0]
#             val = item[1]
#             condition.__setattr__(key, val)
#         return condition
#
#     def _expand_factors(self, factors):
#         factors_ = deepcopy(factors)
#         for name in factors_:
#             levels = factors_[name]
#             factors_[name] = zip([name]*len(levels), levels)
#         return product(*factors_.itervalues())


def _convert_property_value(value):
    if hasattr(value, 'to_dict'):
        value = value.to_dict()
    elif hasattr(value, 'func_name'):
        value = value.func_name
    return value

