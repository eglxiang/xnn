from itertools import product, islice
from operator import mul
from copy import deepcopy


# Don't allow user to return more than this number of conditions
# in a single call to get_all_conditions_changes()
MAXCONDITIONSINMEMORY = 10000


class ExperimentCondition(object):
    """
    The :class:`ExperimentCondition` class represents a structure holding experiment variable values.

    This class is used to parametrize experiment conditions.
    """

    def to_dict(self):
        """
        Returns a JSON-serializable python dictionary representing experiment condition variables.

        Returns
        -------
        dictionary of native python types
            A dictionary holding experiment condition variables
        """
        properties = deepcopy(self.__dict__)
        for key in properties:
            properties[key] = _convert_property_value(properties[key])
        return properties


class Experiment(object):
    """
    The :class:`Experiment` class manages specifying and iterating through an experiment design.

    It is intended to specify factors whose values define models (e.g. neural nets) and to
    manage iterating, querying, and reporting on particular experimental conditions.
    """

    def __init__(self, id, default_condition):
        """
        Instantiates the experiment.

        Parameters
        ----------
        id : a string or None
            An optional experiment ID.
        default_condition : a :class:`ExperimentCondition` instance
            The experiment condition object defining experiment variables.
        """
        self.id = id
        self.factors = dict()
        self.default_condition = default_condition
        self.conditions = None

    def to_dict(self):
        """
        Returns a JSON-serializable python dictionary representing the experiment specification.

        Returns
        -------
        dictionary of native python types
            A dictionary with keys "factors" and "default_condition" holding the specified
            factors and condition information.
        """
        factors = dict()
        for factorkey in self.factors:
            for item in self.factors[factorkey]:
                factors.setdefault(factorkey, [])
                converted = _convert_property_value(item)
                factors[factorkey].append(converted)
        return dict(
            factors=factors,
            default_condition=self.default_condition.to_dict()
        )

    def add_factor(self, name, levels):
        """
        Adds a factor to the experiment specifying levels of a particular condition to vary.

        Parameters
        ----------
        name : a string that must match one of the keys in the default_condition member variable
            the name of a key in default_condition whose values are to be varied.
        levels : a list of appropriate types (not type-safe)
            the values that define the levels of the factor
        """
        if name not in self.default_condition.__dict__:
            errstr = "name %s is not a valid property. Must be one of these:\n\t%s" \
                % (name, '\n\t'.join(self.default_condition.to_dict().keys()))
            raise RuntimeError(errstr)
        self.factors[name] = levels
        self._generate_conditions()

    def get_num_conditions(self):
        """
        Returns the number of experiment conditions (the cross-product of factor levels)

        Returns
        -------
        int
            An integer specifying the number of experiment conditions in the experiment.
        """
        if self.factors:
            return reduce(mul, [len(self.factors[factor]) for factor in self.factors], 1)
        else:
            return 0

    def get_nth_condition_changes(self, n, conditions=None):
        """
        Returns a dictionary containing the experiment variables for a particular condition
        that differ from the set of values specified in default_condition.

        Parameters
        ----------
        n : an int
            the condition number to query
        conditions : an optional itertools.product instance
            cartesian product of experiment factors.
            If not specified uses the conditions generated for the specified experiment.
        Returns
        -------
        dict
            A dictionary where keys/values are experiment condition variable names and values
            for those items that are based on specified factors.
        """
        conditions_ = self.conditions if conditions is None else conditions
        condition = next(islice(conditions_, n, None), {})
        # print n
        condition = dict(condition) if condition is not None else {}
        # TODO: allow maintaining position in generator rather than resetting
        self._generate_conditions()
        return condition

    def get_nth_condition(self, n, conditions=None):
        """
        Returns a :class:`ExperimentCondition` instance containing the experiment variables for a particular condition.

        Parameters
        ----------
        n : an int
            the condition number to query
        conditions : an optional itertools.product instance
            cartesian product of experiment factors.
            If not specified uses the conditions generated for the specified experiment.

        Returns
        -------
        A :class:`ExperimentCondition` instance
            Contains the experiment variables for a particular condition.
        """
        conditions_ = self.conditions if conditions is None else conditions
        cond = next(islice(conditions_, n, None), {})
        self._generate_conditions()
        return self._patch_condition(cond)

    def get_conditions_iterator(self, start=0, stop=None):
        """
        Returns an iterator over experiment conditions between start and end condition indices.

        Parameters
        ----------
        start : an int (default 0)
            A condition index to start the iterator.
        stop : an int or None (Default None)
            A condition index to end the iterator (None means continue through all conditions).

        Returns
        -------
        An iterator which iterates over :class:`ExperimentCondition` instances
        defined by the cross-product of experiment factors.
        """
        stop = self.get_num_conditions() if stop is None else stop
        for i in range(start, stop):
            yield self.get_nth_condition(i)

    def get_conditions_slice_iterator(self, variable_keys, fixed_dict):
        """
        Returns an iterator over a slice of experiment conditions defined by specified factor(s)
        to vary and specified fixed values.

        Parameters
        ----------
        variable_keys : a list of strings where each string must match a particular factor key
            Specifies the list of factor names to iterate over their values
        fixed_dict : a dict where key/value pairs define experiment variables to hold fixed
            Fix these factors at a particular level

        Returns
        -------
        An iterator which iterates over :class:`ExperimentCondition` instances
        defined by the specified fixed and variable factors specified.
        """
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
        """
        Returns a dictionary of dictionaries containing the experiment variables for each
        condition that differ from the set of values specified in default_condition.

        Returns
        -------
        dict of dicts
            A dictionary where key is condition id and value is a dictionary of
            key/value pairs listing the factor values specific to each condition
        """
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


def _convert_property_value(value):
    if hasattr(value, 'to_dict'):
        value = value.to_dict()
    elif hasattr(value, 'func_name'):
        value = value.func_name
    return value
