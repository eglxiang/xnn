from itertools import product, islice
from operator import mul
from copy import deepcopy
from collections import OrderedDict
from itertools import ifilter


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

    def from_dict(self, condition_dict):
        """
        Creates object attributes specified by keys in condition_dict and sets their values. Intended for instantiating
        an object of this type from a dictionary (e.g. when loading spec from file).

        Parameters
        ----------
        condition_dict : a python dict
            Dictionary of condition keys/values to populate this object.
        """
        for key, value in condition_dict.iteritems():
            setattr(self, key, value)


class ExperimentGroup(object):
    """
    The :class:`ExperimentGroup` class manages a particular group within an experiment design. The group is a node
    within a tree defined by a linked list of :class:`ExperimentGroup` objects. An experiment group can also function
    as a minimal (flat) experiment.

    It is intended to handle nested (hierarchical) experiment designs.
    """

    def __init__(self, name, parent=None):
        """
        Instantiates a :class:`ExperimentGroup` object.

        Parameters
        ----------
        name : a string
            A name to identify the experiment group.
        parent : a :class:`ExperimentGroup` instance
            The parent experiment group within the experiment design.
        """
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
        """
        Adds a factor to the experiment group specifying levels of a particular independent variable.

        Parameters
        ----------
        name : a string
            The name of the factor.
        levels : a list of appropriate types. Can also be a single value (gets converted to a list of size 1).
            The values that define the levels of the factor
        """
        if not isinstance(values, list):
            values = [values]
        self.local_factors[name] = values
        chained_factors = self._get_chained_factors()
        self.cross_product = self._get_factor_crossproduct(chained_factors)

    def get_num_conditions(self):
        """
        Returns the number of experiment conditions for this group (the cross-product of chained factor levels)

        Returns
        -------
        int
            An integer specifying the number of experiment conditions comprising the cross-product of chained factors.
        """

        chained_factors = self._get_chained_factors()
        n = reduce(mul, [len(chained_factors[factor]) for factor in chained_factors], 1) if chained_factors else 0
        return n

    def get_nth_condition(self, n, base_condition=None):
        """
        Returns the variables defining a particular condition as a dictionary or a condition object depending on how
        the method is called.

        Parameters
        ----------
        n : an int
            the condition number to query within the cross-product of chained factors.
        base_condition : an optional :class:`ExperimentGroup` instance
            If specified then the returned condition will be an appropriately patched condition object rather than a
            dictionary of condition variables.
        Returns
        -------
        dict
            Either a dictionary where keys/values are experiment condition variable names and values
            for those items that are based on specified factors, or a patched condition object.
        """
        if not self.cross_product:
            chained_factors = self._get_chained_factors()
            self.cross_product = self._get_factor_crossproduct(chained_factors)
        changes = next(islice(self.cross_product, n, None), {})
        chained_factors = self._get_chained_factors()
        self.cross_product = self._get_factor_crossproduct(chained_factors)
        if base_condition:
            return self._patch_condition(base_condition, changes)
        else:
            return dict(changes)

    def get_condition_iterator(self, base_condition=None, start=0, stop=None):
        """
        Returns an iterator over experiment conditions for this group between start and end condition indices.

        Parameters
        ----------
        base_condition : an optional :class:`ExperimentGroup` instance
            If specified then the conditions yielded by the iterator will be appropriately patched condition objects
            rather than dictionaries of condition variables.
        start : an int (default 0)
            A condition index to start the iterator.
        stop : an int or None (Default None)
            A condition index to end the iterator (None means continue through all conditions).

        Returns
        -------
        An iterator which iterates over :class:`ExperimentCondition` instances
        defined by the cross-product of chained experiment factors for this group.
        """
        stop = self.get_num_conditions() if stop is None else stop
        for i in range(start, stop):
            yield self.get_nth_condition(i, base_condition=base_condition)

    def get_conditions_slice_iterator(self, variable_keys, fixed_dict, base_condition=None):
        """
        Returns an iterator over a slice of experiment conditions for this group defined by specified factor(s)
        to vary and specified fixed values.

        Parameters
        ----------
        variable_keys : a list of strings where each string must match a particular factor key
            Specifies the list of factor names to iterate over their values
        fixed_dict : a dict where key/value pairs define experiment variables to hold fixed
            Fix these factors at a particular level
        base_condition : an optional :class:`ExperimentGroup` instance
            If specified then the conditions yielded by the iterator will be appropriately patched condition objects
            rather than dictionaries of condition variables.

        Returns
        -------
        An iterator which iterates over :class:`ExperimentCondition` instances
        defined by the specified fixed and variable chained factors specified for this group.
        """
        chained_factors = self._get_chained_factors()
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

    # private methods
    def _get_chained_factors(self):

        groups = self._get_chain('root')
        chained_factors = dict()
        # children override parent factors of the same name
        for group in groups:
            for key in group.local_factors:
                chained_factors[key] = group.local_factors[key]
        return chained_factors

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
        self.results = dict()
        self.group_ordering = []
        self._populate_default_factors()

    def add_results(self, condition_num, results):
        """
        Adds a set of results for a given experimental condition. The results are intended to be stored as a
        JSON-serializable data structure.

        Parameters
        ----------
        parentlayer : lasagne :class:`Layer`
            layer whose outputs are the input to the dense layer.
        num_units : int
            The number of units in the dense layer.
        nonlinearity : function
            The nonlinearity function to use on this layer.  If None, :py:meth:`xnn.nonlinearities.rectify` is used.
        condition_num : int
            The condition number in the experiment associated with these results.
        results : JSON-serializable python data
            The result data to be stored for the given condition number. E.g. performance of net on test data.
        """
        self.results[condition_num] = results

    def to_dict(self):
        groupsdict = {groupname: self.groups[groupname].to_dict() for groupname in self.groups}
        outdict = dict(
            name=self.name,
            default_condition=self.default_condition.to_dict(),
            groups=groupsdict,
            group_ordering=self.group_ordering,
            results=self.results
        )
        return outdict

    @staticmethod
    def from_dict(experiment_dict):
        default_condition = ExperimentCondition()
        default_condition.from_dict(experiment_dict['default_condition'])
        expt = Experiment(experiment_dict['name'], default_condition)
        expt.default_condition = default_condition
        expt.results = experiment_dict['results']
        for groupname in ['base']+experiment_dict['group_ordering']:
            groupdict = experiment_dict['groups'][groupname]
            if groupname != 'base':
                parentname = groupdict['parentname']
                expt.add_group(groupname, parentname)
            for name, values in groupdict['factors'].iteritems():
                expt.add_factor(name, values, groupname)
        return expt

    def add_group(self, groupname, parentname='base'):
        if groupname in self.groups:
            raise RuntimeError("group %s already exists! Each group name must be unique." % groupname)
        parent = self.groups[parentname]
        self.groups[groupname] = ExperimentGroup(groupname, parent)
        if parentname in self.leaves:
            del(self.leaves[parentname])
        self.leaves[groupname] = self.groups[groupname]
        self.group_ordering.append(groupname)

    def add_factor(self, name, values, groupname='base'):
        if not hasattr(self.default_condition, name):
            raise RuntimeError("name %s is not an attribute of default experiment condition!" % name)
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
                items = dict(condition_num=n, groupname=groupname)
                if changes_only:
                    items['changes'] = self.groups[groupname].get_nth_condition(n_offset, cond)
                else:
                    items['condition'] = self.groups[groupname].get_nth_condition(n_offset, cond)
                return items

    def get_conditions(self, numlist, changes_only=False):
        condition_list = []
        for n in numlist:
            condition_list.append(self.get_nth_condition(n, changes_only=changes_only))
        return condition_list

    def get_group_condition_iterator(self, groupname, changes_only=False, start=0, stop=None):
        cond = None if changes_only else self.default_condition
        return self.groups[groupname].get_condition_iterator(cond, start, stop)

    def get_all_condition_iterator(self, changes_only=False, start=0, stop=None):
        stop = self.get_num_conditions() if stop is None else stop
        for i in range(start, stop):
            yield self.get_nth_condition(i, changes_only=changes_only)

    def get_all_condition_changes_dict(self):
        return dict([(i, item) for i, item in enumerate(
            self.get_all_condition_iterator(changes_only=True))])

    def get_condition_numbers(self, fixed_dict, groupname=None):
        def _filter_func(items):
            filter_list = [items['changes'][key] == val for key, val in fixed_dict.iteritems()]
            if groupname is not None:
                filter_list.append(items['groupname']==groupname)
            return all(filter_list)
        all_it = self.get_all_condition_iterator(changes_only=True)
        filtered_it = ifilter(_filter_func, all_it)
        return [c['condition_num'] for c in filtered_it]

    def _populate_default_factors(self):
        for key in self.default_condition.to_dict():
            val = getattr(self.default_condition, key)
            self.add_factor(key, val)


# private helper function
def _convert_property_value(value):
    if hasattr(value, 'to_dict'):
        value = value.to_dict()
    elif hasattr(value, 'func_name'):
        value = value.func_name
    return value
