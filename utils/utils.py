import numpy as np
import theano 
import theano.tensor as T
import time
import sys
import pylab as pl
import signal
import pydot
from PIL import Image
from ..layers import Layer

def theano_digitize(x, bins):
    """
    Equivalent to numpy digitize.

    Parameters
    ----------
    x : Theano tensor or array_like
        The array or matrix to be digitized
    bins : array_like
        The bins with which x should be digitized

    Returns
    -------
    A Theano tensor
        The indices of the bins to which each value in input array belongs.
    """
    binned = T.zeros_like(x) + len(bins)
    for i in range(len(bins)):
        bin=bins[i]
        if i == 0:
            binned=T.switch(T.lt(x,bin),i,binned)
        else:
            ineq = T.and_(T.ge(x,bins[i-1]),T.lt(x,bin))
            binned=T.switch(ineq,i,binned)
    binned=T.switch(T.isnan(x), len(bins), binned)
    return binned

def numpy_one_hot(x,numclasses=None):
    """
    Changes an array of values to one-hot encoding

    Parameters
    ----------
    x : Array_like of ints
        The array to be one-hot encoded
    numclasses : int or None, optional
        The number of classes to use for encoding.
        If int, must be at least of length x.max()
        If None or default, max(x)+1 will be used

    Returns
    -------
    An array_like
        Array of shape (x.shape,numclasses)
        The one-hot encoded version of x
    """
    if type(x) == list:
        x = np.array(x)
    if x.shape==():
        x = x[None]
    if numclasses is None:
        numclasses = x.max() + 1
    result = np.zeros(list(x.shape) + [numclasses], dtype=theano.config.floatX)
    z = np.zeros(x.shape, dtype=theano.config.floatX)
    for c in range(numclasses):
        z *= 0
        z[np.where(x==c)] = 1
        result[...,c] += z
    return result

def Tnanmean(x,axis=None,keepdims=False):
    """
    Theano version of numpy nanmean.

    Parameters
    ----------
    x : Theano.tensor or array_like
        The array to be nanmeaned
    axis : int or None, optional
        Axis to do nanmean along.
        If None or default, nanmean is done over entire array
    keepdims : Boolean, optional
        If this is set to True, the axes which are reduced are left in the result as dimensions with size one.


    Returns
    -------
    An array-like
        Nan is returned for slices that contain only NaNs.
    """
    mask1 = T.switch(T.isnan(x),0,x)
    mask2 = T.switch(T.isnan(x),0,1)
    return T.true_div(mask1.sum(axis=axis,keepdims=keepdims),mask2.sum(axis=axis,keepdims=keepdims))

def Tnanmax(x,axis=None,keepdims=False):
    """
    Theano version of numpy nanmax.

    Parameters
    ----------
    x : Theano.tensor or array_like
        The array to be nanmaxed
    axis : int or None, optional
        Axis to do nanmax along.
        If None or default, nanmax is done over entire array
    keepdims : Boolean, optional
        If this is set to True, the axes which are reduced are left in the result as dimensions with size one.


    Returns
    -------
    An array-like
        Nan is returned for slices that contain only NaNs.
    """
    mask = T.switch(T.isnan(x),-np.inf,x)
    return mask.max(axis=axis,keepdims=keepdims)

def Tnansum(x, axis=None, keepdims=False):
    """
    Theano version of numpy nansum.

    Parameters
    ----------
    x : Theano.tensor or array_like
        The array to be nansumed
    axis : int or None, optional
        Axis to do nansum along.
        If None or default, nansum is done over entire array
    keepdims : Boolean, optional
        If this is set to True, the axes which are reduced are left in the result as dimensions with size one.


    Returns
    -------
    An array-like
        Nan is returned for slices that contain only NaNs.
    """
    mask = T.switch(T.isnan(x),0,x)
    return mask.sum(axis=axis,keepdims=keepdims)

class Progbar(object):
    '''
    A progress bar with eta shown.

    Parameters
    ----------
    target : int
        Total number of steps expected
    width : int, optional
        Width of progress bar. Default is 30 characters
    verbose : int, 1 or 2, optional
        TODO: Write docstring

    Notes
    -----
    Once the Progbar object has been created, update it by using the add function each iteration
    '''
    def __init__(self, target, width=30, verbose=1):
        self.width = width
        self.target = target
        self.sum_values = {}
        self.unique_values = []
        self.start = time.time()
        self.total_width = 0
        self.seen_so_far = 0
        self.verbose = verbose

    def _update(self, current, values=[]):
        '''
            @param current: index of current step
            @param values: list of tuples (name, value_for_last_step).
            The progress bar will display averages for these values.
        '''
        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [v, 1]
                self.unique_values.append(k)
            else:
                if type(v)==str:
                    self.sum_values[k][0] = v
                else:
                    self.sum_values[k][0] += v * (current-self.seen_so_far)
                    self.sum_values[k][1] += (current-self.seen_so_far)

        now = time.time()
        if self.verbose == 1:
            prev_total_width = self.total_width
            sys.stdout.write("\b" * (self.total_width+1))
            sys.stdout.write("\r")

            bar = '%d/%d [' % (current, self.target)
            prog = float(current)/self.target
            prog_width = int(self.width*prog)
            if prog_width > 0:
                bar += ('='*(prog_width-1))
                if current < self.target:
                    bar += '>'
                else:
                    bar += '='
            bar += ('.'*(self.width-prog_width))
            bar += ']'
            sys.stdout.write(bar)
            self.total_width = len(bar)

            if current:
                time_per_unit = (now - self.start) / current
            else:
                time_per_unit = 0
            eta = time_per_unit*(self.target - current)
            info = ''
            if current < self.target:
                info += ' - ETA: %ds' % eta
            else:
                info += ' - %ds' % (now - self.start)
            for k in self.unique_values:
                if type(k[0]) == str:
                    info += ' - %s: %s' % (k, self.sum_values[k][0])
                else:
                    info += ' - %s: %.4f' % (k, self.sum_values[k][0]/self.sum_values[k][1])

            self.total_width += len(info)
            if prev_total_width > self.total_width:
                info += ((prev_total_width-self.total_width) * " ")

            sys.stdout.write(info)
            sys.stdout.flush()
            self.seen_so_far = current

            if current >= self.target:
                sys.stdout.write("\n")

        if self.verbose == 2:
            if current >= self.target:
                info = '%ds' % (now - self.start)
                for k in self.unique_values:
                    info += ' - %s: %.4f' % (k, self.sum_values[k][0]/self.sum_values[k][1])
                sys.stdout.write(info + "\n")

    def add(self, n, values=[]):
        '''
        Update progress bar

        Parameters
        ----------
        n : int,
            Number of steps since last update
        values : list of tuples, optional
            List of (key, value) pairs to be shown with the progress bar.
            The key is a string to be used as the label.
            The value can be a float or a string
            If it is a string, the string will be displayed
            If it is a float, the average value for this key will be displayed
        '''
        self._update(self.seen_so_far+n, values)

class GracefulStop(object):
    '''
    TODO: Add docstring
    '''
    def __enter__(self):
        self.signal_received = False
        self.old_handler = signal.getsignal(signal.SIGINT)
        self.count=0
        signal.signal(signal.SIGINT,self.handler)

    def handler(self,signal,frame):
        self.count+=1
        self.signal_received = (signal,frame)
        if self.count == 1:
            print "---Cancelling training after this epoch. To cancel immediately, press ctrl-C again---"
        else:
            signal.signal(signal.SIGINT,self.old_handler)

    def __exit__(self,type,value,traceback):
        signal.signal(signal.SIGINT,self.old_handler)
        if self.signal_received:
            self.old_handler(*self.signal_received)

class _lengthExpection(Exception):
    pass

class _nonNegativeExpection(Exception):
    pass

class _noProbVectorException(Exception):
    pass


def probEmbarrasingMistakeForAge(fHuman,pMachine,bins=[18,25,35,45,55,65,100]):
    """
    TODO: Write docstring
    """
    # compute probabiliity of embarrasing mistake for age estimation
    # fHuman is a n-by-100 dimensional vector of human estimates of age
    # fHuman[j][k]=10 means that 10 people thought the age of person j was k
    # pMachine is probability of machine producing an age estimate within a specific interval
    # pMachine[j][0]=0.1  means that 10% of time the machine estimate the age of the person j is in category 0
    # the  upper limits in years for each of the categories are in an array defined in bins

    categoryUpperLims=bins
    nc=len(categoryUpperLims)
    if type(fHuman)==list:
        fHuman = array(fHuman)
    if type(pMachine)==list:
        pMachine = array(pMachine)
    if fHuman.shape[1] != 100:
        raise _lengthExpection('vector of human age estimates is not of dimension 100')
    if pMachine.shape[1]!=nc:
        raise _lengthExpection('vector of probability estimates is not of dimension expected:', nc)
    if sum(fHuman<0)>0:
        raise _nonNegativeExpection('vector of human age estimates has negative values')
    if sum(pMachine<0)>0:
        raise _nonNegativeExpection('vector of machine probabilities contains negative values')
    if pMachine.sum()!=pMachine.shape[0]:
        raise _nonNegativeExpection('vector of machine probabilities does not add up to one')

    # use frequency of human labels to do probabiity density estimation
    # with Gaussian kernels

    pHuman = zeros(fHuman.shape)
    for k1 in range(100):
        for k2 in range(100):
            d2= (k2-k1)**2.
            sd = 0.2*(k2+1.)  # slack in the age estimate is 20 % of the estimate
            s2 = sd**2.
            pHuman[:,k1] = pHuman[:,k1] + exp(-0.5*d2/s2) * fHuman[:,k2]/(sd+0.)

    pHuman = pHuman/pHuman.sum(1,keepdims=True)
    # based on the year by year probabiilty estimates
    # compute the probability of each age category

    pHumanC=zeros(pMachine.shape)
    c1=categoryUpperLims[0]+1
    c0=0
    c1=categoryUpperLims[0]
    pHumanC[:,0]=pHuman[:,0:c1+1].sum(1)
    for k in range(1,nc):
        c0=categoryUpperLims[k-1]+1
        c1=categoryUpperLims[k]
        pHumanC[:,k]=pHuman[:,c0:c1+1].sum(1)


    # a machine probability estimate is not embarrasing if either one of the two people agrees with us, or
    # the two people provide estimates that bracket our estimate. If X1 and X2 are the two human estimates
    # and y is the machine estimate then
    # prob(no embarrasing | y)= p(X1 <= y)p(X2 >= y)

    # Compute the expected probability of embarrasing (1 - prob OK)
    probOK=zeros((fHuman.shape[0],))
    for y in range(nc):
    #y is the index to the category chosen by humans
        Fy = pHumanC[:,0:y+1].sum(1)
        Gy = (1.-Fy)+pHumanC[:,y]
        probOK = probOK+pMachine[:,y]*Fy*Gy
    return 1.-probOK


def typechecker(f):
    """
    Function decorator for functions which return theano variables.
    If no inputs are theano variables, eval() is called on the output.

    Parameters
    ----------
    f : function
        Function to be decorated
    """
    def typecheck(*args,**kwargs):
        all_args = list(args)+kwargs.values()
        tmp = T.matrix()
        for a in all_args:
            if type(a)==type(tmp):
                return f(*args,**kwargs)
        return f(*args,**kwargs).eval()
    tc = typecheck
    tc.__name__ = f.__name__
    return tc

def draw_to_file(model, filename, **kwargs):
    """
    Draws a network diagram to a file

    Parameters
    ----------
    model : list or :class:`Model` instance
       List of the layers, as obtained from lasange.layers.get_all_layers, or model object
    filename : string
       The filename to save output to.
    **kwargs : see docstring of _get_pydot_graph for other options
    """
    def _get_hex_color(layer_type):
        """
        Determines the hex color for a layer. Some classes are given
        default values, all others are calculated pseudorandomly
        from their name.
        :parameters:
            - layer_type : string
                Class name of the layer

        :returns:
            - color : string containing a hex color.

        :usage:
            >>> color = _get_hex_color('MaxPool2DDNN')
            '#9D9DD2'
        """

        if 'Input' in layer_type:
            return '#A2CECE'
        if 'Conv' in layer_type:
            return '#7C9ABB'
        if ('Dense' in layer_type) or ('Local' in layer_type):
            return '#6CCF8D'
        if 'Pool' in layer_type:
            return '#9D9DD2'
        else:
            return '#{0:x}'.format(hash(layer_type) % 2**24)


    def _get_pydot_graph(layers, output_shape=True, verbose=False):
        """
        Creates a PyDot graph of the network defined by the given layers.
        :parameters:
            - layers : list
                List of the layers, as obtained from lasange.layers.get_all_layers
            - output_shape: (default `True`)
                If `True`, the output shape of each layer will be displayed.
            - verbose: (default `False`)
                If `True`, layer attributes like filter shape, stride, etc.
                will be displayed.
            - verbose:
        :returns:
            - pydot_graph : PyDot object containing the graph

        """
        pydot_graph = pydot.Dot('Network', graph_type='digraph')
        pydot_nodes = {}
        pydot_edges = []
        for i, layer in enumerate(layers):
            layer_name = layer.name
            layer_type = '{0}'.format(layer.__class__.__name__)
            key = repr(layer)
            label = layer_name + '(' + layer_type + ')'
            color = _get_hex_color(layer_type)
            eol='\n'
            if verbose:
                for attr in ['num_filters', 'num_units', 'ds',
                             'filter_shape', 'stride', 'strides', 'p', 'shape']:
                    if hasattr(layer, attr):
                        label += eol + \
                            '{0}: {1}'.format(attr, getattr(layer, attr))
                if hasattr(layer, 'nonlinearity'):
                    try:
                        nonlinearity = layer.nonlinearity.__name__
                    except AttributeError:
                        nonlinearity = layer.nonlinearity.__class__.__name__
                    label += eol + 'nonlinearity: {0}'.format(nonlinearity)

            if output_shape:
                label += eol + \
                    '({0})'.format(layer.output_shape)

            nodeshape = 'box'
            if 'Input' in layer_type:
                nodeshape = "ellipse"
            pydot_nodes[key] = pydot.Node(key,
                                          label=label,
                                          shape=nodeshape,
                                          fillcolor=color,
                                          style='filled',
                                          )


            if hasattr(layer, 'input_layers'):
                for input_layer in layer.input_layers:
                    pydot_edges.append([repr(input_layer), key])

            if hasattr(layer, 'input_layer'):
                pydot_edges.append([repr(layer.input_layer), key])

        for node in pydot_nodes.values():
            pydot_graph.add_node(node)
        for edge in pydot_edges:
            pydot_graph.add_edge(
                pydot.Edge(pydot_nodes[edge[0]], pydot_nodes[edge[1]]))
        return pydot_graph
    if isinstance(model,Layer):
        model = [model]
    elif hasattr(model,'layers'):
        model = model.layers.values()
    elif not isinstance(model,list):
        raise TypeError("model must be a Layer or list of Layers, or a Model object")
    dot = _get_pydot_graph(model, **kwargs)
    dot.get_node("")
    ext = filename[filename.rfind('.') + 1:]
    with open(filename, 'w') as fid:
        fid.write(dot.create(format=ext))

    im = Image.open(filename)
    return im