import xnn
import numpy as np
import cPickle
import theano
import theano.tensor as T
from collections import OrderedDict
from xnn import layers

__all__=['Model']


class Model(object):
    """
    The Model class stores the layers and connectivity structure of the model, and
    provides utility functions for building a network from lasagne layers.  The
    Model also provides methods for associating network inputs and outputs with
    fields in the data dictionary.  One advantage of using the Model class over
    lasagne layers is that the Model provides serialization and de-serialization.

    """

    def __init__(self,name='model'):
        self.name    = name
        self.layers  = OrderedDict()
        self.inputs  = OrderedDict()
        self.outputs = OrderedDict()
        self.eval_outputs = OrderedDict()
        self._predict_func = None
        self._unique_name_counters = dict()


    def add_layer(self,layer,name=None):
        """
        Add a layer to the model.  Layers should be added in a topological order: all inputs to a layer (unless it is an input layer) should be added to the model first.

        :param layer: A lasagne layer object
        :param name: The name of the layer is used to index the output of that layer in the data returned from the predict method.  Although names are generated automatically, it is generally a good idea to keep track of the names of layers whose output is important.

        :return: The lasagne layer object that was added to the network.
        """
        if name is None and layer.name is not None:
            name = self._get_unique_name_from_layer(layer)
        if layer.name is None and name is not None:
            layer.name = self._get_unique_name(name)
            name = layer.name
        if layer.name is None and name is None:
            layer.name = self._get_unique_name_from_layer(layer)
            name = layer.name
        else: 
            name = self._get_unique_name(name)
            layer.name=name
        self.layers[name]=layer
        return layer

    def add_full_net_from_layer(self,outlayer):
        """
        Create a model from a single output layer.  This will not work for multi-output models.

        :param outlayer: A lasagne layer object.
        """
        layers = xnn.layers.get_all_layers(outlayer)
        for l in layers:
            self.add_layer(l)

    def _get_unique_name_from_layer(self,layer,namebase=''):
        if layer.name is not None:
            name = self._get_unique_name(namebase+layer.name)
        else:
            name = self._get_unique_name(namebase+layer.__class__.__name__)
        return name

    def _get_unique_name(self,namebase):
        if namebase in dir(layers):
            self._unique_name_counters.setdefault(namebase, 0)
            c = self._unique_name_counters[namebase]
            unique_name = namebase + '_%d' % c
            self._unique_name_counters[namebase] += 1
            return unique_name
        else:
            return namebase

    def make_dropout_layer(self,parentlayer,p=0.5,name=None,drop_type='standard'):
        """
        Convenience function to both construct a lasagne dropout layer and add it to the model.

        :param parentlayer: The lasagne layer object whose outputs are the input to the dropout layer
        :param p: The dropout parameter.  Depending on the dropout type, this parameter means different things.  For standard dropout, this is the percent of neurons to mask.  For Gaussian dropout, this is the variance of the gaussian noise.
        :param name: The name of the dropout layer.
        :param drop_type: Either 'standard or 'gauss' for standard and Gaussian dropout.
        
        :return: The lasagne layer object that was added to the network.
        """
        if drop_type == 'standard':
            droplayer  = xnn.layers.DropoutLayer(parentlayer,p=p)
        elif drop_type == 'gauss':
            droplayer  = xnn.layers.GaussianDropoutLayer(parentlayer,sigma=p)
        else:
            raise ValueError("drop_type must be 'standard' or 'gauss'")
        if name is None:
            name = self._get_unique_name_from_layer(droplayer)
            droplayer.name=name
        self.add_layer(droplayer,name=name)
        return droplayer

    def make_dense_layer(self,parentlayer,num_units,nonlinearity=None,name=None):
        """
        Convenience function to both construct a lasagne dense (fully-connected) layer and add it to the model.

        :param parentlayer: The lasagne layer object whose outputs are the input to the dense layer.
        :param num_units: The number of units in the dense layer.
        :param nonlinearity: The nonlinearity function to use on this layer.  If None, :py:meth:`xnn.nonlinearities.rectify` is used.
        :param name: The name of the dense layer.

        :return: The lasagne layer object that was added to the network.
        """
        if nonlinearity is None:
            nonlinearity = xnn.nonlinearities.rectify
        denselayer = xnn.layers.DenseLayer(parentlayer,num_units=num_units,nonlinearity=nonlinearity)
        if name is None:
            name = self._get_unique_name_from_layer(denselayer)
            denselayer.name = name
        self.add_layer(denselayer,name=name)
        return denselayer

    def make_bound_input_layer(self,shape,inputlabelkey,name=None,input_var=None):
        """
        Create an input layer, add it to the model, and bind its input to a field in the data.

        :param shape: The shape of the input (as a tuple) that this layer should expect.
        :param inputlabelkey: The key in the data dictionary whose value should be fed to the constructed input layer.
        :param name: The name of the input layer
        :param input_var: The theano symbol that represents the input to this layer in theano's symbolic graph.  If None, a symbol is created internally.
        
        :return: The lasagne layer object that was added to the network.
        """
        lin = xnn.layers.InputLayer(shape,input_var=input_var,name=name)
        if name is None:
            name = self._get_unique_name_from_layer(lin)
            lin.name = name
        self.add_layer(lin,name=name)
        self.bind_input(lin,inputlabelkey)
        return lin

    def make_dense_drop_stack(self,parent_layer,num_units_list=None,drop_p_list=None,nonlin_list=None,namebase=None,drop_type_list=None):
        """
        Create a sequence of dense layers followed by dropout layers.

        :param parent_layer: The lasagne layer object whose outputs are the input to the first dense layer.
        :param num_units_list: A list of integers specifying the number of units.  The length of this list is the number of dense layers that will be added to the model.
        :param drop_p_list: A list of parameters for the dropout layers.  The length should be the same as :py:attr:`num_units_list`, :py:attr:`drop_type_list`, and :py:attr:`nonlin_list`.  If None, all values are 0.5.
        :param nonlin_list: A list of nonlinearities to apply to the dense layers.  If None, all nonlinearities will be :py:meth:`xnn.nonlinearities.rectify`
        :param namebase: Base name to prepend to dense and dropout layer names.
        :param drop_type_list: A list of strings specifying either 'standard' or 'gauss' for the dropout types of each layer in the stack. If None, all types are 'standard'.
        
        :return: The lasagne layer object of the final layer added to the network by this method.
        """
        pl = parent_layer
        # if namebase is None:
        #     namebase="l_"
        for i in xrange(len(num_units_list)):
            nhu        = num_units_list[i]
            p          = drop_p_list[i] if drop_p_list is not None else 0.5
            nl         = nonlin_list[i] if nonlin_list is not None else xnn.nonlinearities.rectify
            dt         = drop_type_list[i] if drop_type_list is not None else 'standard'
            if namebase is None:
                nameden = 'DenseLayer'
                namedro = 'DropoutLayer'
            else:
                nameden = namebase+'_DenseLayer_'
                namedro = namebase+'_DropoutLayer_'
            if nl == 'prelu':
                denselayer = self.make_dense_layer(pl,nhu,nonlinearity=xnn.nonlinearities.linear,name=nameden)
                prelu = self.add_layer(xnn.layers.PReLULayer(denselayer))
                droplayer  = self.make_dropout_layer(prelu,p=p,name=namedro,drop_type=dt)
            else:
                denselayer = self.make_dense_layer(pl,nhu,nonlinearity=nl,name=nameden)
                droplayer  = self.make_dropout_layer(denselayer,p=p,name=namedro,drop_type=dt)
            pl         = droplayer
        return pl

    def bind_input(self, input_layer, input_key):
        """
        Specifies that a particular model layer should be given input from the data dictionary.

        :param input_layer: The lasagne layer object to which to bind the input.
        :param input_key: The key in the data dictionary whose value will be passed to the input layer

        """
        if not isinstance(input_key, str):
            raise Exception("input_key must be a string")
        if not isinstance(input_layer, xnn.layers.InputLayer):
            raise Exception("input_layer must be an object of type InputLayer")
        self.inputs.setdefault(input_key, [])
        self.inputs[input_key].append(input_layer)

    def bind_output(self, output_layer, loss_function, target, target_type='label', aggregation_type='mean', scale=1.0, is_eval_output=False, weight_key=None):
        """
        Specifies that a particular model layer should be treated as an output, which data key holds the labels for this output, and the cost function used for this output.
        
        :param output_layer: The lasagne layer object that will be treated as an output.
        :param loss_function: The function that will be applied to this output and the target to compute the cost.
        :param target: The key in the data dictionary whose value is the target for this output, or the name of the layer whose output should be reconstructed.
        :param target_type: Either 'label', for when the target is a field in the data dictionary or 'recon', when the target is the output of another layer.
        :param aggregation_type: How to aggregate the cost across the examples in a batch.  Must be one of these values: 'mean', 'sum', 'weighted_mean', 'weighted_sum', 'nanmean', 'nansum', 'nanweighted_mean', 'nanweighted_sum'
        :param scale: Scale the cost for this output by a constant.
        :param is_eval_output: Specify whether this output should be added to a list of evaluation outputs.
        :param weight_key: Key in the data dictionary whose value is the weight of each example in the batch for this output.  Only specified if :py:attr:`aggregation_type` is a weighted type.

        """
        aggregation_types = ['mean', 'sum', 'weighted_mean','weighted_sum','nanmean','nansum','nanweighted_mean','nanweighted_sum']
        target_types = ['label', 'recon']
        if aggregation_type not in aggregation_types:
            raise ValueError("Invalid aggregation type. Expected one of: %s" % aggregation_types)
        if target_type not in target_types:
            raise ValueError("Invalid target type. Expected one of: %s" % target_types)
        if not isinstance(target, str):
            raise ValueError("Target must be a string")
        if (weight_key is not None) and (type(weight_key)!=str):
            raise ValueError("weight_key must be either None or a string")
        if ('weighted' in aggregation_type) and (weight_key is None):
            raise ValueError("Weighted aggregation types must have a weight key")
        if not (isinstance(scale,float) or isinstance(scale,int)):
            raise ValueError("Scale must be a float or an int")
        self.outputs[output_layer.name] = dict(
            output_layer     = output_layer,
            target           = target,
            target_type      = target_type,
            loss_function    = loss_function,
            aggregation_type = aggregation_type,
            weight_key       = weight_key,
            scale            = float(scale)
        )
        if is_eval_output:
            self.bind_eval_output(output_layer,target)

    def bind_eval_output(self, output_layer, target):
        """
        Add an output layer to a list of evaluation outputs.

        :param output_layer: The layer object whose output will be considered an evaluation output.
        :param target: The key in the data dictionary whose value is the target for this output.

        """
        if output_layer.name not in self.layers:
            raise Exception("Can only bind eval outputs to layers that exist in the model.")
        self.eval_outputs[output_layer.name]=dict(
            output_layer   = output_layer,
            target         = target
        )

    def to_dict(self):
        """
        Return a dictionary that represents this model.

        """
        d = {}
        ls = []
        for lname,l in self.layers.iteritems():
            if hasattr(l,'to_dict'):
                ldict = l.to_dict()
            else:
                ldict = self._layer_to_dict(lname,l)
            ls.append(ldict)

        inputs = OrderedDict()
        for iname,layers in self.inputs.iteritems():
            inputs.setdefault(iname, [])
            for layer in layers:
                inputs[iname].append(layer.name)

        outputs = OrderedDict()
        for oname, output in self.outputs.iteritems():
            if hasattr(output['loss_function'],'to_dict'):
                lfval = output['loss_function'].to_dict()
            else:
                lfval = output['loss_function'].func_name
            outputs[oname] = dict(
                loss_function    = lfval,
                output_layer     = output['output_layer'].name,
                target_type      = output['target_type'],
                target           = output['target'],
                aggregation_type = output['aggregation_type'],
                weight_key       = output['weight_key'],
                scale            = output['scale']
            )

        eval_outputs=OrderedDict()
        for oname, output in self.eval_outputs.iteritems():
            eval_outputs[oname]= dict(
                output_layer   = output['output_layer'].name,
                target         = output['target']
            )

        d['layers']  = ls
        d['inputs']  = inputs
        d['outputs'] = outputs
        d['eval_outputs'] = eval_outputs
        d['name']    = self.name
        return d

    def from_dict(self,indict):
        """
        Add layers, inputs, and outputs to the current model from a dictionary representation.

        :param indict: A dictionary representation of a model.
        """
        import copy
        indict = copy.deepcopy(indict)
        self._build_layers_from_list(indict['layers'])
        self._bind_inputs_from_list(indict['inputs'])
        self._bind_outputs_from_list(indict['outputs'])
        self._bind_eval_outputs_from_list(indict['eval_outputs'])

    def from_dict_static(indict): 
        """
        Construct a model from a dictionary representation

        :param indict: A dictionary representation of a model.
        :return: A model build from the dictionary.
        """
        m = Model(indict['name'])
        m._build_layers_from_list(indict['layers'])
        m._bind_inputs_from_list(indict['inputs'])
        m._bind_outputs_from_list(indict['outputs'])
        m._bind_eval_outputs_from_list(indict['eval_outputs'])
        return m
    from_dict_static = staticmethod(from_dict_static)
                
    def _layer_to_dict(self,lname,l): 
        ltype = type(l).__name__
        ldict = dict(name=lname,
                     layer_type=ltype)
        if hasattr(l,'input_layer'):
            iln = l.input_layer.name if l.input_layer is not None else None
            ldict['incoming']=iln
        elif hasattr(l,'input_layers'):
            iln = [ilay.name for ilay in l.input_layers]
            ldict['incomings']=iln

        directGetList = {'p','num_units','num_filters','filter_size','stride',
                         'untie_biases','border_mode','pool_size','img_shape',
                         'pad','ignore_border','axis','rescale','sigma',
                         'outdim','pattern','width','val','batch_ndim',
                         'indices','input_size','output_size','flip_filters',
                         'edgeprotect','mode','seed','prior','local_filters',
                         'dimshuffle','partial_sum','input_shape','output_shape','shape'}
        nameGetList   = {'nonlinearity','convolution','pool_function','merge_function'}

        for dga in directGetList:
            if hasattr(l,dga):
                ldict[dga] = getattr(l,dga)
        for nga in nameGetList:
            if hasattr(l,nga):
                at = getattr(l,nga)
                if at is not None:
                    if hasattr(at, "func_name"):
                        ldict[nga] = at.func_name
                    else:
                        ldict[nga] = at.__dict__.copy()
                        ldict[nga]['type'] = at.__class__.__name__
        return ldict

    def _build_layers_from_list(self,ll):
        nameGetList   = {'nonlinearity','convolution','pool_function','merge_function'}
        for lspec in ll:
            t = lspec['layer_type']
            lclass  = getattr(xnn.layers,t)
            if hasattr(lclass,'from_dict'):
                l = lclass.from_dict(lspec)
            else:
                linnames = []
                if 'incoming' in lspec:
                    linnames = [lspec['incoming']]
                elif 'incomings' in lspec:
                    linnames = lspec['incomings']

                lin = []
                for n in linnames:
                    lin.append(self.layers[n] if n is not None else None)
                lin     = lin[0] if len(lin)==1 else lin
                linit   = lclass.__init__
                largs   = linit.func_code.co_varnames[1:linit.func_code.co_argcount]
                argdict = dict(name=lspec['name'])
                for a in largs:
                    if a == 'incoming' or a == 'incomings':
                        argdict[a] = lin
                    elif a in nameGetList:
                        argdict[a] = self._initialize_arg(a,lspec[a])
                    elif a in lspec:
                        argdict[a] = lspec[a]
                l = lclass(**argdict)
            self.add_layer(l)

    def _bind_inputs_from_list(self,il):
        for labelkey, lnames in il.iteritems():
            for ln in lnames:
                self.bind_input(self.layers[ln],labelkey)

    def _bind_outputs_from_list(self,ol):
        for layername, outdict in ol.iteritems():
            l         = self.layers[layername]
            if isinstance(outdict['loss_function'],dict):
                f = xnn.objectives.from_dict(outdict['loss_function'])
            else:
                fname     = outdict['loss_function']
                f         = getattr(xnn.objectives,fname)
            wgt_key   = outdict['weight_key']
            targ      = outdict['target']
            targ_type = outdict['target_type']
            agg       = outdict['aggregation_type']
            scale     = outdict['scale']
            self.bind_output(l,f,targ,targ_type,agg,scale,weight_key=wgt_key)

    def _bind_eval_outputs_from_list(self,el):
        for layername, edict in el.iteritems():
            l = self.layers[layername]
            targ = edict['target']
            self.bind_eval_output(l,targ)

    def _initialize_arg(self,a,spec):
        #TODO: expand this to take care of other objects that need to be re-initialized
        if a == 'nonlinearity':
            if isinstance(spec,dict):
                nonlin = getattr(xnn.nonlinearities,spec.pop('type'))
                return nonlin(**spec)
            return getattr(xnn.nonlinearities,spec)
        if a == 'convolution':
            if spec == 'conv2d':
                return theano.tensor.nnet.conv2d
        else:
            return None

    def save_model(self,fname):
        """
        Save model to a file.

        :param fname: The filename into which the model will be saved.
        """
        d = self.to_dict()
        all_layers = [self.layers[k] for k in self.layers.keys()]
        p = xnn.layers.get_all_param_values(all_layers)
        m = dict(model=d,params=p)
        with open(fname,'wb') as f:
            cPickle.dump(m,f,cPickle.HIGHEST_PROTOCOL)

    def load_model(self,fname):
        """
        Load model from a file.

        :param fname: The filename that contains the model to be loaded.
        """
        with open(fname,'rb') as f:
            d = cPickle.load(f)
        self.from_dict(d['model'])
        all_layers = [self.layers[k] for k in self.layers.keys()]
        xnn.layers.set_all_param_values(all_layers,d['params'])

    def _get_tensor(self,layer):
        variable_type_dict = {
            2:T.matrix,
            3:T.tensor3,
            4:T.tensor4
        }
        ldim = len(layer.shape)
        assert ldim in variable_type_dict
        var_type = variable_type_dict[ldim]
        return var_type(layer.name)

    def _get_predict(self,data_dict,data_in_gpu):
        ins = []
        givens = dict()
        if data_in_gpu:
            data_dict = dict([(self.inputs[k][i],data_dict[k]) for k in self.inputs for i in range(len(self.inputs[k]))])
        else:
            data_dict = dict()
            for k in self.inputs:
                for l in self.inputs[k]:
                    i = self._get_tensor(l)
                    ins.append(i)
                    data_dict[l]=i
        outs = xnn.layers.get_output(self.layers.values(),inputs=data_dict,deterministic=True)
        f = theano.function(ins,outs,no_default_updates=True)
        self._predict_func = f

    def predict(self,data_dict,layer_names=None,data_in_gpu=False):
        """
        Run data through a model and collect the output.

        :param data_dict: A dictionary of data on which the model will run.  This dictionary must contain keys whose values were bound to the input layers of the model.
        :param layer_names: A list of layer names for which the output will be returned.  If None, output of all layers will be returned
        :param data_in_gpu: If True, data is on the gpu and the values in :py:attr:`data_dict` are the theano shared variables that contain the data.  If False, the values in :py:attr:`data_dict` are numpy arrays containing the data
        """
        f = self._predict_func
        if f is None:
            self._get_predict(data_dict,data_in_gpu)
            f = self._predict_func
        ins = []
        if not data_in_gpu:
            if isinstance(data_dict,dict):
                for k in self.inputs:
                    for l in self.inputs[k]:
                        ins.append(data_dict[k])
            else:
                ins = [data_dict] * len(self.inputs)
        outs = f(*ins)
        if layer_names is None:
            layer_names = self.layers.keys()
        outDict=dict()
        for i,ln in enumerate(self.layers.keys()):
            if ln in layer_names:
                outDict[ln]=outs[i]
        return outDict
