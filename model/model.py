import xnn
import numpy as np
import cPickle
import theano.tensor as T
from collections import OrderedDict

__all__=['Model']

class Model():
    def __init__(self,name=None):
        self.name    = name
        self.layers  = OrderedDict()
        self.inputs  = OrderedDict()
        self.outputs = OrderedDict()

    def add_layer(self,layer,name=None):
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
        layers = xnn.layers.get_all_layers(outlayer)
        for l in layers:
            self.add_layer(l)

    def _get_unique_name_from_layer(self,layer,namebase=''):
        if layer.name is not None:
            name = self._get_unique_name(namebase+layer.name)
        else:
            name = self._get_unique_name(namebase+layer.__class__.__name__)
        return name

    def _get_unique_name(self,namebase,counter=0):
        while namebase in self.layers.keys():
            namebase+= '_'+str(counter)
        return namebase

    def make_dropout_layer(self,parentlayer,p=0.5,name=None,drop_type='standard'):
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

    def make_dense_layer(self,parentlayer,num_hidden,nonlinearity=None,name=None):
        if nonlinearity is None:
            nonlinearity = xnn.nonlinearities.rectify
        denselayer = xnn.layers.DenseLayer(parentlayer,num_units=num_hidden,nonlinearity=nonlinearity)
        if name is None:
            name = self._get_unique_name_from_layer(denselayer)
            denselayer.name = name
        self.add_layer(denselayer,name=name)
        return denselayer

    def make_bound_input_layer(self,shape,inputlabelkey,name=None,input_var=None):
        lin = xnn.layers.InputLayer(shape,input_var=input_var,name=name)
        if name is None:
            name = self._get_unique_name_from_layer(lin)
            lin.name = name
        self.add_layer(lin)
        self.bind_input(lin,inputlabelkey)
        return lin

    def make_dense_drop_stack(self,parent_layer,num_hidden_list=None,drop_p_list=None,nonlin_list=None,namebase=None,drop_type_list=None):
        pl = parent_layer
        if namebase is None:
            namebase="l_"
        for i in xrange(len(num_hidden_list)):
            nhu        = num_hidden_list[i]
            p          = drop_p_list[i] if drop_p_list is not None else 0.5
            nl         = nonlin_list[i] if nonlin_list is not None else xnn.nonlinearities.rectify
            dt         = drop_type_list[i] if drop_type_list is not None else 'standard'
            nameden    = self._get_unique_name(namebase+'_dense_'+str(i),counter=i) 
            namedro    = self._get_unique_name(namebase+'_drop_'+str(i),counter=i)
            denselayer = self.make_dense_layer(pl,nhu,nonlinearity=nl,name=nameden)
            droplayer  = self.make_dropout_layer(denselayer,p=p,name=namedro,drop_type=dt)
            pl         = droplayer
        return pl

    def bind_input(self, input_layer, input_key):
        if not isinstance(input_key, str):
            raise Exception("input_key must be a string")
        if not isinstance(input_layer, xnn.layers.InputLayer):
            raise Exception("input_layer must be an object of type InputLayer")
        self.inputs.setdefault(input_key, [])
        self.inputs[input_key].append(input_layer)

    def bind_output(self, output_layer, loss_function, target, target_type='label', aggregation_type='mean', weight_key=None):
        aggregation_types = ['mean', 'sum', 'weighted_mean','weighted_sum']
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
        self.outputs[output_layer.name] = dict(
            output_layer     = output_layer,
            target           = target,
            target_type      = target_type,
            loss_function    = loss_function,
            aggregation_type = aggregation_type,
            weight_key       = weight_key
        )

    def to_dict(self):
        d = {}
        ls = []
        for lname,l in self.layers.iteritems():
            ltype = type(l).__name__
            ldict = dict(name=lname,
                         layer_type=ltype)
            if hasattr(l,'input_layer'):
                iln = l.input_layer.name if l.input_layer is not None else None
                ldict['incoming']=iln
            elif hasattr(l,'input_layers'):
                iln = [ilay.name for ilay in l.input_layers]
                ldict['incomings']=iln

            directGetList = {'p','num_units','num_filters','stride',
                             'untie_biases','border_mode','pool_size',
                             'pad','ignore_border','axis','rescale','sigma',
                             'outdim','pattern','width','val','batch_ndim',
                             'indices','input_size','output_size','flip_filters',
                             'dimshuffle','partial_sum','input_shape','output_shape','shape'}
            nameGetList   = {'nonlinearity','convolution','pool_function','merge_function'}

            for dga in directGetList:
                if hasattr(l,dga):
                    ldict[dga] = getattr(l,dga)
            for nga in nameGetList:
                if hasattr(l,nga):
                    at = getattr(l,nga)
                    if at is not None:
                        ldict[nga]=at.__name__
            
            ls.append(ldict)

        inputs = dict()
        for iname,layers in self.inputs.iteritems():
            inputs.setdefault(iname, [])
            for layer in layers:
                inputs[iname].append(layer.name)

        outputs = dict()
        for oname, output in self.outputs.iteritems():
            target = output['target']
            outputs[oname] = dict(
                loss_function    = output['loss_function'].func_name,
                output_layer     = output['output_layer'].name,
                target_type      = output['target_type'],
                target           = target,
                aggregation_type = output['aggregation_type']
            )

        d['layers']  = ls
        d['inputs']  = inputs
        d['outputs'] = outputs
        d['name']    = self.name
        return d

    def from_dict(self,indict): 
        self._build_layers_from_list(indict['layers'])
        self._bind_inputs_from_list(indict['inputs'])
        self._bind_outputs_from_list(indict['outputs'])

    def _build_layers_from_list(self,ll):
        nameGetList   = {'nonlinearity','convolution','pool_function','merge_function'}
        for lspec in ll:
            t = lspec['layer_type']

            linnames = []
            if 'incoming' in lspec:
                linnames = [lspec['incoming']]
            elif 'incomings' in lspec:
                linnames = lspec['incomings']

            lin = []
            for n in linnames:
                lin.append(self.layers[n] if n is not None else None)
            lin     = lin[0] if len(lin)==1 else lin
            lclass  = getattr(xnn.layers,t)
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
            fname     = outdict['loss_function']
            f         = getattr(xnn.objectives,fname)
            targ      = outdict['target']
            targ_type = outdict['target_type']
            agg       = outdict['aggregation_type']
            self.bind_output(l,f,targ,targ_type,agg)


    def _initialize_arg(self,a,spec):
        #TODO: expand this to take care of other objects that need to be re-initialized
        if a == 'nonlinearity':
            return getattr(xnn.nonlinearities,spec)
        else:
            return None

    def save_model(self,fname):
        d = self.to_dict()
        all_layers = [self.layers[k] for k in self.layers.keys()]
        p = xnn.layers.get_all_param_values(all_layers)
        m = dict(model=d,params=p)
        with open(fname,'wb') as f:
            cPickle.dump(m,f,cPickle.HIGHEST_PROTOCOL)

    def load_model(self,fname):
        with open(fname,'rb') as f:
            d = cPickle.load(f)
        self.from_dict(d['model'])
        all_layers = [self.layers[k] for k in self.layers.keys()]
        xnn.layers.set_all_param_values(all_layers,d['params'])

    def predict(self,datadict,layer_names=None):
        if layer_names is None:
            layer_names = self.layers.keys()
            layers = self.layers.values()
        else:
            layers = []
            if type(layer_names) == str:
                layer_names = [layer_names]
            for layer_name in layer_names:
                assert layer_name in self.layers
                layers.append(self.layers[layer_name])

        if isinstance(datadict,dict):
            X = dict()
            for labelkey,layerlist in self.inputs.iteritems():
                for layer in layerlist:
                    X[layer]=datadict[labelkey]
        else:
            X = datadict

        outs = xnn.layers.get_output(layers,inputs=X,deterministic=True)
        outs = dict([(layer_name,out.eval()) for layer_name,out in zip(layer_names,outs)])
        return outs

