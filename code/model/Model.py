import lasagne
from lasagne.layers import get_output
import theano.tensor as T


class Model():
    def __init__(self,name=None):
        self.name=name
        self.layers = {}
        self.inputs = {}
        self.outputs = {}

    def addLayer(self,layer,name=None):
        if name is None and layer.name is not None:
            name = layer.name
        if layer.name is None and name is not None:
            layer.name = name
        if (layer.name is None and name is None) or (name!=layer.name):
            raise Exception("Layer must have a consistent name")
        self.layers[name]=layer
        return layer

    def addDenseDropStack(self,parent_layer,num_hidden_list=None,drop_p_list=None,nonlin_list=None,namebase=None):
        pl = parent_layer
        if namebase is None:
            namebase="l_"
        for i in xrange(len(num_hidden_list)):
            nhu = num_hidden_list[i]
            if drop_p_list is not None:
                p = drop_p_list[i]
            else:
                # default dropout value
                p = 0.5
            if nonlin_list is not None:
                nl = nonlin_list[i]
            else:
                # default nonlinearity
                nl = lasagne.nonlinearities.rectify
            denselayer = lasagne.layers.dense.DenseLayer(pl,num_units=nhu,nonlinearity=nl)
            self.addLayer(denselayer,name=namebase+'_dense_'+str(i))
            droplayer  = lasagne.layers.noise.DropoutLayer(denselayer,p=p)
            self.addLayer(droplayer,name=namebase+'_drop_'+str(i))
            pl = droplayer
        return pl

    def bindInput(self, input_key, input_layer):
        if not isinstance(input_key, str):
            raise Exception("input_key must be a string")
        if not isinstance(input_layer, lasagne.layers.input.InputLayer):
            raise Exception("input_layer must be an object of type InputLayer")
        self.inputs.setdefault(input_key, [])
        self.inputs[input_key].append(input_layer)

    def bindOutput(self, binding_name, output_layer, loss_function, target, target_type='label'):
        target_types = ['label', 'recon']
        if target_type not in target_types:
            raise ValueError("Invalid target type. Expected one of: %s" % target_types)
        if (target_type == 'label') and (not isinstance(target, str)):
            raise ValueError("target must be a string if target type is label")
        if (target_type == 'recon') and (not isinstance(target, lasagne.layers.base.Layer)):
            raise ValueError("target must be a Layer object if target type is recon")
        self.outputs[binding_name] = dict(
            output_layer=output_layer,
            target=target,
            target_type=target_type,
            loss_function=loss_function
        )

    def to_dict(self):
        d = {}
        ls = []
        for lname,l in self.layers.iteritems():
            ltype = l.__class__.__name__
            if type(l) == lasagne.layers.input.InputLayer:
                ls.append(dict(name=lname,
                               shape=l.shape,
                               layer_type=ltype))
                continue
            iln = l.input_layer.name if l.input_layer is not None else None
            ldict = dict(name=lname,
                           input_layer=iln,
                           input_shape=l.input_shape,
                           output_shape=l.output_shape,
                           layer_type=ltype)
            if type(l) == lasagne.layers.noise.DropoutLayer:
                ldict['p'] = l.p
            if type(l) == lasagne.layers.dense.DenseLayer:
                ldict['nonlinearity'] = l.nonlinearity
            ls.append(ldict)

        inputs = dict()
        for iname,layers in self.inputs.iteritems():
            inputs.setdefault(iname, [])
            for layer in layers:
                inputs[iname].append(layer.name)

        outputs = dict()
        for oname, output in self.outputs.iteritems():
            target = output['target']
            if output['target_type'] == 'recon':
                target = target.name
            outputs[oname] = dict(
                loss_function=output['loss_function'].func_name,
                output_layer=output['output_layer'].__class__.__name__,
                target_type=output['target_type'],
                target=target
            )

        d['layers'] = ls
        d['inputs'] = inputs
        d['outputs'] = outputs
        d['name'] = self.name
        return d





