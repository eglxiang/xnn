import lasagne
import theano
import theano.tensor as T
from xnn import layers
from collections import OrderedDict

class ParamUpdateSettings():
    def __init__(self,
                 update=lasagne.updates.nesterov_momentum,
                 learning_rate=0.01,
                 momentum=0.9,
                 ):
        self.update        = update
        self.learning_rate = learning_rate
        self.momentum      = momentum

    def to_dict(self):
        properties           = self.__dict__.copy()
        properties['update'] = properties['update'].func_name
        return properties

class TrainerSettings(object):
    def __init__(self,
                 global_update_settings=ParamUpdateSettings(),
                 batch_size=128,
                 dataSharedVarDict=None, **kwargs):
        self.global_update_settings = global_update_settings
        self.batch_size             = batch_size
        self.dataSharedVarDict      = dataSharedVarDict
        # self.update_dict = {}
        self.__dict__.update(kwargs)

    def to_dict(self):
        properties = self.__dict__.copy()
        del(properties['dataSharedVarDict'])
        properties['global_update_settings'] = properties['global_update_settings'].to_dict()
        return properties

class Trainer(object):
    def __init__(self, model, trainerSettings = TrainerSettings()):
        self.__dict__.update(trainerSettings.__dict__)
        self.layer_updates = dict()
        self._set_model(model)
        self.train_func = None

    def bindUpdate(self, layerlist, update_settings):
        self.train_func = None
        if type(layerlist) != list:
            layerlist = [layerlist]
        for layer in layerlist:
            self.layer_updates[layer.name] = update_settings

    def _set_model(self,model):
        self.train_func = None
        self.model = model

    def init_ins_variables(self):
        inputs = self.model.inputs
        ins    = OrderedDict()
        for input_key,input_layers in inputs.iteritems():
            for input_layer in input_layers:
                ins[input_key] = input_layer.input_var
        return ins

    def get_outputs(self):
        all_layers_dict = self.model.layers
        outputs         = self.model.outputs
        all_layers      = all_layers_dict.values()
        all_outs        = layers.get_output(all_layers, deterministic=False)
        all_outs_dict   = dict(zip(all_layers_dict.keys(),all_outs))
        outsTrain       = [all_outs_dict[outputlayer] for outputlayer in outputs.keys()]
        return all_outs_dict,outsTrain

    def get_cost(self,layer_name,layer_dict,all_outs_dict,ins):
        preds = all_outs_dict[layer_name]
        target_type = layer_dict['target_type']
        if target_type == 'label':
            targs = T.matrix('targets')
            ins[layer_dict['target']] = targs
        elif target_type == 'recon':
            targs = all_outs_dict[layer_dict['target']]
        else:
            raise Exception('This should have been caught earlier')
        cost = layer_dict['loss_function'](preds,targs)
        aggregation_type = layer_dict['aggregation_type']
        if aggregation_type == 'mean':
            cost = cost.mean()
        elif aggregation_type == 'sum':
            cost = cost.sum()
        elif aggregation_type == 'weighted_mean':
            weights = T.matrix('weights')
            ins[layer_dict['weight_key']] = weights
            cost = T.sum(cost*weights)/T.sum(weights)
        elif aggregation_type == 'weighted_sum':
            weights = T.matrix('weights')
            ins[layer_dict['weight_key']] = weights
            cost = T.sum(cost*weights)
        else:
            raise Exception('This should have been caught earlier')
        return cost,ins

    def get_update(self,layer_name,ins,costTotal):
        all_layers = self.model.layers
        params = all_layers[layer_name].get_params()
        update = OrderedDict()
        if len(params)>0:
            lr = T.scalar('lr_%s'%layer_name)
            mom = T.scalar('mom_%s'%layer_name)
            ins['learning_rate_%s'%layer_name] = lr
            ins['momentum_%s'%layer_name] = mom
            if layer_name in self.layer_updates:
                update_function = self.layer_updates[layer_name].update
            else:
                update_function = self.global_update_settings.update
            update = update_function(costTotal, params, lr, mom)
        return update, ins

    def _create_train_func(self):
        if self.model is None:
            raise Exception("No model has been set to train!")
        inputs     = self.model.inputs
        outputs    = self.model.outputs
        all_layers = self.model.layers

        # Get costs
        ins = self.init_ins_variables()
        all_outs_dict,outsTrain = self.get_outputs()

        costs = []
        for layer_name, layer_dict in outputs.iteritems():
            # {layer_name:{output_layer,target,target_type,loss_function,aggregation_type}}
            cost,ins = self.get_cost(layer_name,layer_dict,all_outs_dict,ins)
            costs.append(cost)
        costTotal = T.sum(costs)
        # Get updates
        updates = OrderedDict()
        for layer_name in all_layers:
            update,ins = self.get_update(layer_name,ins,costTotal)
            updates.update(update)

        # Create functions
        givens = dict()
        if self.dataSharedVarDict is not None:
            batch_index = T.iscalar('Batch index')
            ins['batch_index'] = batch_index
            batch_slice = slice(batch_index * self.batch_size,(batch_index + 1) * self.batch_size)
            for input_key,input_layers in self.model.inputs.iteritems():
                for input_layer in input_layers:
                    inVar = ins.pop(input_key)
                    givens[inVar] = self.dataSharedVarDict[input_key][batch_slice]
            for output_layer_name,output_layer_dict in self.model.outputs.iteritems():
                if output_layer_dict['target_type'] == 'label':
                    targKey = output_layer_dict['target']
                    targVar = ins.pop(targKey)
                    givens[targVar]=self.dataSharedVarDict[targKey][batch_slice]
                weightKey = output_layer_dict['weight_key']
                if weightKey in ins.keys():
                    weightVar = ins.pop(weightKey)
                    givens[weightVar]=self.dataSharedVarDict[weightKey][batch_slice]

        train = theano.function(
            ins.values(),
            outsTrain,
            updates=updates,
            givens=givens,
            on_unused_input='warn'
        )

        self.train_func = train
        self.insKeys = ins.keys()
        return self.train_func

    def train_step(self,batch_dict):
        if self.train_func is None:
            self.train_func = self._create_train_func()
        ins = []
        param_names = ['learning_rate', 'momentum']
        for key in self.insKeys:
            for param in param_names:
                if param in key:
                    layer_name = key.split(param+'_')[1]
                    if layer_name in self.layer_updates:
                        inval = getattr(self.layer_updates[layer_name], param)
                    else:
                        inval = getattr(self.global_update_settings, param)
                    break
            else:
                inval = batch_dict[key]
            ins.append(inval)
        return self.train_func(*ins)

    def to_dict(self):
        # Set self.model to None in dict
        pass
