import lasagne
import theano
import theano.tensor as T
from collections import OrderedDict

class ParamUpdateSettings():
    def __init__(self,
                 update=lasagne.updates.nesterov_momentum,
                 learning_rate=0.01,
                 momentum=0.9,
                 wc_function=lasagne.regularization.l2,
                 wc_strength=0.0001):
        self.update = update
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.wc_function = wc_function
        self.wc_strength = wc_strength

    def to_dict(self):
        properties = self.__dict__.copy()
        properties['update'] = properties['update'].func_name
        properties['wc_function'] = properties['wc_function'].func_name
        return properties

class TrainerSettings(object):
    def __init__(self,
                 global_update_settings=ParamUpdateSettings(),
                 batch_size=128,
                 dataInGpu=False, **kwargs):
        self.global_update_settings = global_update_settings
        self.batch_size = batch_size
        self.dataInGpu  = dataInGpu
        # self.update_dict = {}
        self.__dict__.update(kwargs)

    def to_dict(self):
        properties = self.__dict__.copy()
        properties['global_update_settings'] = properties['global_update_settings'].to_dict()
        return properties

class Trainer(object):
    def __init__(self, trainerSettings, model):
        self.__dict__.update(trainerSettings.__dict__)
        self.layer_updates = dict()
        self._set_model(model)
        self.train_func = self._create_train_func()

    def bindUpdate(self, layer, update_settings):
        self.layer_updates[layer.name] = update_settings

    def _set_model(self,model):
        self.model = model

    def init_ins_variables(self,inputs):
        insTrain = []
        insKeys = []
        for input_key,input_layers in inputs.iteritems():
            for input_layer in input_layers:
                insTrain.append(input_layer.input_var)
                insKeys.append(input_key)
        return insTrain,insKeys

    def get_outputs(self,layers,outputs):
        all_layers = layers.values()
        all_outs = lasagne.layers.get_output(all_layers, deterministic=False)
        all_outs_dict = dict(zip(layers.keys(),all_outs))
        outsTrain = [all_outs_dict[outputlayer] for outputlayer in outputs.keys()]
        return all_layers,all_outs,all_outs_dict,outsTrain

    def get_cost(self,layer_name,layer_dict,all_outs_dict,insTrain,insKeys):
        preds = all_outs_dict[layer_name]
        if layer_dict['target_type'] == 'label':
            targs = T.matrix('targets')
            insTrain.append(targs)
            insKeys.append(layer_dict['target'])
        elif layer_dict['target_type'] == 'recon':
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
            insTrain.append(weights)
            insKeys.append('weights_%s'%layer)
            cost = T.sum(cost*weights)/T.sum(weights)
        elif aggregation_type == 'weighted_sum':
            weights = T.matrix('weights')
            insTrain.append(weights)
            insKeys.append('weights_%s'%layer)
            cost = T.sum(cost*weights)
        else:
            raise Exception('This should have been caught earlier')
        return cost,insTrain,insKeys

    def get_update(self,layers, layername,insTrain,insKeys,costTotal):
        params = layers[layername].get_params()
        if len(params)>0:
            lr = T.scalar('lr_%s'%layername)
            mom = T.scalar('mom_%s'%layername)
            insTrain.append(lr)
            insKeys.append('learning_rate_%s'%layername)
            insTrain.append(mom)
            insKeys.append('momentum_%s'%layername)
            if layername in self.layer_updates:
                update_function = self.layer_updates[layername].update
            else:
                update_function = self.global_update_settings.update
            update = update_function(costTotal, params, lr, mom)
        return update, insTrain,insKeys

    def _create_train_func(self):
        if self.model is None:
            raise Exception("No model has been set to train!")
        inputs = self.model.inputs
        outputs = self.model.outputs
        layers = self.model.layers

        # Get costs
        insTrain,insKeys = self.init_ins_variables(inputs)
        all_layers,all_outs,all_outs_dict,outsTrain = self.get_outputs(layers,outputs)
        
        costs = []
        for layer_name, layer_dict in outputs.iteritems():
            # {layer_name:{output_layer,target,target_type,loss_function,aggregation_type}}
            cost,insTrain,insKeys = self.get_cost(layer_name,layer_dict,all_outs_dict,insTrain,insKeys)
            costs.append(cost)
        costTotal = T.sum(costs)
        # Get updates
        updates = OrderedDict()
        for layername in layers:
            update,insTrain,insKeys = self.get_update(layers,layername,insTrain,insKeys,costTotal)
            updates.update(update)

        # Create functions
        if self.dataInGpu: # TODO: fix!
            batch_index = T.scalar('Batch index')
            batch_slice = slice(batch_index * self.batch_size,(batch_index + 1) * self.batch_size)
            ins.append(batch_index)
            train = theano.function(
                insTrain, # batch_index
                outsTrain,
                updates=updates,
                givens={
                    xBatch: data['X'][batch_slice],
                    yBatch: data['y'][batch_slice],
                    wBatch: data['w'][batch_slice]
                },
                on_unused_input='warn'
            )
        else:
            train = theano.function(
                insTrain, # xBatch,yBatch,wBatch
                outsTrain,
                updates=updates,
                on_unused_input='warn'
            )
        self.train_func = train
        self.insKeys = insKeys
        return self.train_func

    def train_step(self,batch_dict):
        ins = []
        param_names = ['learning_rate', 'momentum']
        for key in self.insKeys:
            for param in param_names:
                if param in key:
                    layername = key.split(param+'_')[1]
                    if layername in self.layer_updates:
                        inval = getattr(self.layer_updates[layername], param)
                    else:
                        inval = getattr(self.global_update_settings, param)
                    break
            else:
                inval = batch_dict[key]
            # ins.append(batch_dict[key])
            ins.append(inval)
        return self.train_func(*ins)

    def to_dict():
        # Set self.model to None in dict
        pass

def train_test():
    from model.Model import Model
    import numpy as np
    m = Model('test model')
    l_in = m.addLayer(lasagne.layers.InputLayer(shape=(10,3)), name="l_in")
    l_h1 = m.addLayer(lasagne.layers.DenseLayer(l_in, 100), name="l_h1")
    l_out = m.addLayer(lasagne.layers.DenseLayer(l_h1, 3), name="l_out")

    m.bindInput(l_in, "pixels")
    m.bindOutput(l_h1, lasagne.objectives.categorical_crossentropy, "emotions", "label", "mean")
    m.bindOutput(l_out, lasagne.objectives.squared_error, "l_in", "recon", "mean")

    global_update_settings = ParamUpdateSettings(learning_rate=0.1, momentum=0.5)

    trainer_settings = TrainerSettings(update_settings=global_update_settings)
    trainer = Trainer(trainer_settings,m)

    batch_dict = dict(
        # learning_rate_default=0.1,
        # momentum_default=0.5,
        pixels=np.random.rand(10,3).astype(theano.config.floatX),
        emotions=np.random.rand(10,100).astype(theano.config.floatX)
    )
    outs = trainer.train_step(batch_dict)
    
if __name__ == "__main__":
    train_test()
