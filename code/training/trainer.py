class TrainerSettings(object):
    def __init__(self, batch_size=128,
            dataInGpu=True, **kwargs):
        self.batch_size = 128
        self.dataInGpu  = True
        # self.update_dict = {}
        self.__dict__.update(kwargs)
    def to_dict(self):
        return self.__dict__

class Trainer(object):
    def __init__(self, trainerSettings, model):
        self.__dict__.update(trainerSettings.__dict__)
        self._set_model(model)
        self.train_func = self._create_train_func()

    def _set_model(self,model):
        self.model = model
        # self.update_dict = {}
        # if model is not None:
            # for layer in self.model.layers:
                # self.update_dict[layer]={}

    # def bind_update_settings(self,layerName,settingsDict):
    #     if self.model is None:
    #         raise Exception("No model has been set to train!")
    #     self.update_dict[layerName].update(settingsDict)

    def _create_train_func(self):
        if self.model is None:
            raise Exception("No model has been set to train!")
        inputs = self.model.inputs
        outputs = self.model.outputs
        layers = self.model.layers
        # Get costs
        insTrain = []
        insKeys = []
        for input_key,input_layers in inputs.iteritems():
            for input_layer in input_layers:
                insTrain.append(input_layer.input_var)
                insKeys.append(input_key)
        costs = []
        output_layers = [output['layer'] for output in outputs]
        outsTrain = lasagne.layers.get_output(output_layers, deterministic=False)
        out_vars_train_dict = dict(zip(outputs.keys(),outsTrain))
        for layer_name, layer_dict in outputs.iteritems():
            # {layer_name:{output_layer,target,target_type,loss_function,aggregation_type}}
            preds = out_vars_train_dict[layer_name]
            if layer_dict['target_type'] == 'label':
                targs = T.matrix('targets')
                insTrain.append(targs)
                insKeys.append(layer_dict['target'])
            elif layer_dict['target_type'] == 'recon':
                targs = out_vars_train_dict[layer_dict['target']]
            else:
                raise Exception('This should have been caught earlier')
            cost = layer_dict['loss_function'](preds,targs)
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
            costs.append(cost)
        costTotal = T.sum(costs)
        # Get updates
        updates = []
        for layer in layers:
            lr = T.scalar('lr_%s'%layer)
            mom = T.scalar('mom_%s'%layer)
            insTrain.append(lr)
            insKeys.append('learning_rate_%s'%layer)
            insTrain.append(mom)
            insKeys.append('momentum_%s'%layer)
            params = layers[layer].get_params()
            updates += lasagne.updates.nesterov_momentum(
                costTotal, params, lr, mom)

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
                }
            )
        else:
            train = theano.function(
                insTrain, # xBatch,yBatch,wBatch
                outsTrain,
                updates=updates
            )
        self.train_func = train
        self.insKeys = insKeys
        return self.train_func

    def train_step(self,batch_dict):
        ins = []
        for key in self.insKeys:
            if 'learning_rate' in key:
                if not key in batch_dict:
                    if 'learning_rate_default' not in batch_dict:
                        raise Exception("Either specify learning_rate_default or learning_rate for each layer!")
                    inval = batch_dict['learning_rate_default']
                else:
                    inval = batch_dict[key]
            elif 'momentum' in key:
                if not key in batch_dict:
                    if 'momentum_default' not in batch_dict:
                        raise Exception("Either specify momentum_default or momentum for each layer!")
                    inval = batch_dict['momentum_default']
                else:
                    inval = batch_dict[key]
            else:
                inval = batch_dict[key]
            # ins.append(batch_dict[key])
            ins.append(inval)
        return self.train_func(ins)

    def to_dict():
        # Set self.model to None in dict
        pass
