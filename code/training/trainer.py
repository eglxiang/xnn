class TrainerSettings(object):
    def __init__(self, batch_size=128,
            dataInRam=True, **kwargs):
        self.batch_size = 128
        self.dataInRam  = True
        self.model = None
        self.bg = None
        self.update_dict = {}
        self.__dict__.update(kwargs)

class Trainer(object):
    def __init__(self, trainerSettings, model=None, batchGenerator=None):
        self.__dict__.update(trainerSettings.__dict__)
        self.set_model(model)
        self.set_batch_generator(batchGenerator)
        self.iter_funcs = self._create_funcs()

    def set_model(self,model):
        self.model = model
        self.update_dict = {}
        for layer in self.model.layers:
            self.update_dict[layer]={}

    def set_batch_generator(self,batchGenerator):
        self.bg = batchGenerator

    def bind_update_settings(self,layerName,settingsDict):
        if self.model is None:
            raise Exception("No model has been set to train!")
        self.update_dict[layerName].update(settingsDict)

    def _create_funcs(self):
        if self.model is None:
            raise Exception("No model has been set to train!")
        if self.bg is None:
            raise Exception("No batch generator has been set to train with!")
        outputs = self.model.outputs
        layers = self.model.layers
        # Get costs
        insTrain = []
        costs = []
        out_vars_train = lasagne.layers.get_output(outputs.keys(), deterministic=False)
        out_vars_train_dict = dict(zip(outputs.keys(),out_vars_train))
        out_vars_eval = outputs.keys(),lasagne.layers.get_output(outputs.keys(), deterministic=True)
        for layer_name in outputs.keys():
            # {layer_name:{output_layer,target,target_type,loss_function,aggregation_type}}
            layer_dict = outputs[layer_name]
            preds = out_vars_train_dict[layer_name]
            if layer_dict['target_type'] == 'label':
                targs = T.matrix('targets')
            elif layer_dict['target_type'] == 'recon':
                targs = out_vars_train_dict[layer_dict['target']]
            else:
                raise Exception('This should have been caught earlier')
            cost = layer_dict['loss_function'](preds,targs)
            if aggregation_type == 'mean':
                cost = cost.mean()
            elif aggregation_type == 'sum':
                cost = cost.sum()
            elif aggregation_type == 'normalized_sum':
                weights = T.matrix('weights')
                cost = cost.sum()/weights.sum()
            else:
                raise Exception('This should have been caught earlier')
            costs.append(cost)
        costTotal = T.sum(costs)
        # Get updates
        updates = []
        for layer in self.update_dict:
            lr = T.scalar('lr_%s'%layer)
            mom = T.scalar('mom_%s'%layer)
            params = layers[layer].get_params()
            updates += lasagne.updates.nesterov_momentum(
                costTotal, params, lr, mom)

        # Create functions
        if self.dataInRam:
            batch_index = T.scalar('Batch index')
            batch_slice = slice(batch_index * self.batch_size,(batch_index + 1) * self.batch_size)
            ins.append(batch_index)
            train = theano.function(
                [insTrain], # batch_index
                [outsTrain],
                updates=updates,
                givens={
                    xBatch: data['X'][batch_slice],
                    yBatch: data['y'][batch_slice],
                    wBatch: data['w'][batch_slice]
                }
            )
            evaluate = theano.function(
                [insEval], # batch_index
                [outsEval],
                givens={
                    xBatch: data['X'][batch_slice],
                    yBatch: data['y'][batch_slice],
                    wBatch: data['w'][batch_slice]
                }
            )
        else:
            train = theano.function(
                [insTrain], # xBatch,yBatch,wBatch
                [outsTrain],
                updates=updates
            )
            evaluate = theano.function(
                [insEval], # xBatch,yBatch,wBatch
                [outsEval]
            )
        self.iter_funcs = dict(train=train,evaluate=evaluate)
        return self.iter_funcs

    def train_step(self,*args):
        return self.iter_funcs['train'](*args)

    def to_dict():
        # Set self.model and self.bg to None in dict
        pass
