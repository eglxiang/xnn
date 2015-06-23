class Trainer(Object):
    def __init__(self):
        self.num_epochs = 1000
        self.batch_size = 128
        self.dataInRam  = True

        self.iter_functs = self._create_funcs()
    def train(self):
        pass



    def _create_funcs(self):






        # Get costs
        costs = self._bind_data_to_outputs(updates_dict)
        for update_key in updates_dict:
            # {name:[type, outputKey, outputVar, costVar]}
            pass
        # Get updates
        updates = []
        for _ in _:
            updates += lasagne.updates.nesterov_momentum(
                cost, params, learning_rate, momentum)

        # Create functions
        if self.dataInRam:
            batch_index = T.scalar('Batch index')
            batch_slice = slice(batch_index * self.batch_size,(batch_index + 1) * self.batch_size)
            ins.append(batch_index)
            train = theano.function(
                [ins], # batch_index
                [outsTrain],
                updates=updates,
                givens={
                    xBatch: data['X'][batch_slice],
                    yBatch: data['y'][batch_slice],
                    wBatch: data['w'][batch_slice]
                }
            )
            evaluate = theano.function(
                [ins], # batch_index
                [outsEval],
                givens={
                    xBatch: data['X'][batch_slice],
                    yBatch: data['y'][batch_slice],
                    wBatch: data['w'][batch_slice]
                }
            )
        else:
            train = theano.function(
                [ins], # xBatch,yBatch,wBatch
                [outsTrain],
                updates=updates
            )
            evaluate = theano.function(
                [ins], # xBatch,yBatch,wBatch
                [outsEval]
            )
