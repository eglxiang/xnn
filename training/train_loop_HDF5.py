import numpy as np
import time
import itertools

def train(iter_funcs, samplers, weights, batch_size=128):
    
    for epoch in itertools.count(1):
        batch_train_losses = []
        batch_train_accuracies = []
        start = time.time()
        for b in samplers['train']():
            wgt = weights['train'](b)
            #wgt /= wgt.sum()
            [batch_train_loss,batch_train_accuracy] = iter_funcs['train'](b['pixels'],b['number'],wgt)
            batch_train_losses.append(batch_train_loss)
            batch_train_accuracies.append(batch_train_accuracy)
        now = time.time()
        print "%d batches trained in %0.5f seconds"%(samplers['train'].numbatches,now-start)

        avg_train_loss = np.mean(batch_train_losses)
        avg_train_accuracy = np.mean(batch_train_accuracies)

        batch_valid_losses = []
        batch_valid_accuracies = []
        start = time.time()
        for b in samplers['valid']():
            batch_valid_loss, batch_valid_accuracy = iter_funcs['valid'](b['pixels'],b['number']) 
            batch_valid_losses.append(batch_valid_loss)
            batch_valid_accuracies.append(batch_valid_accuracy)
        now = time.time()
        print "%d batches validated in %0.5f seconds"%(samplers['valid'].numbatches,now-start)

        avg_valid_loss = np.mean(batch_valid_losses)
        avg_valid_accuracy = np.mean(batch_valid_accuracies)

        yield {
            'number': epoch,
            'train_loss': avg_train_loss,
            'train_accuracy': avg_train_accuracy,
            'valid_loss': avg_valid_loss,
            'valid_accuracy': avg_valid_accuracy,
        }

def run(iter_funcs, samplers, weights, num_epochs=1000, batch_size=128, stats_file=None):
    print("Starting training...")
    now = time.time()

    if stats_file is not None:
        fid = open(stats_file, 'w')
        fid.write('epoch,trainloss,validloss,validacc\n')
        fid.close()
    try:
        for epoch in train(iter_funcs, samplers,weights, batch_size=batch_size):
            print("Epoch {} of {} took {:.3f}s".format(
                epoch['number'], num_epochs, time.time() - now))
            now = time.time()
            print("  training loss:\t\t{:.6f}".format(epoch['train_loss']))
            print("  training accuracy:\t\t{:.2f} %%".format(
                epoch['train_accuracy'] * 100))
            print("  validation loss:\t\t{:.6f}".format(epoch['valid_loss']))
            print("  validation accuracy:\t\t{:.2f} %%".format(
                epoch['valid_accuracy'] * 100))

            if stats_file is not None:
                fid = open(stats_file, 'a')
                fid.write('%d,' % epoch['number'])
                fid.write('%.4f,' % epoch['train_loss'])
                fid.write('%.4f,' % epoch['valid_loss'])
                fid.write('%.4f' % epoch['valid_accuracy'])
                fid.write('\n')
                fid.close()

            if epoch['number'] >= num_epochs:
                break

    except KeyboardInterrupt:
        pass
