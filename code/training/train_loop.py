import numpy as np
import time
import itertools

BATCH_SIZE = 600
NUM_EPOCHS = 500

def train(iter_funcs, dataset, batch_size=BATCH_SIZE):
    """Train the model with `dataset` with mini-batch training. Each
       mini-batch has `batch_size` recordings.
    """
    num_batches_train = dataset['num_examples_train'] // batch_size
    num_batches_valid = dataset['num_examples_valid'] // batch_size

    for epoch in itertools.count(1):
        batch_train_losses = []
        for b in range(num_batches_train):
            batch_train_loss = iter_funcs['train'](b)
            batch_train_losses.append(batch_train_loss)

        avg_train_loss = np.mean(batch_train_losses)

        batch_valid_losses = []
        batch_valid_accuracies = []
        for b in range(num_batches_valid):
            batch_valid_loss, batch_valid_accuracy = iter_funcs['valid'](b)
            batch_valid_losses.append(batch_valid_loss)
            batch_valid_accuracies.append(batch_valid_accuracy)

        avg_valid_loss = np.mean(batch_valid_losses)
        avg_valid_accuracy = np.mean(batch_valid_accuracies)

        yield {
            'number': epoch,
            'train_loss': avg_train_loss,
            'valid_loss': avg_valid_loss,
            'valid_accuracy': avg_valid_accuracy,
        }

def run(iter_funcs, dataset, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, stats_file=None):
    print("Starting training...")
    now = time.time()

    if stats_file is not None:
        fid = open(stats_file, 'w')
        fid.write('epoch,trainloss,validloss,validacc\n')
        fid.close()
    try:
        for epoch in train(iter_funcs, dataset, batch_size=batch_size):
            print("Epoch {} of {} took {:.3f}s".format(
                epoch['number'], num_epochs, time.time() - now))
            now = time.time()
            print("  training loss:\t\t{:.6f}".format(epoch['train_loss']))
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
