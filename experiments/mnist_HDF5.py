from experiments.mnist_base import build_model, create_iter_functions
from data.HDF5RamPool import *
from data.samplers import *
from data.weights import *
from training.train_loop_HDF5 import run
import argparse
import os.path
import json

NUM_EPOCHS = 1500
BATCH_SIZE = 128 
NUM_HIDDEN_UNITS = 512
LEARNING_RATE = 0.075
MOMENTUM = 0.9
MISSING_PCT = .9#.99#.5
RANDOM_SEED = 100
DEFAULT_NAME = os.path.splitext(os.path.basename(__file__))[0]

def main(config, stats_file=None):

    frN = HDF5FieldReader('number',preprocessFunc=makeFloatX)
    frA = HDF5FieldReader('angle')
    bl = HDF5BatchLoad('data/mnist.hdf5',[frN,frA,pixelReader])
    samplers = {}
    weights = {}
    for part in ['train','valid','test']:
        rp = HDF5RamPool(bl,part,refreshPoolProp=0)
        sm = 'sequential' if part != 'train' else 'uniform'
        samplers[part] = Sampler(rp,
                                [CategoricalSampler('number')],
                                samplemethod=sm,
                                batchsize=BATCH_SIZE,
                                nanOthers=False)
        weights[part] = CategoricalWeighter('number')

    output_layer = build_model(
        input_dim= 28*28, #  dataset['input_dim'],
        output_dim= 10, #dataset['output_dim'],
        batch_size=config['batch_size'],
        num_hidden_units=config['num_hidden_units']
    )

    iter_funcs = create_iter_functions(
        [],
        output_layer,
        batch_size=config['batch_size'],
        learning_rate=config['learning_rate'],
        momentum=config['momentum'],
        dataInVRAM=False
    )

    run(iter_funcs=iter_funcs,
        samplers=samplers,
        weights=weights,
        num_epochs=config['num_epochs'],
        batch_size=config['batch_size'], stats_file=stats_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Experiment runner for mnist_1",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-o", "--outputdir", type=str, required=False, help='Output directory')
    parser.add_argument("-n", "--runname", type=str, required=False, default=DEFAULT_NAME, help='Name for training run')
    parser.add_argument("-c", "--config", type=json.loads, required=False, help="JSON configuration string")
    args = parser.parse_args()

    config = dict(
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        num_hidden_units=NUM_HIDDEN_UNITS,
        learning_rate=LEARNING_RATE,
        momentum=MOMENTUM,
        missing_pct=MISSING_PCT,
        random_seed=RANDOM_SEED
    )

    if args.config is not None:
        for key in args.config:
            config[key] = args.config[key]

    if args.outputdir is not None:
        json.dump(config,
                  open(os.path.join(args.outputdir, args.runname+'_config.json'), 'w'),
                  indent=4, sort_keys=True)
    stats_filename = os.path.join(args.outputdir, args.runname+'_stats.csv') if args.outputdir is not None else None
    main(config=config, stats_file=stats_filename)
