from experiments.mnist_base import build_model, create_iter_functions
from data.mnist_loader import load_data
from training.train_loop import run
import argparse
import os.path
import json

NUM_EPOCHS = 400
BATCH_SIZE = 600
NUM_HIDDEN_UNITS = 512
LEARNING_RATE = 0.1
MOMENTUM = 0.9
MISSING_PCT = .9#.99#.5
RANDOM_SEED = 100
DEFAULT_NAME = os.path.splitext(os.path.basename(__file__))[0]

def main(config, stats_file=None):
    dataset = load_data(missing_pct=config['missing_pct'],
                        random_seed=config['random_seed'])

    output_layer = build_model(
        input_dim=dataset['input_dim'],
        output_dim=dataset['output_dim'],
        batch_size=config['batch_size'],
        num_hidden_units=config['num_hidden_units']
    )

    iter_funcs = create_iter_functions(
        dataset,
        output_layer,
        batch_size=config['batch_size'],
        learning_rate=config['learning_rate'],
        momentum=config['momentum']
    )

    run(iter_funcs=iter_funcs,
        dataset=dataset,
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

    json.dump(config,
              open(os.path.join(args.outputdir, args.runname+'_config.json'), 'w'),
              indent=4, sort_keys=True)

    main(config=config, stats_file=os.path.join(args.outputdir, args.runname+'_stats.csv'))