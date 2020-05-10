import argparse
import logging

from train.train_seq import train

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--config', action='store', dest='config',
    #                     help='Path to train yaml config file')
    # parser.add_argument('--save', action='store', dest='save',
    #                     help='Save dir')
    # opt = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    config_path = 'Configuration/config.yaml'
    train(config_path, './Experiments', 'Experiment01')
