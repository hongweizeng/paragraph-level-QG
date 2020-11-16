import os
import yaml
import time
from easydict import EasyDict
import json

DEFAULT_BEST_CHECKPOINT_NAME = 'best.ckpt'
DEFAULT_LATEST_CHECKPOINT_NAME = 'latest.ckpt'
DEFAULT_TRAIN_CACHED_DATASET_SUFFIX = '.train.pt'
DEFAULT_VALID_CACHED_DATASET_SUFFIX = '.valid.pt'
DEFAULT_VOCAB_CACHED_DATASET_SUFFIX = '.vocab.pt'
DEFAULT_CONFIG_NAME = 'config.json'


def preprocess_args(args):
    yaml_file = args.config
    with open(yaml_file, 'r') as stream:
        config_dict = yaml.safe_load(stream)

    config = EasyDict(config_dict)

    # Checkpoint Management.
    cached_model_dir = os.path.join(config['cached_model_dir'], config['setup'])
    config.save_path = os.path.join(
        cached_model_dir, "training_%d" % int(time.strftime("%Y%m%d%H%M%S")))
    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path)

    config.cached_best_model = os.path.join(config.save_path, DEFAULT_BEST_CHECKPOINT_NAME)
    config.cached_latest_model = os.path.join(config.save_path, DEFAULT_LATEST_CHECKPOINT_NAME)

    # Logging.
    config.log_file = os.path.join(config.save_path, 'train.log')

    # Output
    config.output =  os.path.join(config.save_path, config.output)

    config_object = {k: v for k, v in config.items()}
    with open(os.path.join(config.save_path, DEFAULT_CONFIG_NAME), 'w') as json_writer:
        json.dump(config_object, json_writer, indent=4)


    return config


# if __name__ == '__main__':
#     parse_config('configs/ssr.yml')