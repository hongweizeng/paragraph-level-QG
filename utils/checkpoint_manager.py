import os
import torch


class Checkpoint(object):
    model_state_dict: None
    epoch_num: None
    metric_scores: None
    optimizer_state_dict: None


class CheckpointManager(object):

    DEFAULT_BEST_MODEL_NAME = 'best.ckpt'
    DEFAULT_LATEST_MODEL_NAME = 'latest.ckpt'

    DEFAULT_CONFIG_NAME = 'config.json'

    def __init__(self, directory):
        self.checkpoint_directory = directory
        self.config = os.path.join(directory, self.DEFAULT_CONFIG_NAME)

    @property
    def best_model_path(self):
        checkpoint_path = os.path.join(self.checkpoint_directory, self.DEFAULT_BEST_MODEL_NAME)
        return checkpoint_path

    @property
    def latest_model_name(self):
        checkpoint_path =  os.path.join(self.checkpoint_directory, self.DEFAULT_LATEST_MODEL_NAME)
        return checkpoint_path

    @property
    def best_checkpoint(self):
        checkpoint = torch.load(self.best_model_path)
        return checkpoint

    @property
    def latest_checkpoint(self):
        checkpoint = torch.load(self.best_model_path)
        return checkpoint