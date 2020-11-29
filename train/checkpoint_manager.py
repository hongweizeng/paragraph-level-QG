# https://github.com/tensorflow/tensorflow/blob/v2.3.0/tensorflow/python/training/checkpoint_management.py
import os
from copy import deepcopy
from collections import deque
import torch
from utils.logging import logger

def remove_checkpoint(name):
    if os.path.exists(name):
        os.remove(name)


def copy_checkpoint(name):
    pass


class Checkpoint(object):
    def __init__(self, model_state_dict, optimizer_state_dict, epoch_num, training_step, scores, config=None):
        self.model_state_dict = model_state_dict
        self.optimizer_state_dict = optimizer_state_dict
        self.epoch_num = epoch_num
        self.training_step = training_step
        self.scores = scores
        self.config =config


class CheckpointManager(object):

    DEFAULT_BEST_MODEL_NAME = 'best.ckpt'
    DEFAULT_LATEST_MODEL_NAME = 'latest.ckpt'

    DEFAULT_CONFIG_NAME = 'config.json'

    def __init__(self, config, save_path):
        self.save_path = save_path
        # self.config_path = os.path.join(save_path, self.DEFAULT_CONFIG_NAME)

        self.max_to_keep = config['max_to_keep']
        if config.max_to_keep > 0:
            self.checkpoint_queue = deque([], maxlen=self.max_to_keep)

    @property
    def best_checkpoint_path(self):
        checkpoint_path = os.path.join(self.save_path, self.DEFAULT_BEST_MODEL_NAME)
        return checkpoint_path

    @property
    def latest_checkpoint_path(self):
        checkpoint_path =  os.path.join(self.save_path, self.DEFAULT_LATEST_MODEL_NAME)
        return checkpoint_path

    def load_best_checkpoint(self):
        checkpoint = torch.load(self.best_checkpoint_path)
        return checkpoint

    def load_latest_checkpoint(self):
        checkpoint = torch.load(self.latest_checkpoint_path)
        return checkpoint

    def save(self, checkpoint, is_improving):
        checkpoint_path = os.path.join(self.save_path,
                                       'step_%d_acc_%.3f.ckpt' % (checkpoint.training_step, checkpoint.scores['acc']))
        logger.info("Saving checkpoint to %s" % checkpoint_path)
        torch.save(checkpoint, checkpoint_path)

        torch.save(checkpoint, self.latest_checkpoint_path)
        if is_improving:
            torch.save(checkpoint, self.best_checkpoint_path)

        if self.max_to_keep > 0:
            if len(self.checkpoint_queue) == self.checkpoint_queue.maxlen:
                to_delete = self.checkpoint_queue.popleft()
                remove_checkpoint(to_delete)
            self.checkpoint_queue.append(checkpoint_path)