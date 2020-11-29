import os
import sys
import time
import functools
from tqdm import tqdm
from collections import deque
from easydict import EasyDict
import torch
import torch.nn as nn
from torchtext.vocab import Vocab as TorchTextVocab
from shutil import copyfile
import numpy as np

from utils import logger, count_params
from datasets.common import QgDataset, setup_iterator, PAD_TOKEN, PAD_INDEX_NAME
from datasets.collate_functions import master_collate_function

from train.optimizers import Optimizer
from train.criteria import Criterion
from train.checkpoint_manager import Checkpoint, CheckpointManager
from train.scorer_manager import ScorerManager
from train.statistics import Statistics

DEFAULT_BEST_CHECKPOINT_NAME = 'best.ckpt'
DEFAULT_LATEST_CHECKPOINT_NAME = 'latest.ckpt'


class Trainer(object):
    def __init__(self, train_dataset: QgDataset, valid_dataset: QgDataset,
                 vocabularies: TorchTextVocab, model:nn.Module,
                 config:EasyDict):
        self.config = config
        self.training_config = config['train']
        pad_token_id = self.config['model'].pad_token_id
        vocab_size = self.config['model'].vocab_size

        self.device = next(model.parameters()).device
        collate_fn = functools.partial(master_collate_function, pad_token_id=pad_token_id, device=self.device)
        self.train_iter = setup_iterator(dataset=train_dataset, collate_fn=collate_fn,
                                    batch_size=self.training_config['batch_size'], random=True)
        self.valid_iter = setup_iterator(dataset=valid_dataset, collate_fn=collate_fn,
                                    batch_size=128, random=False)

        self.vocabularies = vocabularies
        self.padding_index = vocabularies['token'].stoi[PAD_TOKEN]

        self.model = model
        logger.info('Setup %s optimizer with initialized learning rate = %.5f' %
                    (self.training_config['optimizer']['optimizer_name'],
                     self.training_config['optimizer']['learning_rate']))
        self.optimizer: Optimizer = Optimizer(model, self.training_config['optimizer'])

        logger.info('Setup criterion with cross-entropy, copy and coverage')
        self.criterion: Criterion = Criterion(self.training_config['criterion'],
                                              pad_token_id=pad_token_id, vocab_size=vocab_size, device=self.device)

        logger.info('Setup checkpoint manager')
        self.checkpoint_manager: CheckpointManager = CheckpointManager(self.training_config['checkpoint'],
                                                                       save_path=self.config['save_path'])

        logger.info('Setup scorer manager with %s criteria' % self.training_config['scorer']['criteria'])
        self.scorer_manager: ScorerManager = ScorerManager(self.training_config['scorer'])

        self.lr = self.optimizer._learning_rate

        # self.save_path = save_path
        # self.best_loss = 1e10
        # self.cached_best_model = os.path.join(save_path, DEFAULT_BEST_CHECKPOINT_NAME)
        # self.cached_latest_model = os.path.join(save_path, DEFAULT_LATEST_CHECKPOINT_NAME)
        # self.max_to_keep = config['max_to_keep']
        # if config['max_to_keep'] > 0:
        #     self.checkpoint_queue = deque([], maxlen=config['max_to_keep'])

        self.num_train_epochs = self.training_config['num_train_epochs']
        self.valid_steps = self.training_config['valid_steps']
        self.training_step = 0

    def train_from(self, train_from=None):
        if train_from is not None and os.path.exists(train_from):
            checkpoint: Checkpoint = torch.load(train_from)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint['epoch_num']
        else:
            start_epoch = 0
        return start_epoch

    def train(self, train_from=None):
        logger.info(' * Dataset size, train = %d, valid = %d' %
                    (len(self.train_iter.dataset), len(self.valid_iter.dataset)))
        logger.info(' * Vocabulary size, token = %d' % len(self.vocabularies['token']))
        logger.info(' * Number of params to train = %d' % count_params(self.model))
        logger.info(' * Number of epochs to train = %d' % self.num_train_epochs)

        # Restore from checkpoint
        start_epoch = self.train_from(train_from)

        for epoch in range(start_epoch, self.num_train_epochs):

            if self.scorer_manager.early_stop():
                break

            # Train
            with tqdm(desc='[Training (epoch = %d)]' % epoch, total=len(self.train_iter), ncols=150) as progress_bar:

                report_state = Statistics()
                for step, batch_data in enumerate(self.train_iter):
                    self.model.train()

                    # loss, batch_stat = self.run_batch(batch_data)
                    model_output = self.model(batch_data)
                    # loss, batch_stat = self.criterion(batch_data, model_output)
                    loss, batch_stat = self.criterion.compute_loss_v2(batch_data, model_output)

                    report_state.update(batch_stat)
                    report_state.report_training_step_with_tqdm(progress_bar)

                    self.training_step += 1

                    self.optimizer.zero_grad()
                    self.optimizer.backward(loss)
                    self.optimizer.step()

                    if self.training_step % self.valid_steps == 0:
                        valid_stat = self.validation()

                        # New line
                        print()
                        logger.info(
                            '[Validation (Epoch = %d, Step = %d)]: perplexity = %.3f, accuracy = %.2f, xent = %.5f,'
                            ' loss = %.5f, lr=%.5f' % (epoch, self.training_step,
                             valid_stat.ppl(), valid_stat.accuracy(), valid_stat.xent(), valid_stat.loss,
                             self.optimizer.get_learning_rate()))

                        # Scorer Management
                        self.scorer_manager(valid_stat, self.training_step)
                        if self.scorer_manager.early_stop():
                            # Early Stop. [sys.exit()]
                            logger.info("The training has been early stopped...")
                            break
                        else:
                            is_improving = self.scorer_manager.is_improving()

                            # Learning rate scheduler
                            self.optimizer.update_learning_rate(is_improving)

                            # Checkpoint Management
                            checkpoint = Checkpoint(
                                model_state_dict=self.model.state_dict(), optimizer_state_dict=self.optimizer.state_dict(),
                                epoch_num=epoch, training_step=self.training_step, scores=valid_stat.scores())
                            self.checkpoint_manager.save(checkpoint, is_improving)

            progress_bar.close()

    def validation(self):
        self.model.eval()
        report_state = Statistics()
        for step, batch_data in enumerate(self.valid_iter):
            with torch.no_grad():
                model_output = self.model(batch_data)
                # loss, batch_stat = self.criterion(batch_data, model_output)
                # loss, batch_stat = self.criterion.compute_loss(batch_data, model_output)
                loss, batch_stat = self.criterion.compute_loss_v2(batch_data, model_output)
            report_state.update(batch_stat)

        return report_state
