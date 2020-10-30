import torch
import torch.nn as nn
import os
from collections import deque
from tqdm import tqdm
from easydict import EasyDict
from torchtext.vocab import Vocab

from utils import logger, count_params, Statistics
from data import PAD_TOKEN


DEFAULT_BEST_CHECKPOINT_NAME = 'best.ckpt'


class Trainer(object):
    def __init__(self, vocabularies: Vocab, model:nn.Module, optimizer: torch.optim, criterion:nn.Module,
                 save_path:str, config:EasyDict):
        self.config = config

        self.vocabularies = vocabularies
        self.padding_index = vocabularies['token'].stoi[PAD_TOKEN]

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

        self.save_path = save_path
        self.best_loss = 1e10
        self.cached_best_model = os.path.join(save_path, DEFAULT_BEST_CHECKPOINT_NAME)
        self.max_to_keep = config['max_to_keep']
        if config['max_to_keep'] > 0:
            self.checkpoint_queue = deque([], maxlen=config['max_to_keep'])

    def train_from(self, train_from=None):
        if train_from is not None and os.path.exists(train_from):
            checkpoint = torch.load(train_from)
            self.model.load_state_dict(checkpoint['model'])
            start_epoch = checkpoint['epoch']
        else:
            start_epoch = 0
        return start_epoch

    def train(self, train_iter, valid_iter, num_train_epochs, train_from=None):
        logger.info(' * Dataset size, train = %d, valid = %d' % (len(train_iter.dataset), len(valid_iter.dataset)))
        logger.info(' * Vocabulary size, token = %d' % len(self.vocabularies['token']))
        logger.info(' * Number of params to train = %d' % count_params(self.model))
        logger.info(' * Number of epochs to train = %d' % num_train_epochs)

        # Restore from checkpoint
        start_epoch = self.train_from(train_from)

        for epoch in range(start_epoch, num_train_epochs):
            self.model.train()

            # Train
            self.run_epoch(train_iter, epoch_num=epoch, train=True)

            # Validation
            self.model.eval()
            with torch.no_grad():
                valid_stat = self.run_epoch(valid_iter, epoch_num=epoch, train=False)

            # Save
            self.save_model(epoch, valid_stat)

    def run_epoch(self, data_iter, epoch_num, train):
        TAGS = 'Training' if train else 'Evaluation'
        if train:
            progress_bar = tqdm(desc='[%s (epoch = %d)] ' % (TAGS, epoch_num), unit='it', total=len(data_iter), ncols=150)

        report_state = Statistics()
        for step, batch_data in enumerate(data_iter):

            loss, batch_stat = self.run_batch(batch_data)
            # model_output = self.model(batch_data)
            #
            # loss, batch_stat = self.criterion(model_output, batch_data.decoder_output)
            report_state.update(batch_stat)

            if train:
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config['max_grad_norm'])    # gradient clipping
                self.optimizer.step()

                report_state.report_training_step_with_tqdm(progress_bar)

        if train:
            progress_bar.close()

        logger.info('Epoch %d, %s perplexity = %.3f, accuracy = %.2f' %
                    (epoch_num, TAGS, report_state.ppl(), report_state.accuracy()))

        return report_state

    def run_batch(self, batch_data):
        raise NotImplementedError

    def report(self, stat):
        pass

    def adjust_learning_rate(self, epoch):
        # halving the learning rate after epoch 8
        lr = self.optimizer.state_dict()["param_groups"][0]['lr']
        if epoch >= 8 and epoch % 2 == 0:
            lr *= 0.5
            state_dict = self.optimizer.state_dict()
            for param_group in state_dict["param_groups"]:
                param_group["lr"] = lr
            self.optimizer.load_state_dict(state_dict)

    def save_model(self, epoch, valid_stat):
        checkpoint_path = os.path.join(self.save_path, 'epoch_%d_acc_%.2f.ckpt' % (epoch, valid_stat.accuracy()))
        torch.save(self.model.state_dict(), checkpoint_path)
        if valid_stat.xent() < self.best_loss:
            self.best_loss = valid_stat.xent()
            torch.save(self.model.state_dict(), self.cached_best_model)

        if self.max_to_keep > 0:
            if len(self.checkpoint_queue) == self.checkpoint_queue.maxlen:
                to_delete = self.checkpoint_queue.popleft()
                remove_checkpoint(to_delete)
            self.checkpoint_queue.append(checkpoint_path)

    def early_stop(self):
        pass


def remove_checkpoint(name):
    if os.path.exists(name):
        os.remove(name)