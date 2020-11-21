import os
import time
from tqdm import tqdm
from collections import deque
from easydict import EasyDict
import torch
import torch.nn as nn
from torchtext.vocab import Vocab
from shutil import copyfile
import numpy as np

from utils import logger, count_params, Statistics
from datasets.common import PAD_TOKEN


DEFAULT_BEST_CHECKPOINT_NAME = 'best.ckpt'
DEFAULT_LATEST_CHECKPOINT_NAME = 'latest.ckpt'

def time_since(t):
    """ Function for time. """
    return time.time() - t


def user_friendly_time(s):
    """ Display a user friendly time from number of second. """
    s = int(s)
    if s < 60:
        return "{}s".format(s)

    m = s // 60
    s = s % 60
    if m < 60:
        return "{}m {}s".format(m, s)

    h = m // 60
    m = m % 60
    if h < 24:
        return "{}h {}m {}s".format(h, m, s)

    d = h // 24
    h = h % 24
    return "{}d {}h {}m {}s".format(d, h, m, s)


def eta(start, completed, total):
    """ Function returning an ETA. """
    # Computation
    took = time_since(start)
    time_per_step = took / completed
    remaining_steps = total - completed
    remaining_time = time_per_step * remaining_steps

    return user_friendly_time(remaining_time)


def progress_bar(completed, total, step=5):
    """ Function returning a string progress bar. """
    percent = int((completed / total) * 100)
    bar = '[='
    arrow_reached = False
    for t in range(step, 101, step):
        if arrow_reached:
            bar += ' '
        else:
            if percent // t != 0:
                bar += '='
            else:
                bar = bar[:-1]
                bar += '>'
                arrow_reached = True
    if percent == 100:
        bar = bar[:-1]
        bar += '='
    bar += ']'
    return bar


class Trainer(object):
    def __init__(self, vocabularies: Vocab, model:nn.Module,
                 optimizer: torch.optim, lr_scheduler: torch.optim.lr_scheduler, criterion:nn.Module,
                 save_path:str, config:EasyDict):
        self.config = config

        self.vocabularies = vocabularies
        self.padding_index = vocabularies['token'].stoi[PAD_TOKEN]

        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.criterion = criterion

        self.lr = config['sgd_learning_rate']

        self.save_path = save_path
        self.best_loss = 1e10
        self.cached_best_model = os.path.join(save_path, DEFAULT_BEST_CHECKPOINT_NAME)
        self.cached_latest_model = os.path.join(save_path, DEFAULT_LATEST_CHECKPOINT_NAME)
        self.max_to_keep = config['max_to_keep']
        if config['max_to_keep'] > 0:
            self.checkpoint_queue = deque([], maxlen=config['max_to_keep'])

    def train_from(self, train_from=None):
        if train_from is not None and os.path.exists(train_from):
            checkpoint = torch.load(train_from)
            # self.model.load_state_dict(checkpoint['model'])
            # start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint)
            start_epoch = 10
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

            # Learning rate scheduler
            # halving the learning rate after epoch 8
            self.adjust_learning_rate(epoch)
            # if epoch >= 8 and epoch % 2 == 0:
            #     self.lr_scheduler.step()

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
                self.model.zero_grad()
                loss.backward()     # TODO, div batch size
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config['max_grad_norm'])    # gradient clipping
                self.optimizer.step()

                report_state.report_training_step_with_tqdm(progress_bar, batch_loss=loss.item())

        if train:
            progress_bar.close()

        logger.info('Epoch %d, %s perplexity = %.3f, accuracy = %.2f, xent = %.5f, loss = %.5f' %
                    (epoch_num, TAGS, report_state.ppl(), report_state.accuracy(),
                     report_state.xent(), report_state.loss / len(data_iter)))
        # logger.info('Epoch %d, %s perplexity = %.3f, accuracy = %.2f, loss = %.5f' %
        #             (epoch_num, TAGS, report_state.ppl(), report_state.accuracy(), report_state.loss / len(data_iter)))

        return report_state

    def run_batch(self, batch_data):
        eos_trg = batch_data.question_ids[:, 1:]
        # if self.config['model']['use_pointer']:
        if self.config['model'].use_pointer:
            eos_trg = batch_data.question_extended_ids_para[:, 1:]  #TODO, paragraph or evidences, that is a question.

        model_output = self.model(batch_data)

        # Cross Entropy Loss
        logits = model_output['logits']
        batch_size, nsteps, _ = logits.size()
        preds = logits.view(batch_size * nsteps, -1)
        targets = eos_trg.contiguous().view(-1)
        loss = self.criterion(preds, targets)

        non_pad_mask = targets.ne(self.padding_index)
        num_correct_words = preds.max(-1)[1].eq(targets).masked_select(non_pad_mask).sum().item()
        num_words = non_pad_mask.sum().item()

        batch_state = Statistics(loss.item(), num_words, num_correct_words)

        # Coverage Loss
        # avg_tmp_coverage = coverage_sum / (i + 1)
        # coverage_loss = torch.sum(torch.min(energy, avg_tmp_coverage), dim=1)
        attention_scores = model_output['attentions']
        coverage_scores = model_output['coverages']

        # coverage_loss = sum([torch.sum(torch.min(cov_score / (decoding_step + 1), attn_score))

        coverage_loss = 0
        for decoding_step, (cov_score, attn_score) in enumerate(zip(coverage_scores, attention_scores)):
            step_coverage_loss = torch.min(cov_score / (decoding_step + 1), attn_score).sum()
            coverage_loss += step_coverage_loss

        loss = loss + coverage_loss * 0.0
        return loss, batch_state

    def report(self, stat):
        pass

    def adjust_learning_rate(self, epoch):
        # halving the learning rate after epoch 8
        lr = self.optimizer.state_dict()["param_groups"][0]['lr']
        # if epoch >= 2 and epoch % 2 == 0:     # TODO, this is for adam optimizer with learning_rate = 0.001
        if epoch >= 8 and epoch % 2 == 0:       # TODO, this is for sgd optimizer with learning_rate = 0.1
            lr *= 0.5
            state_dict = self.optimizer.state_dict()
            for param_group in state_dict["param_groups"]:
                param_group["lr"] = lr
            self.optimizer.load_state_dict(state_dict)

    def save_model(self, epoch, valid_stat):
        checkpoint_path = os.path.join(self.save_path, 'epoch_%d_acc_%.2f.ckpt' % (epoch, valid_stat.accuracy()))
        torch.save(self.model.state_dict(), checkpoint_path)
        copyfile(checkpoint_path, self.cached_latest_model)
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

    def customized_train(self, train_iter, valid_iter, num_train_epochs):
        logger.info(' * Dataset size, train = %d, valid = %d' % (len(train_iter.dataset), len(valid_iter.dataset)))
        logger.info(' * Vocabulary size, token = %d' % len(self.vocabularies['token']))
        logger.info(' * Number of params to train = %d' % count_params(self.model))
        logger.info(' * Number of epochs to train = %d' % num_train_epochs)

        batch_num = len(train_iter)
        best_loss = 1e10
        for epoch in range(1, num_train_epochs + 1):
            self.model.train()
            print("epoch {}/{} :".format(epoch, num_train_epochs), end="\r")
            start = time.time()
            # halving the learning rate after epoch 8
            if epoch >= 8 and epoch % 2 == 0:
                self.lr *= 0.5
                state_dict = self.optimizer.state_dict()
                for param_group in state_dict["param_groups"]:
                    param_group["lr"] = self.lr
                self.optimizer.load_state_dict(state_dict)

            for batch_idx, train_data in enumerate(train_iter, start=1):
                batch_loss = self.customized_step(train_data)

                self.model.zero_grad()
                batch_loss.backward()
                # gradient clipping
                nn.utils.clip_grad_norm_(self.model.parameters(),
                                         self.config['max_grad_norm'])

                self.optimizer.step()
                batch_loss = batch_loss.detach().item()
                msg = "{}/{} {} - ETA : {} - loss : {:.4f}" \
                    .format(batch_idx, batch_num, progress_bar(batch_idx, batch_num),
                            eta(start, batch_idx, batch_num), batch_loss)
                print(msg, end="\r")

            val_loss = self.customized_evaluate(valid_iter, msg)
            if val_loss <= best_loss:
                best_loss = val_loss
                self.customized_save_model(val_loss, epoch)

            print("Epoch {} took {} - final loss : {:.4f} - val loss :{:.4f}"
                  .format(epoch, user_friendly_time(time_since(start)), batch_loss, val_loss))

    def customized_evaluate(self, dev_loader, msg):
        self.model.eval()
        num_val_batches = len(dev_loader)
        val_losses = []
        for i, val_data in enumerate(dev_loader, start=1):
            with torch.no_grad():
                val_batch_loss = self.customized_step(val_data)
                val_losses.append(val_batch_loss.item())
                msg2 = "{} => Evaluating :{}/{}".format(
                    msg, i, num_val_batches)
                print(msg2, end="\r")

        val_loss = np.mean(val_losses)

        return val_loss

    def customized_step(self, batch_data):
        eos_trg = batch_data.question_ids[:, 1:]
        if self.config['use_pointer']:
            eos_trg = batch_data.question_extended_ids_para[:, 1:]  #TODO, paragraph or evidences, that is a question.

        logits = self.model(batch_data, src_padding_idx=self.padding_index)

        batch_size, nsteps, _ = logits.size()
        preds = logits.view(batch_size * nsteps, -1)
        targets = eos_trg.contiguous().view(-1)
        loss = self.criterion(preds, targets)

        return loss

    def customized_save_model(self, loss, epoch):
        state_dict = self.model.state_dict()
        loss = round(loss, 2)
        model_save_path = os.path.join(
            self.save_path, str(epoch) + "_" + str(loss))
        torch.save(state_dict, model_save_path)

def remove_checkpoint(name):
    if os.path.exists(name):
        os.remove(name)