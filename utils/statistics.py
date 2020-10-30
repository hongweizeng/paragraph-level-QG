# https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/utils/statistics.py

from __future__ import division
import time
import math
import sys

from .logging import logger


class Statistics(object):
    """
    Accumulator for loss statistics.
    Currently calculates:
    * accuracy
    * perplexity
    * bleu-n
    * elapsed time
    """

    def __init__(self, loss=0, n_words=0, n_correct=0):
        self.loss = loss
        self.n_words = n_words
        self.n_correct = n_correct
        self.n_src_words = 0
        self.start_time = time.time()

    def update(self, stat, update_n_src_words=False):
        """
        Update statistics by suming values with another `Statistics` object
        Args:
            stat: another statistic object
            update_n_src_words(bool): whether to update (sum) `n_src_words`
                or not
        """
        self.loss += stat.loss
        self.n_words += stat.n_words
        self.n_correct += stat.n_correct

        if update_n_src_words:
            self.n_src_words += stat.n_src_words

    def accuracy(self):
        """ compute accuracy """
        return 100 * (self.n_correct / self.n_words)

    def xent(self):
        """ compute cross entropy """
        return self.loss / self.n_words

    def ppl(self):
        """ compute perplexity """
        return math.exp(min(self.loss / self.n_words, 100))

    def elapsed_time(self):
        """ compute elapsed time """
        return time.time() - self.start_time

    def output(self, step, num_steps, learning_rate, start):
        """Write out statistics to stdout.
        Args:
           step (int): current step
           n_batch (int): total batches
           start (int): start time of step.
        """
        t = self.elapsed_time()
        step_fmt = "%2d" % step
        if num_steps > 0:
            step_fmt = "%s/%5d" % (step_fmt, num_steps)
        logger.info(
            ("Step %s; acc: %6.2f; ppl: %5.2f; xent: %4.2f; " +
             "lr: %7.5f; %3.0f/%3.0f tok/s; %6.0f sec")
            % (step_fmt,
               self.accuracy(),
               self.ppl(),
               self.xent(),
               learning_rate,
               self.n_src_words / (t + 1e-5),
               self.n_words / (t + 1e-5),
               time.time() - start))
        sys.stdout.flush()

    def log_tensorboard(self, prefix, writer, learning_rate, step):
        """ display statistics to tensorboard """
        t = self.elapsed_time()
        writer.add_scalar(prefix + "/xent", self.xent(), step)
        writer.add_scalar(prefix + "/ppl", self.ppl(), step)
        writer.add_scalar(prefix + "/accuracy", self.accuracy(), step)
        writer.add_scalar(prefix + "/tgtper", self.n_words / t, step)
        writer.add_scalar(prefix + "/lr", learning_rate, step)

    def report_training_step_with_tqdm(self, tqdm_bar):
        logging_metrics = {
            'acc': '%6.2f' % self.accuracy(),
            'ppl': '%5.5f' % self.ppl(),
            'xent': '%4.5f' % self.xent()
        }
        tqdm_bar.set_postfix(**logging_metrics)
        tqdm_bar.update()


class CopyStatistics(Statistics):
    def __init__(self, loss=0, n_words=0, n_correct=0, n_copy_words=0, n_copy_correct=0):
        self.n_copy_words = n_copy_words
        self.n_copy_correct = n_copy_correct
        super(CopyStatistics, self).__init__(loss, n_words, n_correct)


    def copy_accuracy(self):
        """ compute accuracy """
        return 100 * (self.n_copy_correct / self.n_copy_words)

    def update(self, stat, update_n_src_words=False):
        """
        Update statistics by suming values with another `Statistics` object
        Args:
            stat: another statistic object
            update_n_src_words(bool): whether to update (sum) `n_src_words`
                or not
        """
        self.loss += stat.loss
        self.n_words += stat.n_words
        self.n_correct += stat.n_correct
        self.n_copy_words += stat.n_copy_words
        self.n_copy_correct += stat.n_copy_correct

    def report_training_step_with_tqdm(self, tqdm_bar):
        logging_metrics = {
            'acc': '%6.2f' % self.accuracy(),
            'ppl': '%5.2f' % self.ppl(),
            'xent': '%4.2f' % self.xent(),
            'c_acc': '%6.2f' % self.copy_accuracy(),
        }
        tqdm_bar.set_postfix(**logging_metrics)
        tqdm_bar.update()
