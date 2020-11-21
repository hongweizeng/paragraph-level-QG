import torch
import torch.nn as nn

from utils.logging import logger
from utils.statistics import Statistics, CopyStatistics


class Criterions(object):
    def __init__(self, generator,
                 std_criterion, copy_criterion=None, coverage_criterion=None,
                 copy_weight=1.0, coverage_weight=0.4, paddding_idx=0):
        self.generator = generator

        self.std_criterion = std_criterion
        self.copy_criterion = copy_criterion
        self.coverage_criterion = coverage_criterion

        self.copy_weight = copy_weight
        self.coverage_weight = coverage_weight

        self.paddding_idx = paddding_idx

    def __call__(self,
                     deco_output, copy_prob, copy_dist,  # output
                     gold_output, copy_gate, copy_from,  # ground_truth
                     coverage_pred  # Coverage
                     ):
        deco_dist_log = self.generator(deco_output)

        if self.copy_criterion is not None:
            deco_dist_log = (deco_dist_log * (1 - copy_prob) + 1e-8) * (1 - copy_gate.unsqueeze(2))
            deco_loss = self.std_criterion(deco_dist_log.transpose(1, 2), gold_output)

            copy_dist_log = torch.log(copy_dist * copy_prob + 1e-8) * (copy_gate.unsqueeze(2))
            copy_loss = self.copy_criterion(copy_dist_log.transpose(1, 2), copy_from)

            total_loss = deco_loss + copy_loss
        else:
            deco_loss = self.std_criterion(deco_dist_log.transpose(1, 2), gold_output)

            total_loss = deco_loss

        if self.coverage_criterion is not None:
            coverage_pred = [cv for cv in coverage_pred]

            coverage_loss = torch.sum(torch.stack(coverage_pred, 1), 1)
            coverage_loss = torch.sum(coverage_loss, 0)
            total_loss += coverage_loss * self.coverage_weight

        non_pad_mask = gold_output.ne(self.paddding_idx)
        num_correct_words = deco_dist_log.max(-1)[1].eq(gold_output).masked_select(non_pad_mask).sum().item()
        num_words = non_pad_mask.sum().item()
        stats = Statistics(total_loss.item(), num_words, num_correct_words)

        # non_pad_mask = gold_output.ne(self.paddding_idx)
        # non_copy_mask = copy_gate.ne(1)
        # decoding_mask = non_pad_mask | non_copy_mask
        #
        # num_correct_words = deco_dist_log.max(-1)[1].eq(gold_output).masked_select(decoding_mask).sum().item()
        # num_words = decoding_mask.sum().item()

        # copy_mask = copy_gate.ne(0)
        # num_copy_correct_words = copy_dist.max(-1)[1].eq(copy_from).masked_select(copy_mask).sum().item()
        # num_copy_words = copy_mask.sum().item()

        # stats = CopyStatistics(total_loss.item(), num_words, num_correct_words,
        #                        n_copy_words=num_copy_words, n_copy_correct=num_copy_correct_words)

        return total_loss, stats


def setup_criterions(generator, configs, tgt_padding_idx, device):
    logger.info('Setup criterions...')
    std_criterion = nn.NLLLoss(reduction='sum', ignore_index=tgt_padding_idx).cuda(device)
    copy_criterion = nn.NLLLoss(reduction='sum').cuda(device) if configs.copy else None
    coverage_criterion = True if configs.coverage else None

    criterions = Criterions(generator, std_criterion, copy_criterion, coverage_criterion,
                            configs.copy_weight, configs.coverage_weight, paddding_idx=tgt_padding_idx)
    return criterions