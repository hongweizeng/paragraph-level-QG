import math
import torch
import torch.nn as nn
from torch_scatter import scatter_max, scatter_sum, scatter

from utils.logging import logger
from train.statistics import Statistics


INF = 1e12


class Criterion(object):
    # def __init__(self, generator,
    #              std_criterion, copy_criterion=None, coverage_criterion=None,
    #              copy_weight=1.0, coverage_weight=0.4, paddding_idx=0):
    def __init__(self, config, pad_token_id, vocab_size, device):
        # self.generator = config.generator
        self.paddding_idx = pad_token_id
        self.vocab_size = vocab_size
        self.device = device

        self.vocab_criterion = nn.NLLLoss(ignore_index=pad_token_id, reduction=config.reduction).cuda(device)
        # self.xent_criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)

        self.use_pointer = config.copy
        if self.use_pointer:
            self.copy_criterion = nn.NLLLoss(reduction=config.reduction).cuda(device)
            self.copy_weight = config.copy_weight

        self.use_coverage = config.coverage
        if self.use_coverage:
            self.coverage_criterion = nn.NLLLoss(ignore_index=pad_token_id, reduction=config.reduction).cuda(device)
            self.coverage_weight = config.coverage_weight

            self.coverage_scatter = config.coverage_scatter


    def __call__(self, batch_data, model_output):
        eos_trg = batch_data.question_ids[:, 1:]
        if self.use_pointer:
            eos_trg = batch_data.question_extended_ids_para[:, 1:]  #TODO, paragraph or evidences, that is a question.

        # if self.config['model']['use_pointer']:
        batch_size = eos_trg.size(0)

        logits = model_output['logits']
        logits = torch.stack(logits, dim=1)  # [b, t, |V|]
        probs = torch.softmax(logits, dim=2)  # [b, t, |V|]

        coverage_loss = []
        if self.use_pointer:

            # Logits accumulations
            logits = []
            probs = []

            ext_src_seq = batch_data.paragraph_extended_ids
            num_oov = max(torch.max(ext_src_seq - self.vocab_size + 1), 0)

            if self.coverage_scatter == 'none':
                coverage = ext_src_seq.data.new_zeros(size=(batch_size, num_oov), dtype=torch.float32)
            else:
                coverage = ext_src_seq.data.new_zeros(size=(batch_size, self.vocab_size + num_oov), dtype=torch.float32)

            for seq_idx, (logit, energy, copy_gate) in enumerate(
                    zip(model_output['logits'], model_output['energies'], model_output['copy_gates'])):
                # zeros = torch.zeros((batch_size, num_oov)).type(dtype=logit.dtype)      #TODO:
                zeros = logit.data.new_zeros(size=(batch_size, num_oov))

                g_prob = torch.softmax(logit, dim=-1) * (1. - copy_gate) + 1e-8
                c_prob = torch.softmax(energy, dim=-1) * copy_gate + 1e-8

                extended_prob = torch.cat([g_prob, zeros], dim=1)
                out = torch.zeros_like(extended_prob)# - INF
                # out = scatter_sum(c_prob, ext_src_seq, out=out)
                out, _ = scatter_max(c_prob, ext_src_seq, out=out)  # TODO: scatter_sum.
                # out = c_prob
                # out = out.masked_fill(out == -INF, 0)
                prob = extended_prob + out
                # prob = prob.masked_fill(prob == 0, -INF)

                probs.append(prob)

                if self.use_coverage:
                    if self.coverage_scatter == 'sum':
                        extended_attn = torch.zeros_like(extended_prob)  # - INF
                        attention = scatter_sum(c_prob, ext_src_seq, out=extended_attn)
                    elif self.coverage_scatter == 'max':
                        # attention, _ = scatter_max(c_prob, ext_src_seq, out=extended_attn)
                        attention = out
                    else:
                        attention = c_prob

                    # step_coverage = coverage / max(1, seq_idx)
                    step_coverage = coverage
                    step_coverage_loss = torch.sum(torch.min(attention, step_coverage), dim=1)
                    # min_cover, _ = torch.cat((attention.unsqueeze(2), coverage.unsqueeze(2) / max(1, seq_idx)), dim=-1).min(2)
                    # min_cover = min_cover.sum(1)
                    # assert (step_coverage_loss == min_cover).all(), 'why'

                    coverage_loss += [step_coverage_loss]
                    coverage = coverage + attention

            probs = torch.stack(probs, dim=1)   # [b, t, |V|]

        probs = probs.view(-1, probs.size(2))
        log_probs = torch.log(probs)
        targets = eos_trg.contiguous().view(-1)
        loss = self.vocab_criterion(log_probs, targets)

        non_pad_mask = targets.ne(self.paddding_idx)
        num_correct_words = probs.max(-1)[1].eq(targets).masked_select(non_pad_mask).sum().item()
        num_words = non_pad_mask.sum().item()

        if self.use_coverage:
            # loss = loss + self.coverage_weight * torch.sum(torch.stack(coverage_loss, dim=-1).view(-1) * (1. - non_pad_mask.float()))
            loss = loss + self.coverage_weight * torch.sum(torch.stack(coverage_loss, dim=-1).view(-1) * non_pad_mask.float())

        loss_item = loss.item()
        if self.vocab_criterion.reduction == 'mean':
            loss_item = loss.item() * num_words
        batch_state = Statistics(loss_item, num_words, num_correct_words)

        if math.isnan(loss_item) or loss_item > 1e20:
            logger.info('catch NaN')

        return loss, batch_state


def setup_criterions(generator, configs, tgt_padding_idx, device):
    logger.info('Setup criterions...')
    std_criterion = nn.NLLLoss(reduction='sum', ignore_index=tgt_padding_idx).cuda(device)
    copy_criterion = nn.NLLLoss(reduction='sum').cuda(device) if configs.copy else None
    coverage_criterion = True if configs.coverage else None

    # criterions = Criterion(generator, std_criterion, copy_criterion, coverage_criterion,
    #                         configs.copy_weight, configs.coverage_weight, paddding_idx=tgt_padding_idx)
    # return criterions
