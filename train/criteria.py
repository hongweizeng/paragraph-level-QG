import math
import torch
import torch.nn as nn
from torch_scatter import scatter_max, scatter_sum

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


    def __call__(self, batch_data, model_output):

        eos_trg = batch_data.question_ids[:, 1:]
        # if self.config['model']['use_pointer']:
        batch_size = eos_trg.size(0)

        logits = model_output['logits']

        if self.use_pointer:
            eos_trg = batch_data.question_extended_ids_para[:, 1:]  #TODO, paragraph or evidences, that is a question.

            # Logits accumulations
            logits = []
            ext_src_seq = batch_data.paragraph_extended_ids
            for logit, energy, copy_gate in zip(model_output['logits'], model_output['energies'], model_output['copy_gates']):
                num_oov = max(torch.max(ext_src_seq - self.vocab_size + 1), 0)
                # zeros = torch.zeros((batch_size, num_oov)).type(dtype=logit.dtype)      #TODO:
                zeros = logit.data.new_zeros(size=(batch_size, num_oov))
                extended_logit = torch.cat([logit, zeros], dim=1)
                out = torch.zeros_like(extended_logit) - INF
                out, _ = scatter_max(energy, ext_src_seq, out=out)  # TODO: scatter_sum.
                # out = scatter_mean(energy, ext_src_seq, out=out)      #TODO: scatter_sum.
                out = out.masked_fill(out == -INF, 0)
                logit = extended_logit + out
                logit = logit.masked_fill(logit == 0, -INF)


                logits.append(logit)
            logits = torch.stack(logits, dim=1)  # [b, t, |V|]


        # Cross Entropy Loss
        batch_size, nsteps, _ = logits.size()
        preds = logits.view(batch_size * nsteps, -1)
        targets = eos_trg.contiguous().view(-1)
        loss = self.xent_criterion(preds, targets)

        non_pad_mask = targets.ne(self.paddding_idx)
        num_correct_words = preds.max(-1)[1].eq(targets).masked_select(non_pad_mask).sum().item()
        num_words = non_pad_mask.sum().item()

        loss_item = loss.item()
        if self.xent_criterion.reduction == 'mean':
            loss_item = loss.item() * num_words
        batch_state = Statistics(loss_item, num_words, num_correct_words)

        # Coverage Loss
        # avg_tmp_coverage = coverage_sum / (i + 1)
        # coverage_loss = torch.sum(torch.min(energy, avg_tmp_coverage), dim=1)
        attention_scores = model_output['attentions']
        coverage_scores = model_output['coverages']

        # coverage_loss = 0
        # for decoding_step, (cov_score, attn_score) in enumerate(zip(coverage_scores, attention_scores)):
        #     step_coverage_loss = torch.min(cov_score / (decoding_step + 1), attn_score).sum()
        #     coverage_loss += step_coverage_loss
        # loss = loss + coverage_loss * 0.0
        return loss, batch_state


    def compute_loss_v2(self, batch_data, model_output):

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
            coverage = ext_src_seq.data.new_zeros(size=(batch_size, self.vocab_size + num_oov), dtype=torch.float32)

            for seq_idx, (logit, energy, copy_gate) in enumerate(
                    zip(model_output['logits'], model_output['energies'], model_output['copy_gates'])):
                # zeros = torch.zeros((batch_size, num_oov)).type(dtype=logit.dtype)      #TODO:
                zeros = logit.data.new_zeros(size=(batch_size, num_oov))

                g_prob = torch.softmax(logit, dim=-1) * (1. - copy_gate) + 1e-8
                c_prob = torch.softmax(energy, dim=-1) * copy_gate + 1e-8

                extended_prob = torch.cat([g_prob, zeros], dim=1)
                out = torch.zeros_like(extended_prob)# - INF
                out, _ = scatter_max(c_prob, ext_src_seq, out=out)  # TODO: scatter_sum.
                # out = out.masked_fill(out == -INF, 0)
                prob = extended_prob + out
                # prob = prob.masked_fill(prob == 0, -INF)

                probs.append(prob)

                if self.use_coverage:
                    extended_attn = torch.zeros_like(extended_prob)  # - INF
                    attention = scatter_sum(c_prob, ext_src_seq, out=extended_attn)

                    step_coverage = coverage / max(1, seq_idx)
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
            loss = loss + self.coverage_weight * torch.sum(
                torch.stack(coverage_loss, dim=-1).view(-1) * (1. - non_pad_mask.float()))

        loss_item = loss.item()
        if self.vocab_criterion.reduction == 'mean':
            loss_item = loss.item() * num_words
        batch_state = Statistics(loss_item, num_words, num_correct_words)

        if math.isnan(loss_item) or loss_item > 1e20:
            logger.info('catch NaN')

        return loss, batch_state


    def compute_loss(self,batch_data, model_output):
        eos_trg = batch_data.question_ids[:, 1:]
        # if self.config['model']['use_pointer']:
        batch_size = eos_trg.size(0)

        logits = torch.stack(model_output['logits'], dim=1)        # batch_size, question_length, vocab_size
        g_prob_t = torch.softmax(logits, dim=-1)
        # g_prob_t = g_prob_t.permute(1, 0, 2)

        c_outputs = torch.stack(model_output['attentions'], dim=1)  # batch_size, question_length, paragraph_length
        c_gate_values = torch.stack(model_output['copy_gates'], dim=1)  # batch_size, question_length, 1
        c_output_prob = c_outputs * c_gate_values.expand_as(c_outputs) + 1e-8
        g_output_prob = g_prob_t * (1 - c_gate_values).expand_as(g_prob_t) + 1e-8

        # not equal.
        c_switch = batch_data.question_ids[:, 1:].ne(batch_data.question_extended_ids_para[:, 1:]).long()  # batch_size, question_length

        c_output_prob_log = torch.log(c_output_prob)
        g_output_prob_log = torch.log(g_output_prob)

        c_output_prob_log = c_output_prob_log * (c_switch.unsqueeze(2).expand_as(c_output_prob_log))        # not equal
        g_output_prob_log = g_output_prob_log * ((1. - c_switch).unsqueeze(2).expand_as(g_output_prob_log))  # equal

        g_output_prob_log = g_output_prob_log.view(-1, g_output_prob_log.size(2))
        c_output_prob_log = c_output_prob_log.view(-1, c_output_prob_log.size(2))

        # crit = nn.NLLLoss(ignore_index=1, reduction='sum')  # 损失函数按权重相加，并且不平均
        # copyCrit = nn.NLLLoss(reduction='sum')

        g_targets = batch_data.question_ids[:, 1:].contiguous().view(-1)
        g_loss = self.vocab_criterion(g_output_prob_log, g_targets)

        c_targets = batch_data.question_extended_ids_para[:, 1:] - self.vocab_size + 1      # TODO. + 1 ？
        c_targets = c_targets.masked_fill((1 - c_switch).bool(), 0).contiguous().view(-1)
        c_loss = self.copy_criterion(c_output_prob_log, c_targets)

        total_loss = g_loss + c_loss
        loss_item = total_loss.item()

        non_pad_mask = g_targets.ne(self.paddding_idx)

        g_mask = non_pad_mask & (1 - c_switch).bool().view(-1)
        g_num_correct_words = g_output_prob_log.max(-1)[1].eq(g_targets).masked_select(g_mask).sum().item()
        c_mask = non_pad_mask & c_switch.bool().view(-1)
        c_num_correct_words = c_output_prob_log.max(-1)[1].eq(c_targets).masked_select(c_mask).sum().item()

        num_correct_words = g_num_correct_words + c_num_correct_words
        num_words = non_pad_mask.sum().item()

        batch_state = Statistics(loss_item, num_words, num_correct_words)

        # Coverage Loss
        # avg_tmp_coverage = coverage_sum / (i + 1)
        # coverage_loss = torch.sum(torch.min(energy, avg_tmp_coverage), dim=1)
        attention_scores = model_output['attentions']
        coverage_scores = model_output['coverages']

        # coverage_loss = 0
        # for decoding_step, (cov_score, attn_score) in enumerate(zip(coverage_scores, attention_scores)):
        #     step_coverage_loss = torch.min(cov_score / (decoding_step + 1), attn_score).sum()
        #     coverage_loss += step_coverage_loss
        # loss = loss + coverage_loss * 0.0
        return total_loss, batch_state


def setup_criterions(generator, configs, tgt_padding_idx, device):
    logger.info('Setup criterions...')
    std_criterion = nn.NLLLoss(reduction='sum', ignore_index=tgt_padding_idx).cuda(device)
    copy_criterion = nn.NLLLoss(reduction='sum').cuda(device) if configs.copy else None
    coverage_criterion = True if configs.coverage else None

    # criterions = Criterion(generator, std_criterion, copy_criterion, coverage_criterion,
    #                         configs.copy_weight, configs.coverage_weight, paddding_idx=tgt_padding_idx)
    # return criterions
