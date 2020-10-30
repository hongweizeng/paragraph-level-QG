import os
import json
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import functools
from easydict import EasyDict
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
import torch.nn.functional as F
import yaml
import time
from easydict import EasyDict

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from utils import parse_configs, init_logger, logger, Statistics

from data import setup_dataset, PAD_TOKEN, UNK_TOKEN, QgDataset, Example

from trainer import Trainer
from searcher import BeamSearcher, Hypothesis

from model import setup_model


DEFAULT_CONFIG_NAME = 'config.json'
DEFAULT_BEST_CHECKPOINT_NAME = 'best.ckpt'


def collate_function(examples,
                     src_padding_idx=1, ans_padding_idx=1, feat_padding_idx=1, tgt_padding_idx=1,
                     device=None):
    batch_size = len(examples)

    examples.sort(key=lambda x: (len(x.src_ids), len(x.trg_ids)), reverse=True)
    meta_data = [ex.meta_data for ex in examples]
    oov_lst = [ex.oov_lst for ex in examples]

    max_src_len = max(len(ex.src_ids) for ex in examples)
    # max_src_len = min(max(len(ex.src_ids) for ex in examples), 200)     # TODO, handle this magic operation.
    max_trg_len = max(len(ex.trg_ids) for ex in examples)

    src_seqs = torch.LongTensor(batch_size, max_src_len).fill_(src_padding_idx).cuda(device)
    ext_src_seqs = torch.LongTensor(batch_size, max_src_len).fill_(src_padding_idx).cuda(device)

    tag_seqs = torch.LongTensor(batch_size, max_src_len).fill_(ans_padding_idx).cuda(device)
    # pos_seq = torch.LongTensor(batch_size, max_src_len).fill_(feat_padding_idx).cuda(device)
    # ner_seq = torch.LongTensor(batch_size, max_src_len).fill_(feat_padding_idx).cuda(device)
    # cas_seq = torch.LongTensor(batch_size, max_src_len).fill_(feat_padding_idx).cuda(device)

    # relation_mask = torch.zeros([batch_size, max_src_len, max_src_len], dtype=torch.long, device=device)
    coreference_mask = torch.zeros([batch_size, max_src_len, max_src_len], dtype=torch.long, device=device)
    dependency_mask = torch.zeros([batch_size, max_src_len, max_src_len], dtype=torch.long, device=device)
    constituency_mask = torch.zeros([batch_size, max_src_len, max_src_len], dtype=torch.long, device=device)

    dependency_hop_distance = torch.zeros([batch_size, max_src_len, max_src_len], dtype=torch.long, device=device)
    constituency_hop_distance = torch.zeros([batch_size, max_src_len, max_src_len], dtype=torch.long, device=device)
    coreference_hop_distance = torch.zeros([batch_size, max_src_len, max_src_len], dtype=torch.long, device=device)
    dep_and_con_hop_distance = torch.zeros([batch_size, max_src_len, max_src_len], dtype=torch.long, device=device)
    dep_and_cor_hop_distance = torch.zeros([batch_size, max_src_len, max_src_len], dtype=torch.long, device=device)
    con_and_cor_hop_distance = torch.zeros([batch_size, max_src_len, max_src_len], dtype=torch.long, device=device)
    all_hop_distance = torch.zeros([batch_size, max_src_len, max_src_len], dtype=torch.long, device=device)

    trg_seqs = torch.LongTensor(batch_size, max_trg_len).fill_(tgt_padding_idx).cuda(device)
    ext_trg_seqs = torch.LongTensor(batch_size, max_trg_len).fill_(tgt_padding_idx).cuda(device)



    # batch = {'src': [], 'ans': [], 'pos': [], 'ner': [], 'cas': [], 'trg': []}
    for idx, example in enumerate(examples):
        assert len(example.src_ids) == len(example.src_extended_ids) == len(example.ans_tag_ids)
        src_seqs[idx, :len(example.src_ids)] = torch.tensor(example.src_ids)
        ext_src_seqs[idx, :len(example.src_extended_ids)] = torch.tensor(example.src_extended_ids)

        tag_seqs[idx, :len(example.ans_tag_ids)] = torch.tensor(example.ans_tag_ids)
        # pos_seq[idx, :len(example.ans_tag_ids)] = torch.tensor(example.ans_tag_ids)
        # ner_seq[idx, :len(example.ans_tag_ids)] = torch.tensor(example.ans_tag_ids)
        # cas_seq[idx, :len(example.ans_tag_ids)] = torch.tensor(example.ans_tag_ids)

        # ——————————————————    HOP RELATION    --------------------------
        coreference_mask[idx, :len(example.src_ids), :len(example.src_ids)] = \
            torch.from_numpy(example.meta_data['coreference_mask_spatial'].todense())
        dependency_mask[idx, :len(example.src_ids), :len(example.src_ids)] = \
            torch.from_numpy(example.meta_data['dependency_mask_spatial'].todense())
        constituency_mask[idx, :len(example.src_ids), :len(example.src_ids)] = \
            torch.from_numpy(example.meta_data['constituency_mask_spatial'].todense())
        # ——————————————————    HOP DISTANCE    --------------------------


        # ——————————————————    HOP DISTANCE    --------------------------
        dependency_hop_distance[idx, :len(example.src_ids), :len(example.src_ids)] = torch.from_numpy(
            example.meta_data['dependency_hop_distance'])
        constituency_hop_distance[idx, :len(example.src_ids), :len(example.src_ids)] = torch.from_numpy(
            example.meta_data['constituency_hop_distance'])
        coreference_hop_distance[idx, :len(example.src_ids), :len(example.src_ids)] = torch.from_numpy(
            example.meta_data['coreference_hop_distance'])
        dep_and_con_hop_distance[idx, :len(example.src_ids), :len(example.src_ids)] = torch.from_numpy(
            example.meta_data['dep_and_con_hop_distance'])
        dep_and_cor_hop_distance[idx, :len(example.src_ids), :len(example.src_ids)] = torch.from_numpy(
            example.meta_data['dep_and_cor_hop_distance'])
        con_and_cor_hop_distance[idx, :len(example.src_ids), :len(example.src_ids)] = torch.from_numpy(
            example.meta_data['con_and_cor_hop_distance'])
        all_hop_distance[idx, :len(example.src_ids), :len(example.src_ids)] = torch.from_numpy(
            example.meta_data['all_hop_distance'])
        # ——————————————————    HOP DISTANCE    --------------------------


        assert len(example.trg_ids) == len(example.trg_extended_ids)
        trg_seqs[idx, :len(example.trg_ids)] = torch.tensor(example.trg_ids)
        ext_trg_seqs[idx, :len(example.trg_extended_ids)] = torch.tensor(example.trg_extended_ids)


    return EasyDict({'src_seq': src_seqs, 'ext_src_seq': ext_src_seqs, 'tag_seq': tag_seqs,
                     'src_padding_index': src_padding_idx,
                     'coreference_mask': coreference_mask,
                     'dependency_mask': dependency_mask,
                     'constituency_mask': constituency_mask,
                     'dependency_hop_distance': dependency_hop_distance,
                     'constituency_hop_distance': constituency_hop_distance,
                     'coreference_hop_distance': coreference_hop_distance,
                     'dep_and_con_hop_distance': dep_and_con_hop_distance,
                     'dep_and_cor_hop_distance': dep_and_cor_hop_distance,
                     'con_and_cor_hop_distance': con_and_cor_hop_distance,
                     'all_hop_distance': all_hop_distance,
                     'trg_seq': trg_seqs, 'ext_trg_seq': ext_trg_seqs,
                     'oov_lst': oov_lst,
                     'meta_data': meta_data, 'batch_size': batch_size})


class CustomizedTrainer(Trainer):
    def run_batch(self, batch_data):
        # src_seq, ext_src_seq, trg_seq, ext_trg_seq, tag_seq, _ = batch
        eos_trg = batch_data.trg_seq[:, 1:]
        if self.config['use_pointer']:
            eos_trg = batch_data.ext_trg_seq[:, 1:]

        logits = self.model.forward(batch_data, src_padding_idx=self.padding_index)

        batch_size, nsteps, _ = logits.size()
        preds = logits.view(batch_size * nsteps, -1)
        targets = eos_trg.contiguous().view(-1)
        loss = self.criterion(preds, targets)

        non_pad_mask = targets.ne(self.padding_index)
        num_correct_words = preds.max(-1)[1].eq(targets).masked_select(non_pad_mask).sum().item()
        num_words = non_pad_mask.sum().item()
        batch_state = Statistics(loss.item(), num_words, num_correct_words)

        return loss, batch_state


class CustomizedSearcher(BeamSearcher):
    def search_batch(self, batch_data):
        src_seq, ext_src_seq, tag_seq = batch_data.src_seq, batch_data.ext_src_seq, batch_data.tag_seq
        src_padding_idx = self.PAD_INDEX

        enc_mask = (src_seq != src_padding_idx)

        enc_outputs, enc_states = self.model.encoder(batch_data)

        prev_context = torch.zeros(1, 1, enc_outputs.size(-1)).cuda(device=self.device)

        h, c = enc_states  # [2, b, d] but b = 1
        hypotheses = [Hypothesis(tokens=[self.TRG_SOS_INDEX],
                                 log_probs=[0.0],
                                 state=(h[:, 0, :], c[:, 0, :]),
                                 context=prev_context[0]) for _ in range(self.beam_size)]
        # tile enc_outputs, enc_mask for beam search
        ext_src_seq = ext_src_seq.repeat(self.beam_size, 1)
        enc_outputs = enc_outputs.repeat(self.beam_size, 1, 1)
        enc_features = self.model.decoder.get_encoder_features(enc_outputs)
        enc_mask = enc_mask.repeat(self.beam_size, 1)
        num_steps = 0
        results = []
        while num_steps < self.config.max_decode_step and len(results) < self.beam_size:
            latest_tokens = [h.latest_token for h in hypotheses]
            latest_tokens = [idx if idx < len(
                self.tok2idx) else self.TRG_UNK_INDEX for idx in latest_tokens]
            prev_y = torch.tensor(latest_tokens, dtype=torch.long, device=self.device).view(-1)

            # if config.use_gpu:
            #     prev_y = prev_y.to(self.device)

            # make batch of which size is beam size
            all_state_h = []
            all_state_c = []
            all_context = []
            for h in hypotheses:
                state_h, state_c = h.state  # [num_layers, d]
                all_state_h.append(state_h)
                all_state_c.append(state_c)
                all_context.append(h.context)

            prev_h = torch.stack(all_state_h, dim=1)  # [num_layers, beam, d]
            prev_c = torch.stack(all_state_c, dim=1)  # [num_layers, beam, d]
            prev_context = torch.stack(all_context, dim=0)
            prev_states = (prev_h, prev_c)
            # [beam_size, |V|]
            logits, states, context_vector = self.model.decoder.decode(prev_y, ext_src_seq,
                                                                       prev_states, prev_context,
                                                                       enc_features, enc_mask)
            h_state, c_state = states
            log_probs = F.log_softmax(logits, dim=1)
            top_k_log_probs, top_k_ids \
                = torch.topk(log_probs, self.beam_size * 2, dim=-1)

            all_hypotheses = []
            num_orig_hypotheses = 1 if num_steps == 0 else len(hypotheses)
            for i in range(num_orig_hypotheses):
                h = hypotheses[i]
                state_i = (h_state[:, i, :], c_state[:, i, :])
                context_i = context_vector[i]
                for j in range(self.beam_size * 2):
                    new_h = h.extend(token=top_k_ids[i][j].item(),
                                     log_prob=top_k_log_probs[i][j].item(),
                                     state=state_i,
                                     context=context_i)
                    all_hypotheses.append(new_h)

            hypotheses = []
            for h in self.sort_hypotheses(all_hypotheses):
                if h.latest_token == self.TRG_EOS_INDEX:
                    if num_steps >= self.min_decode_step:
                        results.append(h)
                else:
                    hypotheses.append(h)

                if len(hypotheses) == self.beam_size or len(results) == self.beam_size:
                    break
            num_steps += 1
        if len(results) == 0:
            results = hypotheses
        h_sorted = self.sort_hypotheses(results)

        return h_sorted[0]


def main(args, device=None):

    if args.train:
        # 0. Configuration.
        configs = parse_configs(args)
        init_logger(configs['log_file'])
        logger.info('Configs = %s' % configs)

        # Device
        # seed = args.seed
        seed = random.randint(1, 1000000)
        configs.seed = seed

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # 1. Setup data
        logger.info('Setup data ...')
        logger.info("Loading train dataset from %s" % configs['cached_train_path'])
        train_dataset = torch.load(configs['cached_train_path'])
        logger.info("Loading dev dataset from %s" % configs['cached_dev_path'])
        valid_dataset = torch.load(configs['cached_dev_path'])
        logger.info("Loading vocabularies from %s" % configs['cached_vocabularies_path'])
        vocabularies = torch.load(configs['cached_vocabularies_path'])

        PAD_INDEX = vocabularies['token'].stoi[PAD_TOKEN]
        TRG_UNK_INDEX = vocabularies['token'].stoi[UNK_TOKEN]

        collate_fn = functools.partial(collate_function,
                                       src_padding_idx=PAD_INDEX,
                                       ans_padding_idx=PAD_INDEX,
                                       feat_padding_idx=PAD_INDEX,
                                       tgt_padding_idx=PAD_INDEX,
                                       device=device)

        train_iter = DataLoader(dataset=train_dataset, batch_size=configs['batch_size'],
                                collate_fn=collate_fn, sampler=RandomSampler(train_dataset))
        valid_iter = DataLoader(dataset=valid_dataset, batch_size=configs['batch_size'],
                                collate_fn=collate_fn, sampler=SequentialSampler(valid_dataset))

        # 2. Setup model
        logger.info('Building model ...')
        model = setup_model(vocabularies, PAD_INDEX, TRG_UNK_INDEX, configs, device)

        # 3. Setup optimizer
        # optimizer = torch.optim.SGD(model.parameters(), lr=configs.lr)
        optimizer = torch.optim.Adam(model.parameters(), lr=configs['learning_rate'])

        # 4. Setup criterion
        criterion = nn.CrossEntropyLoss(ignore_index=PAD_INDEX, reduction='sum')

        # 5. Setup trainer
        trainer = CustomizedTrainer(vocabularies, model, optimizer, criterion, configs['save_path'], configs)

        trainer.train(train_iter, valid_iter, configs.num_train_epochs, train_from=args.train_from)

    if args.test:
        # 0. Configuration.
        if not args.train:
            assert args.test_from_dir is not None and os.path.isdir(args.test_from_dir), \
                'Test directory %s is not valid' % args.test_from_dir

            with open(os.path.join(args.test_from_dir, DEFAULT_CONFIG_NAME), 'r') as json_reader:
                config_object = json.load(json_reader)
            configs = EasyDict(config_object)

        log_file = os.path.join(configs['save_path'], 'test.log')
        init_logger(log_file)
        logger.info('Configs = %s' % configs)

        checkpoint = os.path.join(configs['save_path'], DEFAULT_BEST_CHECKPOINT_NAME)

        # 1. Setup data
        logger.info("Loading vocabularies from %s" % configs['cached_vocabularies_path'])
        vocabularies = torch.load(configs['cached_vocabularies_path'])
        PAD_INDEX = vocabularies['token'].stoi[PAD_TOKEN]
        TRG_UNK_INDEX = vocabularies['token'].stoi[UNK_TOKEN]

        logger.info("Loading test dataset from %s" % configs['cached_test_path'])
        test_dataset = torch.load(configs['cached_test_path'])

        collate_fn = functools.partial(collate_function,
                                       src_padding_idx=PAD_INDEX,
                                       ans_padding_idx=PAD_INDEX,
                                       feat_padding_idx=PAD_INDEX,
                                       tgt_padding_idx=PAD_INDEX,
                                       device=device)

        test_iter = DataLoader(dataset=test_dataset, batch_size=1,
                                collate_fn=collate_fn, sampler=SequentialSampler(test_dataset))

        # 2. Setup model
        logger.info('Building model ...')
        model = setup_model(vocabularies, PAD_INDEX, TRG_UNK_INDEX, configs, device, checkpoint)
        model.eval()

        # 3. Setup searcher
        searcher = CustomizedSearcher(vocabularies, test_iter, model, configs.save_path, device=device, config=configs)
        searcher.search()


if __name__ == '__main__':
    # Configs
    parser = argparse.ArgumentParser(description='SSR')
    parser.add_argument('--config', '-config', type=str, default='configs/squad_split_v2.yml')

    parser.add_argument('--gpu', '-gpu', type=int, default=0)
    parser.add_argument('--seed', '-seed', type=int, default=73157)

    parser.add_argument('--train', '-train', action='store_true', default=False)
    parser.add_argument('--train_from', '-train_from', type=str, default=None)

    parser.add_argument('--test', '-test', action='store_true', default=False)
    parser.add_argument('--test_from_dir', '-test_from_dir', type=str, default=None)


    args = parser.parse_args()

    assert args.train or args.test, 'train or test, choose one!'

    device = torch.device('cuda:%d' % args.gpu  if args.gpu > -1 else 'cpu')

    main(args, device=device)
