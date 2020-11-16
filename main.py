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
from torch.optim.lr_scheduler import LambdaLR


from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from utils import preprocess_args, init_logger, logger, Statistics
from datasets.common import UNK_TOKEN, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, setup_iterator

from trainer import Trainer
from searcher import BeamSearcher, Hypothesis

# from model import setup_model
from models.eanqg import setup_model, Model
from models.utils import init_parameters, init_embeddings

DEFAULT_CONFIG_NAME = 'config.json'
DEFAULT_BEST_CHECKPOINT_NAME = 'best.ckpt'
DEFAULT_LATEST_CHECKPOINT_NAME = 'latest.ckpt'


def collate_function(examples, pad_token_id=1, device=None):
    batch_size = len(examples)

    examples.sort(key=lambda x: (len(x.paragraph_ids), len(x.question_ids)), reverse=True)
    meta_data = [ex.meta_data for ex in examples]
    paragraph_oov_lst = [ex.paragraph_oov_lst for ex in examples]
    evidences_oov_lst = [ex.evidences_oov_lst for ex in examples]

    max_src_len = max(len(ex.paragraph_ids) for ex in examples)
    max_trg_len = max(len(ex.question_ids) for ex in examples)

    paragraph_ids = torch.LongTensor(batch_size, max_src_len).fill_(pad_token_id).cuda(device)
    paragraph_extended_ids = torch.LongTensor(batch_size, max_src_len).fill_(pad_token_id).cuda(device)
    # paragraph_ans_tag_ids = torch.LongTensor(batch_size, max_src_len).cuda(device)
    paragraph_ans_tag_ids = torch.ones([batch_size, max_src_len], dtype=torch.long, device=device)
    paragraph_pos_tag_ids = torch.LongTensor(batch_size, max_src_len).cuda(device)
    paragraph_ner_tag_ids = torch.LongTensor(batch_size, max_src_len).cuda(device)
    paragraph_dep_tag_ids = torch.LongTensor(batch_size, max_src_len).cuda(device)
    paragraph_cas_tag_ids = torch.LongTensor(batch_size, max_src_len).cuda(device)

    evidences_ids = torch.LongTensor(batch_size, max_src_len).fill_(pad_token_id).cuda(device)
    # evidences_ans_tag_ids = torch.LongTensor(batch_size, max_src_len).fill_(pad_token_id).cuda(device)
    evidences_ans_tag_ids = torch.ones([batch_size, max_src_len], dtype=torch.long, device=device)
    evidences_pos_tag_ids = torch.LongTensor(batch_size, max_src_len).cuda(device)
    evidences_ner_tag_ids = torch.LongTensor(batch_size, max_src_len).cuda(device)
    evidences_dep_tag_ids = torch.LongTensor(batch_size, max_src_len).cuda(device)
    evidences_cas_tag_ids = torch.LongTensor(batch_size, max_src_len).cuda(device)

    question_ids = torch.LongTensor(batch_size, max_trg_len).fill_(pad_token_id).cuda(device)
    question_extended_ids_para = torch.LongTensor(batch_size, max_trg_len).fill_(pad_token_id).cuda(device)
    question_extended_ids_evid = torch.LongTensor(batch_size, max_trg_len).fill_(pad_token_id).cuda(device)

    for idx, example in enumerate(examples):
        assert len(example.paragraph_ids) == len(example.paragraph_extended_ids) == len(example.paragraph_ans_tag_ids)
        paragraph_ids[idx, :len(example.paragraph_ids)] = torch.tensor(example.paragraph_ids)
        paragraph_extended_ids[idx, :len(example.paragraph_extended_ids)] = torch.tensor(example.paragraph_extended_ids)
        paragraph_ans_tag_ids[idx, :len(example.paragraph_ans_tag_ids)] = torch.tensor(example.paragraph_ans_tag_ids)
        paragraph_pos_tag_ids[idx, :len(example.paragraph_pos_tag_ids)] = torch.tensor(example.paragraph_pos_tag_ids)
        paragraph_ner_tag_ids[idx, :len(example.paragraph_ner_tag_ids)] = torch.tensor(example.paragraph_ner_tag_ids)
        paragraph_dep_tag_ids[idx, :len(example.paragraph_dep_tag_ids)] = torch.tensor(example.paragraph_dep_tag_ids)
        paragraph_cas_tag_ids[idx, :len(example.paragraph_cas_tag_ids)] = torch.tensor(example.paragraph_cas_tag_ids)

        evidences_ids[idx, :len(example.evidences_ids)] = torch.tensor(example.evidences_ids)
        evidences_ans_tag_ids[idx, :len(example.evidences_ans_tag_ids)] = torch.tensor(example.evidences_ans_tag_ids)
        evidences_pos_tag_ids[idx, :len(example.evidences_pos_tag_ids)] = torch.tensor(example.evidences_pos_tag_ids)
        evidences_ner_tag_ids[idx, :len(example.evidences_ner_tag_ids)] = torch.tensor(example.evidences_ner_tag_ids)
        evidences_dep_tag_ids[idx, :len(example.evidences_dep_tag_ids)] = torch.tensor(example.evidences_dep_tag_ids)
        evidences_cas_tag_ids[idx, :len(example.evidences_cas_tag_ids)] = torch.tensor(example.evidences_cas_tag_ids)

        assert len(example.question_ids) == len(example.question_extended_ids_para) == len(example.question_extended_ids_evid)
        question_ids[idx, :len(example.question_ids)] = torch.tensor(example.question_ids)
        question_extended_ids_para[idx, :len(example.question_extended_ids_para)] = torch.tensor(
            example.question_extended_ids_para)
        question_extended_ids_evid[idx, :len(example.question_extended_ids_evid)] = torch.tensor(
            example.question_extended_ids_evid)


    return EasyDict({'paragraph_ids': paragraph_ids, 'paragraph_extended_ids': paragraph_extended_ids,
                     'paragraph_ans_tag_ids': paragraph_ans_tag_ids,
                     'paragraph_pos_tag_ids': paragraph_pos_tag_ids, 'paragraph_ner_tag_ids': paragraph_ner_tag_ids,
                     'paragraph_dep_tag_ids': paragraph_dep_tag_ids, 'paragraph_cas_tag_ids': paragraph_cas_tag_ids,

                     'evidences_ids': evidences_ids,
                     'evidences_ans_tag_ids': evidences_ans_tag_ids,
                     'evidences_pos_tag_ids': evidences_pos_tag_ids, 'evidences_ner_tag_ids': evidences_ner_tag_ids,
                     'evidences_dep_tag_ids': evidences_dep_tag_ids, 'evidences_cas_tag_ids': evidences_cas_tag_ids,

                     'question_ids': question_ids,
                     'question_extended_ids_para': question_extended_ids_para,
                     'question_extended_ids_evid': question_extended_ids_evid,

                     'paragraph_oov_lst': paragraph_oov_lst, 'evidences_oov_lst': evidences_oov_lst,

                     'pad_token_id': pad_token_id,

                     'meta_data': meta_data, 'batch_size': batch_size})


def main(args, device=None):

    if args.train:
        # 0. Configuration.
        config = preprocess_args(args)
        init_logger(config['log_file'])
        logger.info('Config = %s' % config)

        # Device
        # seed = args.seed
        seed = random.randint(1, 1000000)
        config.seed = seed

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # 1. Setup data
        logger.info('Setup data ...')
        logger.info("Loading train dataset from %s" % config['cached_train_path'])
        train_dataset = torch.load(config['cached_train_path'])
        logger.info("Loading dev dataset from %s" % config['cached_dev_path'])
        valid_dataset = torch.load(config['cached_dev_path'])
        logger.info("Loading vocabularies from %s" % config['cached_vocabularies_path'])
        vocabularies = torch.load(config['cached_vocabularies_path'])

        PAD_INDEX = vocabularies['token'].stoi[PAD_TOKEN]
        TRG_UNK_INDEX = vocabularies['token'].stoi[UNK_TOKEN]

        # Some related configuration about embeddings.
        config.model['vocab_size'] = len(vocabularies['token'])
        config.model['feature_tag_vocab_size'] = len(vocabularies['feature'])
        config.model['answer_tag_vocab_size'] = len(vocabularies['answer'])
        config.model.pad_token_id = PAD_INDEX
        config.model.unk_token_id = TRG_UNK_INDEX

        # Save configuration.
        config_object = {k: v for k, v in config.items()}
        with open(os.path.join(config.save_path, DEFAULT_CONFIG_NAME), 'w') as json_writer:
            json.dump(config_object, json_writer, indent=4)

        collate_fn = functools.partial(collate_function, pad_token_id=PAD_INDEX, device=device)

        train_iter = setup_iterator(dataset=train_dataset, collate_fn=collate_fn,
                                    batch_size=config['batch_size'], random=True)
        valid_iter = setup_iterator(dataset=valid_dataset, collate_fn=collate_fn,
                                    batch_size=128, random=False)

        # 2. Setup model
        logger.info('Setup model ...')
        model = Model(config['model'])
        init_parameters(model, config)
        logger.info('Initializing shared source and target embedding with glove.6B.300d.')
        init_embeddings(model.embeddings.word_embeddings, vocabularies['token'])
        model.to(device)
        logger.info('Model = %s' % model)

        # model = setup_model(vocabularies, PAD_INDEX, TRG_UNK_INDEX, config, device)

        # 3. Setup optimizer
        if config['optimizer_name'] == 'sgd':
            # optimizer = torch.optim.SGD(model.parameters(), lr=config['sgd_learning_rate'])
            optimizer = torch.optim.SGD(model.parameters(), lr=config['sgd_learning_rate'], momentum=0.8)
            # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
            init_learning_rate = config['sgd_learning_rate']

            criterion = nn.CrossEntropyLoss(ignore_index=PAD_INDEX)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=config['adam_learning_rate'])
            init_learning_rate = config.adam_learning_rate

            criterion = nn.CrossEntropyLoss(ignore_index=PAD_INDEX, reduction='sum')

        logger.info('Setup %s optimizer with initialized learning rate = %.5f' %
                    (config.optimizer_name, init_learning_rate))

        # lr_scheduler = LambdaLR(optimizer, lr_lambda=lambda lr: lr * 0.5)
        lr_scheduler = None

        # 4. Setup criterion
        logger.info('Setup cross-entropy criterion ')
        # criterion = nn.CrossEntropyLoss(ignore_index=PAD_INDEX, reduction='sum')
        # criterion = nn.CrossEntropyLoss(ignore_index=PAD_INDEX)

        # 5. Setup trainer
        trainer = Trainer(vocabularies, model, optimizer, lr_scheduler, criterion, config['save_path'], config)

        trainer.train(train_iter, valid_iter, config.num_train_epochs, train_from=args.train_from)
        # trainer.customized_train(train_iter, valid_iter, 20)

    if args.test:
        if not args.train:

            # 0. Configuration.
            if args.test_from_dir is not None:
                test_directory = args.test_from_dir
                checkpoint_path = os.path.join(test_directory, DEFAULT_LATEST_CHECKPOINT_NAME)
            elif args.test_from_model is not None:
                test_directory  = os.path.dirname(args.test_from_model)
                checkpoint_path = args.test_from_model

            else:
                raise NotImplementedError('Test directory %s is not valid' % args.test_from_dir)

            with open(os.path.join(test_directory, DEFAULT_CONFIG_NAME), 'r') as json_reader:
                config_object = json.load(json_reader)
            config = EasyDict(config_object)

            config.save_path = test_directory
        else:
            checkpoint_path = os.path.join(config.save_path, DEFAULT_BEST_CHECKPOINT_NAME)

        log_file = os.path.join(config.save_path, 'test.log')
        init_logger(log_file)
        logger.info('config = %s' % config)

        # 1. Setup data
        logger.info("Loading vocabularies from %s" % config['cached_vocabularies_path'])
        vocabularies = torch.load(config['cached_vocabularies_path'])
        PAD_INDEX = vocabularies['token'].stoi[PAD_TOKEN]
        TRG_UNK_INDEX = vocabularies['token'].stoi[UNK_TOKEN]

        # Some related configuration about embeddings.
        config.model['vocab_size'] = len(vocabularies['token'])
        config.model['feature_tag_vocab_size'] = len(vocabularies['feature'])
        config.model['answer_tag_vocab_size'] = len(vocabularies['answer'])
        config.model.pad_token_id = PAD_INDEX
        config.model.unk_token_id = TRG_UNK_INDEX

        logger.info("Loading test dataset from %s" % config['cached_test_path'])
        test_dataset = torch.load(config.cached_test_path)

        collate_fn = functools.partial(collate_function, pad_token_id=PAD_INDEX, device=device)

        test_iter = setup_iterator(dataset=test_dataset, collate_fn=collate_fn, batch_size=1, random=False)

        # 2. Setup model
        logger.info('Building model ...')
        # model = setup_model(vocabularies, PAD_INDEX, TRG_UNK_INDEX, config, device, checkpoint_path)
        model = Model(config['model'])
        logger.info('Loading checkpoint from %s' % checkpoint_path)
        model.load_state_dict(torch.load(checkpoint_path))
        model.to(device)
        model.eval()

        # 3. Setup searcher
        searcher = BeamSearcher(vocabularies, test_iter, model, config.save_path,
                                config.beam_size, config.min_decode_step, config.max_decode_step)
        searcher.search()


if __name__ == '__main__':
    # config
    parser = argparse.ArgumentParser(description='SSR')
    parser.add_argument('--config', '-config', type=str, default='configs/eanqg_squad_split_v2.yml')

    parser.add_argument('--gpu', '-gpu', type=int, default=0)
    parser.add_argument('--seed', '-seed', type=int, default=73157)

    parser.add_argument('--train', '-train', action='store_true', default=False)
    parser.add_argument('--train_from', '-train_from', type=str, default=None)

    parser.add_argument('--test', '-test', action='store_true', default=False)
    parser.add_argument('--test_from_dir', '-test_from_dir', type=str, default=None,
                        help='directory')
    parser.add_argument('--test_from_model', '-test_from_model', type=str, default=None,
                        help='checkpoint')


    args = parser.parse_args()

    assert args.train or args.test, 'train or test, choose one!'

    device = torch.device('cuda:%d' % args.gpu  if args.gpu > -1 else 'cpu')

    main(args, device=device)
