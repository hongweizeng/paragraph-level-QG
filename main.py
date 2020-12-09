import os
import json
import argparse
import random
import numpy as np
import torch
from easydict import EasyDict

from utils import preprocess_args, init_logger, logger
from datasets.common import UNK_TOKEN, PAD_TOKEN

from train.trainer import Trainer
from search.searcher import Searcher

# from model import setup_model
from models.utils import init_parameters, init_embeddings, freeze_module
# from models.master import Model
from models.eanqg import Model
from train.checkpoint_manager import Checkpoint, CheckpointManager

DEFAULT_CONFIG_NAME = 'config.json'
DEFAULT_BEST_CHECKPOINT_NAME = 'best.ckpt'
DEFAULT_LATEST_CHECKPOINT_NAME = 'latest.ckpt'


def reset_configs(config):
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

    config.train.pad_token_id = PAD_INDEX
    config.train.vocab_size = len(vocabularies['token'])

    return vocabularies, config


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

        vocabularies, config = reset_configs(config)

        # logger.info("Loading vocabularies from %s" % config['cached_vocabularies_path'])
        # vocabularies = torch.load(config['cached_vocabularies_path'])
        #
        # PAD_INDEX = vocabularies['token'].stoi[PAD_TOKEN]
        # TRG_UNK_INDEX = vocabularies['token'].stoi[UNK_TOKEN]
        #
        # # Some related configuration about embeddings.
        # config.model['vocab_size'] = len(vocabularies['token'])
        # config.model['feature_tag_vocab_size'] = len(vocabularies['feature'])
        # config.model['answer_tag_vocab_size'] = len(vocabularies['answer'])
        # config.model.pad_token_id = PAD_INDEX
        # config.model.unk_token_id = TRG_UNK_INDEX
        #
        # config.train.pad_token_id = PAD_INDEX

        config.train_from = args.train_from

        # Save configuration.
        config_object = {k: v for k, v in config.items()}
        with open(os.path.join(config.save_path, DEFAULT_CONFIG_NAME), 'w') as json_writer:
            json.dump(config_object, json_writer, indent=4)



        # 2. Setup model
        logger.info('Setup model ...')
        model = Model(config['model'])
        init_parameters(model, config)
        logger.info('Initializing shared source and target embedding with glove.6B.300d.')
        init_embeddings(model.embeddings.word_embeddings, vocabularies['token'])
        logger.info('Freezing embeddings.')
        freeze_module(model.embeddings.word_embeddings)
        model.to(device)
        logger.info('Model = %s' % model)

        # model = setup_model(vocabularies, PAD_INDEX, TRG_UNK_INDEX, config, device)

        # # 3. Setup optimizer
        #
        #
        # if config['optimizer_name'] == 'sgd':
        #     # optimizer = torch.optim.SGD(model.parameters(), lr=config['sgd_learning_rate'])
        #     optimizer = torch.optim.SGD(model.parameters(), lr=config['sgd_learning_rate'], momentum=0.8)
        #     # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
        #     init_learning_rate = config['sgd_learning_rate']
        #
        #     criterion = nn.CrossEntropyLoss(ignore_index=PAD_INDEX)
        # else:
        #     optimizer = torch.optim.Adam(model.parameters(), lr=config['adam_learning_rate'], weight_decay=0.000001)
        #     # optimizer = torch.optim.Adam(model.parameters(), lr=config['adam_learning_rate'])
        #     init_learning_rate = config.adam_learning_rate
        #
        #     criterion = nn.CrossEntropyLoss(ignore_index=PAD_INDEX, reduction='sum')
        #
        # logger.info('Setup %s optimizer with initialized learning rate = %.5f' %
        #             (config.optimizer_name, init_learning_rate))
        #
        #
        #
        # # 4. Setup criterion
        # logger.info('Setup cross-entropy criterion ')
        # # criterion = nn.CrossEntropyLoss(ignore_index=PAD_INDEX, reduction='sum')
        # # criterion = nn.CrossEntropyLoss(ignore_index=PAD_INDEX)

        # 5. Setup trainer
        trainer = Trainer(train_dataset, valid_dataset,vocabularies, model, config)
        trainer.train(train_from=args.train_from)

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

            log_file = os.path.join(config.save_path, 'test.log')
            init_logger(log_file)
            logger.info('config = %s' % config)

        else:
            checkpoint_path = os.path.join(config.save_path, DEFAULT_LATEST_CHECKPOINT_NAME)

        # 1. Setup data
        vocabularies, config = reset_configs(config)
        logger.info("Loading test dataset from %s" % config['cached_test_path'])
        test_dataset = torch.load(config.cached_test_path)

        # 2. Setup model
        logger.info('Building model ...')
        model = Model(config['model'])
        checkpoint_manager: CheckpointManager = CheckpointManager(config['train']['checkpoint'],
                                                                  save_path=config['save_path'])

        if checkpoint_path == 'best':
            logger.info('Loading best checkpoint from directory %s' % checkpoint_manager.best_checkpoint_path)
            checkpoint: Checkpoint = checkpoint_manager.load_best_checkpoint()
        elif checkpoint_path == 'latest':
            logger.info('Loading best checkpoint from directory %s' % checkpoint_manager.best_checkpoint_path)
            checkpoint: Checkpoint = checkpoint_manager.load_latest_checkpoint()
        else:
            checkpoint: Checkpoint = torch.load(checkpoint_path)

        model.load_state_dict(checkpoint.model_state_dict)
        model.to(device)
        model.eval()

        # 3. Setup searcher
        search_config = config['inference']
        searcher = Searcher(vocabularies, test_dataset, model, config.save_path, search_config)
        searcher.search()


if __name__ == '__main__':
    # config
    parser = argparse.ArgumentParser(description='SSR')
    parser.add_argument('--config', '-config', type=str, default='configs/test.yml',
                        choices=['configs/du_acl2017.yml', 'configs/zhou_nlpcc2017.yml', 'configs/zhao_emnlp2018.yml',
                                 'configs/eanqg_newsqa.yml', 'configs/test.yml', 'configs/master.yml'])

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
