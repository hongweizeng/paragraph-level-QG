import os
import json
import random
import argparse
from tqdm import tqdm

import torch

from utils.logging import init_logger
logger = init_logger()

from datasets.common import setup_vocab, QgDataset, Example, UNK_TOKEN, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, \
    get_answer_tag, context2ids, question2ids, parse_text_with_stanza


from datasets.squad import read_squad_qas_dict, read_squad_examples
from datasets.newsqa import read_newsqa_examples


def build_vocabularies(examples, vocab_size, min_word_frequency, directory):
    logger.info("Setup vocabularies...")
    logger.info('Setup token vocabulary: Source and Target shares vocabulary.')
    train_src = [ex.meta_data['paragraph']['tokens'] for ex in examples]
    train_trg = [ex.meta_data['question']['tokens'] for ex in examples]
    specials = [UNK_TOKEN, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN]
    corpus = train_src + train_trg
    token_vocab = setup_vocab(corpus, specials=specials, max_size=vocab_size, min_freq=min_word_frequency)

    logger.info('Setup answer tagging vocabulary: (BIO)')
    answer_corpus = ['B', 'I', 'O']
    answer_vocab = setup_vocab(answer_corpus, specials=[UNK_TOKEN, PAD_TOKEN])

    logger.info('Setup feature vocabulary: pos, ner, dep, cas')
    train_ner = [ex.meta_data['paragraph']['ner_tags'] for ex in examples]
    train_pos = [ex.meta_data['paragraph']['pos_tags'] for ex in examples]
    train_dep = [ex.meta_data['paragraph']['dep_tags'] for ex in examples]
    train_cas = [ex.meta_data['paragraph']['cas_tags'] for ex in examples]
    feature_corpus = train_ner + train_pos + train_dep + train_cas
    feature_vocab = setup_vocab(feature_corpus, specials=[UNK_TOKEN, PAD_TOKEN])

    vocabularies = {'token': token_vocab, 'answer': answer_vocab, 'feature': feature_vocab}
    cached_vocabularies_path = os.path.join(directory, 'vocab.pt')
    logger.info('Save vocabularies into %s' % cached_vocabularies_path)
    torch.save(vocabularies, cached_vocabularies_path)
    return vocabularies


def process_features(examples, corpus_type, vocabularies):
    with tqdm(total=len(examples), desc='Processing %s features' % corpus_type) as t:
        for example in examples:
            # Paragraph Input
            example.paragraph_ids, example.paragraph_extended_ids, example.paragraph_oov_lst = context2ids(
                example.meta_data['paragraph']['tokens'], vocabularies['token'].stoi)
            example.paragraph_ner_tag_ids = [vocabularies['feature'].stoi[tag]
                                             for tag in example.meta_data['paragraph']['ner_tags']]
            example.paragraph_pos_tag_ids = [vocabularies['feature'].stoi[tag]
                                             for tag in example.meta_data['paragraph']['pos_tags']]
            example.paragraph_dep_tag_ids = [vocabularies['feature'].stoi[tag]
                                             for tag in example.meta_data['paragraph']['dep_tags']]
            example.paragraph_cas_tag_ids = [vocabularies['feature'].stoi[tag]
                                             for tag in example.meta_data['paragraph']['cas_tags']]
            example.paragraph_ans_tag_ids = [vocabularies['answer'].stoi[tag]
                                             for tag in example.meta_data['paragraph_ans_tag']]

            # Evidences Input
            example.evidences_ids, example.evidences_extended_ids, example.evidences_oov_lst = context2ids(
                example.meta_data['evidences']['tokens'], vocabularies['token'].stoi)
            example.evidences_ner_tag_ids = [vocabularies['feature'].stoi[tag]
                                             for tag in example.meta_data['evidences']['ner_tags']]
            example.evidences_pos_tag_ids = [vocabularies['feature'].stoi[tag]
                                             for tag in example.meta_data['evidences']['pos_tags']]
            example.evidences_dep_tag_ids = [vocabularies['feature'].stoi[tag]
                                             for tag in example.meta_data['evidences']['dep_tags']]
            example.evidences_cas_tag_ids = [vocabularies['feature'].stoi[tag]
                                             for tag in example.meta_data['evidences']['cas_tags']]
            example.evidences_ans_tag_ids = [vocabularies['answer'].stoi[tag]
                                             for tag in example.meta_data['evidences_ans_tag']]

            # Question Input
            example.question_ids, example.question_extended_ids_para = question2ids(
                example.meta_data['question']['tokens'], vocabularies['token'].stoi, example.paragraph_oov_lst)

            _, example.question_extended_ids_evid = question2ids(
                example.meta_data['question']['tokens'], vocabularies['token'].stoi, example.evidences_oov_lst)

            # Answer Input
            example.answer_ids, _, _ = context2ids(
                example.meta_data['answer']['tokens'], vocabularies['token'].stoi)
            example.answer_ner_tag_ids = [vocabularies['feature'].stoi[tag]
                                             for tag in example.meta_data['answer']['ner_tags']]
            example.answer_pos_tag_ids = [vocabularies['feature'].stoi[tag]
                                             for tag in example.meta_data['answer']['pos_tags']]
            example.answer_dep_tag_ids = [vocabularies['feature'].stoi[tag]
                                             for tag in example.meta_data['answer']['dep_tags']]
            example.answer_cas_tag_ids = [vocabularies['feature'].stoi[tag]
                                             for tag in example.meta_data['answer']['cas_tags']]

            t.update()
    t.close()


def setup_dataset(directory, corpus_type, vocabularies, qas_id_dict, vocab_size=50000, min_word_frequency=1,
                  recover=True):
    logger.info('Setup %s dataset in directory %s' % (corpus_type, directory))

    cached_dataset_path = os.path.join(directory, corpus_type + '.pt')
    cached_vocabularies_path = os.path.join(directory, 'vocab.pt')

    if recover and os.path.exists(cached_dataset_path) and os.path.exists(cached_vocabularies_path):
        logger.info('Loading %s dataset from %s' % (corpus_type, cached_dataset_path))
        dataset = torch.load(cached_dataset_path)
        if corpus_type == 'train':
            logger.info('Loading vocabularies from %s' % cached_vocabularies_path)
            vocabularies = torch.load(cached_vocabularies_path)
            return dataset, vocabularies

    if 'squad' in directory:
        qas_id_dict = {**qas_id_dict['train'], **qas_id_dict['dev']}
        examples = read_squad_examples(directory=directory, corpus_type=corpus_type, qas_id_dict=qas_id_dict)
    elif 'newsqa' in directory:
        examples = read_newsqa_examples(directory=directory, corpus_type=corpus_type)
    elif 'test' in directory:
        from datasets.test import read_squad_examples_without_ids
        examples = read_squad_examples_without_ids(corpus_type=corpus_type, qas_id_dict=qas_id_dict)


    if corpus_type == 'train':
        vocabularies = build_vocabularies(
            examples, vocab_size=vocab_size, min_word_frequency=min_word_frequency, directory=directory)
    elif vocabularies is None:
        raise AttributeError('Vocabulary should not be None with the %s dataset' % corpus_type)

    process_features(examples, corpus_type=corpus_type, vocabularies=vocabularies)

    dataset = QgDataset(examples)
    logger.info('Save %s dataset into %s' % (corpus_type, cached_dataset_path))
    torch.save(dataset, cached_dataset_path)
    return dataset, vocabularies


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Paragraph-level QG')
    parser.add_argument('--dataset', '-dataset', type=str, default='newsqa_v2',
                        choices=['squad_split_v1', 'squad_split_v2', 'newsqa_v2', 'test'])
    parser.add_argument('--data_dir', '-data_dir', type=str, default='data')
                        # choices=['data/squad_split_v1', 'data/squad_split_v2', 'data/newsqa'])

    parser.add_argument('--squad_train_path', '-squad_train_path',
                        type=str, default='data/train-v1.1.json')
    parser.add_argument('--squad_dev_path', '-squad_dev_path',
                        type=str, default='data/dev-v1.1.json')

    parser.add_argument('--cached_squad_train_path', '-cached_squad_train_path',
                        type=str, default='data/squad.train.meta')
    parser.add_argument('--cached_squad_dev_path', '-cached_squad_dev_path',
                        type=str, default='data/squad.dev.meta')

    args = parser.parse_args()

    data_directory = os.path.join(args.data_dir, args.dataset)

    if 'squad' in data_directory or 'test' in data_directory:
        squad_train_path = args.squad_train_path
        squad_dev_path = args.squad_dev_path

        cached_squad_train_path = args.cached_squad_train_path
        cached_squad_dev_path = args.cached_squad_dev_path

        train_qas_id_dict = read_squad_qas_dict(file_path=squad_train_path, save_path=cached_squad_train_path, recover=True)
        dev_qas_id_dict = read_squad_qas_dict(file_path=squad_dev_path, save_path=cached_squad_dev_path, recover=True)
        # squad_qas_id_dict = {**train_qas_id_dict, **dev_qas_id_dict}
        squad_qas_id_dict = {'train': train_qas_id_dict, 'dev':dev_qas_id_dict}

    elif 'newsqa' in data_directory:
        squad_qas_id_dict = None
    else:
        raise NotImplementedError('Dataset from %s is not implemented.' % data_directory)

    _, vocabularies = setup_dataset(directory=data_directory, corpus_type='train', qas_id_dict=squad_qas_id_dict,
                  vocabularies=None, vocab_size=20000, min_word_frequency=3, recover=False)

    setup_dataset(directory=data_directory, corpus_type='dev', qas_id_dict=squad_qas_id_dict,
                  vocabularies=vocabularies, recover=False)

    setup_dataset(directory=data_directory, corpus_type='test', qas_id_dict=squad_qas_id_dict,
                  vocabularies=vocabularies, recover=False)