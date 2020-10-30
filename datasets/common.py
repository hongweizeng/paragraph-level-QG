from typing import Dict
import random
from collections import Counter
from tqdm import tqdm
from torchtext.vocab import Vocab as TorchtextVocab
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

import stanza
stanza_nlp = stanza.Pipeline('en', logging_level='WARN', processors='tokenize,mwt,pos,lemma,depparse,ner')


UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"


class Example(object):
    def __init__(self, unique_id, meta_data):
        self.unique_id = unique_id
        self.meta_data = meta_data


class QgDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples
        self.num_examples = len(examples)

    def __getitem__(self, item):
        return self.examples[item]

    def __iter__(self):
        for example in self.examples:
            yield example

    def __len__(self):
        if self.num_examples is not None:
            return self.num_examples
        return len(self.examples)


def setup_vocab(iterator, **kwargs):
    counter = Counter()
    with tqdm(unit_scale=0, unit='lines') as t:
        for tokens in iterator:
            counter.update(tokens)
            t.update(1)
    word_vocab = TorchtextVocab(counter, **kwargs)
    return word_vocab


def setup_iterator(dataset, collate_fn, batch_size, random=True):
    if random:
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)
    iterator = DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=collate_fn, sampler=sampler)
    return iterator


def context2ids(tokens, word2idx):
    ids = list()
    extended_ids = list()
    oov_lst = list()
    # START and END token is already in tokens lst
    for token in tokens:
        if token in word2idx:
            ids.append(word2idx[token])
            extended_ids.append(word2idx[token])
        else:
            ids.append(word2idx[UNK_TOKEN])
            if token not in oov_lst:
                oov_lst.append(token)
            extended_ids.append(len(word2idx) + oov_lst.index(token))

    return ids, extended_ids, oov_lst


def question2ids(tokens, word2idx, oov_lst):
    ids = list()
    extended_ids = list()
    ids.append(word2idx[SOS_TOKEN])
    extended_ids.append(word2idx[SOS_TOKEN])

    for token in tokens:
        if token in word2idx:
            ids.append(word2idx[token])
            extended_ids.append(word2idx[token])
        else:
            ids.append(word2idx[UNK_TOKEN])
            if token in oov_lst:
                extended_ids.append(len(word2idx) + oov_lst.index(token))
            else:
                extended_ids.append(word2idx[UNK_TOKEN])
    ids.append(word2idx[EOS_TOKEN])
    extended_ids.append(word2idx[EOS_TOKEN])

    return ids, extended_ids


def parse_text_with_stanza(sentences):
    parsed_result = {
        'tokens': [],
        'ner_tags': [],
        'pos_tags': [],
        'dep_tags': [],
        'cas_tags': []}

    raw_sentences = []
    for sentence in sentences:
        raw_sentences.append(sentence)
        for word, token in zip(sentence.words, sentence.tokens):
            parsed_result['ner_tags'].append(token.ner)

            parsed_result['tokens'].append(word.text.lower())
            parsed_result['pos_tags'].append(word.pos)
            parsed_result['dep_tags'].append(word.deprel)
            parsed_result['cas_tags'].append('DOWN' if word.text.islower() else 'UP')

    return parsed_result, raw_sentences


def get_answer_tag(source_tokens, target_tokens):
    answer_tag = ['O'] * len(source_tokens)

    source_tokens_length = len(source_tokens)
    target_tokens_length = len(target_tokens)
    for i in range(source_tokens_length - target_tokens_length + 1):
        if source_tokens[i: i + target_tokens_length] == target_tokens:
            answer_tag[i] = 'B'
            answer_tag[i + 1: i + target_tokens_length] = ['I'] * (target_tokens_length - 1)

    return answer_tag



# def build_vocabularies(examples, vocab_size, min_word_frequency, directory):
#     logger.info("Setup vocabularies...")
#     logger.info('Setup token vocabulary: Source and Target shares vocabulary.')
#     train_src = [ex.meta_data['paragraph']['tokens'] for ex in examples]
#     train_trg = [ex.meta_data['question']['tokens'] for ex in examples]
#     specials = [UNK_TOKEN, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN]
#     corpus = train_src + train_trg
#     token_vocab = setup_vocab(corpus, specials=specials, max_size=vocab_size, min_freq=min_word_frequency)
#
#     logger.info('Setup answer tagging vocabulary: (BIO)')
#     answer_corpus = ['B', 'I', 'O']
#     answer_vocab = setup_vocab(answer_corpus, specials=[UNK_TOKEN, PAD_TOKEN])
#
#     logger.info('Setup feature vocabulary: pos, ner, dep, cas')
#     train_ner = [ex.meta_data['paragraph']['ner_tags'] for ex in examples]
#     train_pos = [ex.meta_data['paragraph']['pos_tags'] for ex in examples]
#     train_dep = [ex.meta_data['paragraph']['dep_tags'] for ex in examples]
#     train_cas = [ex.meta_data['paragraph']['cas_tags'] for ex in examples]
#     feature_corpus = train_ner + train_pos + train_dep + train_cas
#     feature_vocab = setup_vocab(feature_corpus, specials=[UNK_TOKEN, PAD_TOKEN])
#
#     vocabularies = {'token': token_vocab, 'answer': answer_vocab, 'feature': feature_vocab}
#     cached_vocabularies_path = os.path.join(directory, 'vocab.pt')
#     logger.info('Save vocabularies into %s' % cached_vocabularies_path)
#     torch.save(vocabularies, cached_vocabularies_path)
#     return vocabularies
#
#
# def process_features(examples, corpus_type, vocabularies):
#     with tqdm(total=len(examples), desc='Processing %s features' % corpus_type) as t:
#         for example in examples:
#             # Paragraph Input
#             example.paragraph_ids, example.paragraph_extended_ids, example.oov_lst = context2ids(
#                 example.meta_data['paragraph']['tokens'], vocabularies['token'].stoi)
#             example.paragraph_ner_tag_ids = [vocabularies['feature'].stoi[tag]
#                                              for tag in example.meta_data['paragraph']['ner_tags']]
#             example.paragraph_pos_tag_ids = [vocabularies['feature'].stoi[tag]
#                                              for tag in example.meta_data['paragraph']['pos_tags']]
#             example.paragraph_dep_tag_ids = [vocabularies['feature'].stoi[tag]
#                                              for tag in example.meta_data['paragraph']['dep_tags']]
#             example.paragraph_cas_tag_ids = [vocabularies['feature'].stoi[tag]
#                                              for tag in example.meta_data['paragraph']['cas_tags']]
#             example.paragraph_ans_tag_ids = [vocabularies['answer'].stoi[tag]
#                                              for tag in example.meta_data['paragraph_ans_tag']]
#
#             # Evidences Input
#             example.evidences_ids, _, _ = context2ids(
#                 example.meta_data['evidences']['tokens'], vocabularies['token'].stoi)
#             example.evidences_ner_tag_ids = [vocabularies['feature'].stoi[tag]
#                                              for tag in example.meta_data['evidences']['ner_tags']]
#             example.evidences_pos_tag_ids = [vocabularies['feature'].stoi[tag]
#                                              for tag in example.meta_data['evidences']['pos_tags']]
#             example.evidences_dep_tag_ids = [vocabularies['feature'].stoi[tag]
#                                              for tag in example.meta_data['evidences']['dep_tags']]
#             example.evidences_cas_tag_ids = [vocabularies['feature'].stoi[tag]
#                                              for tag in example.meta_data['evidences']['cas_tags']]
#             example.evidences_ans_tag_ids = [vocabularies['answer'].stoi[tag]
#                                              for tag in example.meta_data['evidences_ans_tag']]
#
#             # Question Input
#             example.question_ids, example.question_extended_ids = question2ids(
#                 example.meta_data['question']['tokens'], vocabularies['token'].stoi, example.oov_lst)
#
#             # Answer Input
#             example.answer_ids, _, _ = context2ids(
#                 example.meta_data['answer']['tokens'], vocabularies['token'].stoi)
#             example.answer_ner_tag_ids = [vocabularies['feature'].stoi[tag]
#                                              for tag in example.meta_data['answer']['ner_tags']]
#             example.answer_pos_tag_ids = [vocabularies['feature'].stoi[tag]
#                                              for tag in example.meta_data['answer']['pos_tags']]
#             example.answer_dep_tag_ids = [vocabularies['feature'].stoi[tag]
#                                              for tag in example.meta_data['answer']['dep_tags']]
#             example.answer_cas_tag_ids = [vocabularies['feature'].stoi[tag]
#                                              for tag in example.meta_data['answer']['cas_tags']]
#
#             t.update()
#     t.close()
