from typing import Dict
import random
from collections import Counter
from tqdm import tqdm
from torchtext.vocab import Vocab as TorchtextVocab
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler


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