import os
import io
import numpy as np
import torch
from collections import Counter
from tqdm import tqdm
from torchtext.vocab import Vocab as TorchtextVocab
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

from torch.utils.data import Dataset, DataLoader, RandomSampler
from allennlp.predictors.predictor import Predictor
from allennlp_models.structured_prediction.predictors import biaffine_dependency_parser
import allennlp_models.coref
import allennlp_models.tagging
import allennlp_models.structured_prediction

from allennlp.common.util import get_spacy_model
import spacy.tokens.token

# import stanza
# stanza_nlp = stanza.Pipeline('en', logging_level='WARN')
#
# from stanza.server import CoreNLPClient
# client = CoreNLPClient(
#     annotators=['tokenize','ssplit','pos','lemma','ner', 'parse', 'depparse','coref'],
#     timeout=30000,
#     memory='16G'
#

from utils import init_logger
logger = init_logger()

spacy_model = get_spacy_model(spacy_model_name='en_core_web_sm', pos_tags=True, parse=True, ner=True)

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

def get_meta_data_from_stanza(source, answer, target):
    pass

def get_knowledge_from_allennlp(src_lines):
    dependency_instances = []
    constituency_instances = []
    coreference_instances = []
    ex_sentence_indexes = []

    source_tokens = []

    for src_line in src_lines:
        sentences = spacy_model(src_line).sents
        sentences_tokens = [[token for token in sentence] for sentence in sentences]
        num_sentences = len(sentences_tokens)

        source_tokens.append([token for sentence in sentences_tokens for token in sentence])

        if len(ex_sentence_indexes) == 0:
            ex_sentence_indexes += [[0, num_sentences]]
        else:
            last_idx = ex_sentence_indexes[-1][1]
            ex_sentence_indexes += [[last_idx, last_idx + num_sentences]]

        documents_text = []
        for sentence in sentences_tokens:
            sentence_text = [token.text for token in sentence]
            documents_text += [sentence_text]
            pos_tags = [token.tag_ for token in sentence]

            dependency_instances.append(dependency_parser._dataset_reader.text_to_instance(sentence_text, pos_tags))
            constituency_instances.append(
                constituency_parser._dataset_reader.text_to_instance(sentence_text, pos_tags))

        coreference_instances.append(coreference_parser._dataset_reader.text_to_instance(documents_text))


    dependency_results = dependency_parser.predict_batch_instance(dependency_instances)
    dependency_results = [dependency_results[sent_idx[0]: sent_idx[1]] for sent_idx in ex_sentence_indexes]

    constituency_results = constituency_parser.predict_batch_instance(constituency_instances)
    constituency_results = [constituency_results[sent_idx[0]: sent_idx[1]] for sent_idx in ex_sentence_indexes]

    coreference_results = coreference_parser.predict_batch_instance(coreference_instances)

    return [{'source_token': source_token,
             'dependency': dependency, 'constituency': constituency, 'coreference': coreference}
            for source_token, dependency, constituency, coreference in
            zip(source_tokens, dependency_results, constituency_results, coreference_results)]


def setup_dataset(path, extensions, vocabularies, train,
                  cached_dataset_path=None, cached_vocabularies_path=None, recover=True,
                  vocab_size=100000, min_word_frequency=1, max_source_length=200, max_target_length=50):
    TAG = 'train' if train else 'valid'
    # path = os.path.join(path, 'sample') #TODO: magic operation
    path = os.path.join(path, TAG)
    if recover and os.path.exists(cached_dataset_path) and os.path.exists(cached_vocabularies_path):
        logger.info("Loading %s dataset from %s" % (TAG, cached_dataset_path))
        dataset = torch.load(cached_dataset_path)
        if TAG == 'train':
            logger.info("Loading vocabularies from %s" % cached_vocabularies_path)
            vocabularies = torch.load(cached_vocabularies_path)
        return dataset, vocabularies

    src_path, ans_path, trg_path = tuple(os.path.expanduser(path + x) for x in extensions)

    examples = []

    src_lines = open(src_path, encoding='utf-8').readlines()
    ans_lines = io.open(ans_path, mode='r', encoding='utf-8').readlines()
    trg_lines = io.open(trg_path, mode='r', encoding='utf-8').readlines()
    unique_id = 0

    batch_data = []
    batch_size = 10

    with tqdm(total=len(list(src_lines)) / batch_size, desc='Processing %s examples' % TAG) as t:
        for ex_id, (src_line, ans_line, trg_line) in enumerate(zip(src_lines, ans_lines, trg_lines)):
            src_line, ans_line, trg_line = src_line.strip(), ans_line.strip(), trg_line.strip()

            num_src_words = len(src_line.split(' '))
            num_trg_words = len(trg_line.split(' '))

            if num_src_words <= max_source_length and num_trg_words - 1 <= max_target_length:
                if num_src_words * num_trg_words > 0 and num_src_words >= 10:    # The filtering is same as SG-DQG

                    batch_data.append([src_line, ans_line, trg_line])

                    if len(batch_data) == batch_size or ex_id == len(examples) - 1:

                        linguistic_results = get_knowledge_from_allennlp([bd[0] for bd in batch_data])
                        # meta_data = get_knowledge_from_allennlp(src_line, ans_line, trg_line)

                        for (src_line, ans_line, trg_line), linguistic_result in zip(batch_data, linguistic_results):

                            src_words = []
                            src_pos_tags = []
                            src_ner_tags = []
                            src_cas_tags = []

                            for token in linguistic_result['source_token']:
                                src_words += [token.text.lower().strip(' ')]
                                src_pos_tags += [token.tag_]
                                src_ner_tags += [token.ent_type_]
                                src_cas_tags += ['UP' if not token.is_lower else 'DOWN']

                            num_context_words = len(src_words)

                            coreference_mask = np.zeros((num_context_words, num_context_words), dtype=np.int64)
                            for group in linguistic_result['coreference']['clusters']:
                                indexes = [idx for span in group for idx in span]
                                for idx_i in indexes:
                                    for idx_j in indexes:
                                        if idx_i != idx_j:
                                            coreference_mask[idx_i, idx_j] = 1
                            # coreference_mask_spatial = csr_matrix(coreference_mask)

                            words_counter = 0

                            dependency_mask = np.zeros((num_context_words, num_context_words), dtype=np.int64)
                            constituency_mask = np.zeros((num_context_words, num_context_words), dtype=np.int64)

                            for dep_res, const_res in zip(linguistic_result['dependency'],
                                                          linguistic_result['constituency']):

                                for idx, head in enumerate(dep_res['predicted_heads']):
                                    dependency_mask[words_counter + idx, head - 1 + words_counter] = 1
                                    dependency_mask[words_counter + head - 1, idx + words_counter] = 1

                                for (start_ix, end_ix) in const_res['spans']:
                                    if start_ix != end_ix:
                                        constituency_mask[words_counter + start_ix, words_counter + end_ix] = 1

                                words_counter += len(dep_res['words'])

                            assert num_context_words == words_counter

                            dependency_mask = dependency_mask[:num_context_words, :num_context_words]
                            # dependency_mask_spatial = csr_matrix(dependency_mask)
                            constituency_mask = constituency_mask[:num_context_words, :num_context_words]
                            # constituency_mask_spatial = csr_matrix(constituency_mask)

                            # knowledge_mask = {}
                            # same_as_relation = []
                            # coreference_relation = []
                            # dependency_relation = []
                            # constituency_relation = []
                            # knowledge_relation = []

                            # 2. Answer
                            ans_words = [token.text.lower().strip(' ') for sentence in spacy_model(ans_line).sents for token
                                         in sentence]
                            # ans_words = [word.text for sentence in spacy_model(ans_line).sents for word in sentence.words]
                            ans_tag = ['O'] * len(src_words)
                            for i in range(len(src_words) - len(ans_words) + 1):
                                if src_words[i: i + len(ans_words)] == ans_words:
                                    ans_tag[i] = 'B'
                                    ans_tag[i + 1: i + len(ans_words)] = ['I'] * (len(ans_words) - 1)

                            # 3. Target: question.
                            trg_words = [token.text.lower().strip(' ') for sentence in spacy_model(trg_line).sents for token
                                         in sentence]


                            dependency_hop_distance = shortest_path(csgraph=dependency_mask, directed=False)
                            constituency_hop_distance = shortest_path(csgraph=constituency_mask, directed=False)
                            coreference_hop_distance = shortest_path(csgraph=coreference_mask, directed=False)

                            dep_and_con_hop_distance = shortest_path(
                                csgraph=csr_matrix(dependency_mask | constituency_mask), directed=False)
                            dep_and_cor_hop_distance = shortest_path(
                                csgraph=csr_matrix(dependency_mask | coreference_mask), directed=False)
                            con_and_cor_hop_distance = shortest_path(
                                csgraph=csr_matrix(constituency_mask | coreference_mask), directed=False)

                            all_hop_distance = shortest_path(
                                csgraph=csr_matrix(dependency_mask | constituency_mask | coreference_mask), directed=False)

                            meta_data = {'src': src_words, 'pos': src_pos_tags, 'ner': src_ner_tags, 'cas': src_cas_tags,
                                         'dependency_mask_spatial': csr_matrix(dependency_mask),
                                         'constituency_mask_spatial': csr_matrix(constituency_mask),
                                         'coreference_mask_spatial': csr_matrix(coreference_mask),
                                         'dependency_hop_distance': csr_matrix(dependency_hop_distance),
                                         'constituency_hop_distance': csr_matrix(constituency_hop_distance),
                                         'coreference_hop_distance': csr_matrix(coreference_hop_distance),
                                         'dep_and_con_hop_distance': csr_matrix(dep_and_con_hop_distance),
                                         'dep_and_cor_hop_distance': csr_matrix(dep_and_cor_hop_distance),
                                         'con_and_cor_hop_distance': csr_matrix(con_and_cor_hop_distance),
                                         'all_hop_distance': csr_matrix(all_hop_distance),
                                         'ans': ans_tag,
                                         'trg': trg_words}

                            # return meta_data

                            examples.append(Example(unique_id=unique_id, meta_data=meta_data))
                            unique_id += 1

                        batch_data = []
                        t.update()
    t.close()

    if train and vocabularies is None:
        logger.info("Setup vocabularies...")
        logger.info('Setup token vocabulary: Source and Target shares vocabulary.')
        train_src = [ex.meta_data['src'] for ex in examples]
        train_trg = [ex.meta_data['trg'] for ex in examples]
        specials = [UNK_TOKEN, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN]
        corpus = train_src + train_trg
        token_vocab = setup_vocab(corpus, specials=specials, max_size=vocab_size, min_freq=min_word_frequency)

        logger.info('Setup answer tagging vocabulary: (BIO)')
        answer_corpus = ['B', 'I', 'O']
        answer_vocab = setup_vocab(answer_corpus, specials=[UNK_TOKEN, PAD_TOKEN])

        vocabularies = {'token': token_vocab, 'answer': answer_vocab}
        logger.info('Save vocabularies into %s' % cached_vocabularies_path)
        torch.save(vocabularies, cached_vocabularies_path)

    logger.info("Filtering %s data size from %d to %d" % (TAG, len(src_lines), len(examples)))
    with tqdm(total=len(examples), desc='Processing %s features' % TAG) as t:
        for example in examples:

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

            example.src_ids, example.src_extended_ids, example.oov_lst = context2ids(example.meta_data['src'],
                                                                                 vocabularies['token'].stoi)

            example.trg_ids, example.trg_extended_ids = question2ids(example.meta_data['trg'],
                                                                     vocabularies['token'].stoi,
                                                                     example.oov_lst)

            example.ans_tag_ids = [vocabularies['answer'].stoi[token] for token in example.meta_data['ans']]

            t.update()
    t.close()

    dataset = QgDataset(examples)
    logger.info('Save %s dataset into %s' % (TAG, cached_dataset_path))
    torch.save(dataset, cached_dataset_path)
    return dataset, vocabularies


def setup_iterator(dataset, collate_fn, batch_size):
    # collate_fn = functools.partial(collate_custom,
    #                                src_padding_idx=src_padding_idx,
    #                                tgt_padding_idx=tgt_padding_idx,
    #                                device=device,
    #                                is_test=is_test)

    sampler = RandomSampler(dataset)
    iterator = DataLoader(dataset=dataset, batch_size=batch_size,
                          collate_fn=collate_fn, sampler=sampler)
    return iterator

if __name__ == '__main__':
    dependency_parser_url = '~/.allennlp/biaffine-dependency-parser-ptb-2020.04.06.tar.gz'
    logger.info('*** Loading dependency parser from %s' % dependency_parser_url)
    dependency_parser = Predictor.from_path(dependency_parser_url, cuda_device=0)

    constituency_parser_url = '~/.allennlp/elmo-constituency-parser-2020.02.10.tar.gz'
    logger.info('*** Loading constituency parser from %s' % constituency_parser_url)
    constituency_parser = Predictor.from_path(constituency_parser_url, cuda_device=0)

    coreference_parser_url = '~/.allennlp/coref-spanbert-large-2020.02.27.tar.gz'
    logger.info('*** Loading coreference parser from %s' % coreference_parser_url)
    coreference_parser = Predictor.from_path(coreference_parser_url, cuda_device=0)

    data_dir = 'datasets/text-data'
    cached_train_dataset = 'datasets/hotpotqa.train.pth'
    cached_valid_dataset =  'datasets/hotpotqa.valid.pth'
    cached_vocabularies = 'datasets/hotpotqa.vocab.pth'

    extensions = ('.src.txt', '.ans.txt', '.tgt.txt')

    train_dataset, vocabularies = setup_dataset(
        path=data_dir, extensions=extensions, vocabularies=None, train=True,
        cached_dataset_path=cached_train_dataset, cached_vocabularies_path=cached_vocabularies, recover=True,
        vocab_size=50000, min_word_frequency=3, max_source_length=200, max_target_length=50)
    valid_dataset, _ = setup_dataset(
        data_dir, extensions, vocabularies=vocabularies, train=False,
        cached_dataset_path=cached_valid_dataset, cached_vocabularies_path=cached_vocabularies, recover=True,
        max_source_length=200, max_target_length=50)
    # TODO. truncate examples in valid dataset with length > max_source_length.