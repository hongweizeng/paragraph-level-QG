import os
import json
import random
import argparse
from tqdm import tqdm

import torch
import stanza
stanza_nlp = stanza.Pipeline('en', logging_level='WARN', processors='tokenize,mwt,pos,lemma,depparse,ner')

from utils.logging import init_logger
logger = init_logger()

from datasets.common import setup_vocab, QgDataset, Example, UNK_TOKEN, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, \
    get_answer_tag, context2ids, question2ids, parse_text_with_stanza


def read_examples(directory, corpus_type, qas_id_dict):
    id_path = os.path.join(directory, corpus_type + '.txt.id')
    with open(id_path, mode='r', encoding='utf-8') as txt_reader:
        qas_ids = txt_reader.readlines()

    unique_id = 0
    examples = []
    key_error_count = 0
    with tqdm(total=len(qas_ids), desc='Reading %s examples' % corpus_type) as t:
        for qas_id in qas_ids:
            qas_id = qas_id.strip()

            if qas_id in qas_id_dict:
                meta_data = qas_id_dict[qas_id]

                examples.append(Example(unique_id=unique_id, meta_data=meta_data))
                unique_id += 1
            else:
                key_error_count += 1
            t.update()

    t.close()
    logger.info('%s: Final_dataset_size  = original_dataset_size - key_error_size : %d = %d - %d' %
                (corpus_type, len(examples), len(qas_ids), key_error_count))
    return examples






def setup_dataset(directory, corpus_type, vocabularies,
                  vocab_size=100000, min_word_frequency=1, max_source_length=200, max_target_length=50):

    file_path = os.path.join(directory, corpus_type + '.json')
    with open(file_path, mode='r', encoding='utf-8') as json_reader:
        articles = json.load(json_reader)

    unique_id = 0
    examples = []
    with tqdm(total=len(articles), desc='Processing %s examples' % corpus_type) as t:
        for ex_id, article in enumerate(articles):

            for para in article["paragraphs"]:
                paragraph_text = para["context"].replace("''", '" ').replace("``", '" ')
                paragraph_stanza = stanza_nlp(paragraph_text)

                parsed_paragraph, paragraph_sentences = parse_text_with_stanza(paragraph_stanza.sentences)

                for qa in para["qas"]:
                    # total += 1
                    question_text = qa["question"].replace("''", '" ').replace("``", '" ')
                    question_stanza = stanza_nlp(question_text)
                    parsed_question, _ = parse_text_with_stanza(question_stanza.sentences)
                    # question_text = ' '.join(parsed_question['tokens'])

                    answer = qa["answers"][0]
                    answer_text = answer["text"]
                    answer_stanza = stanza_nlp(answer_text)
                    parsed_answer, _ = parse_text_with_stanza(answer_stanza.sentences)
                    answer_text = " ".join(parsed_answer['tokens'])

                    paragraph_ans_tag = get_answer_tag(parsed_paragraph['tokens'], parsed_answer['tokens'])

                    evidences = []
                    for sentence in paragraph_sentences:
                        if answer_text in sentence.text:
                            evidences.append(sentence)
                    if not evidences:
                        evidences = [random.choice(paragraph_sentences)]
                    parsed_evidences, _ = parse_text_with_stanza(evidences)
                    evidences_ans_tag = get_answer_tag(parsed_evidences['tokens'], parsed_answer['tokens'])


                    meta_data = {"paragraph": parsed_paragraph, 'paragraph_ans_tag': paragraph_ans_tag,

                                 "evidences": parsed_evidences, 'evidences_ans_tag': evidences_ans_tag,

                                 "question": parsed_question,

                                 "answer": parsed_answer}

                    examples.append(Example(unique_id=unique_id, meta_data=meta_data))

                    unique_id += 1
            t.update()
    t.close()

    if corpus_type == 'train':
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
    elif vocabularies is None:
        raise AttributeError('Vocabulary should not be None with the %s dataset' % corpus_type)

    # logger.info("Filtering %s data size from %d to %d" % (corpus_type, len(examples), len(processed_examples)))
    with tqdm(total=len(examples), desc='Processing %s features' % corpus_type) as t:
        for example in examples:
            # Paragraph Input
            example.paragraph_ids, example.paragraph_extended_ids, example.oov_lst = context2ids(
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
            example.evidences_ids, _, _ = context2ids(
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
            example.question_ids, example.question_extended_ids = question2ids(
                example.meta_data['question']['tokens'], vocabularies['token'].stoi, example.oov_lst)

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

    dataset = QgDataset(examples)
    cached_dataset_path = os.path.join(directory, corpus_type + '.pt')
    logger.info('Save %s dataset into %s' % (corpus_type, cached_dataset_path))
    torch.save(dataset, cached_dataset_path)
    return dataset, vocabularies


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='SSR')
    parser.add_argument('--dataset', '-dataset', type=str, default='squad_split1')
    args = parser.parse_args()

    data_directory = 'datasets/squad_split_v1'

    _, vocabularies = setup_dataset(directory=data_directory, corpus_type='train', vocabularies=None,
                                    vocab_size=50000, min_word_frequency=3,
                                    max_source_length=200, max_target_length=50)

    setup_dataset(directory=data_directory, corpus_type='dev', vocabularies=vocabularies,
                  max_source_length=200, max_target_length=50)

    setup_dataset(directory=data_directory, corpus_type='test', vocabularies=vocabularies,
                  max_source_length=200, max_target_length=50)
