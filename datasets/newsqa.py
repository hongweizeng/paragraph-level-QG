import os
import csv
import random
from tqdm import tqdm
import torch
import stanza
stanza_nlp = stanza.Pipeline('en', logging_level='WARN', processors='tokenize,mwt,pos,lemma,depparse,ner')
import pandas as pd
import scipy as sp

from utils.logging import logger
from datasets.common import setup_vocab, QgDataset, Example, UNK_TOKEN, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, \
    get_answer_tag, context2ids, question2ids, parse_text_with_stanza


def read_newsqa_stories_dict(file_path, save_path, recover=True):
    if recover and os.path.exists(save_path):
        logger.info('Load meta_data from %s' % save_path)
        qas_dict = torch.load(save_path)
        return qas_dict

    story_dict = {}
    articles = pd.read_csv(file_path, sep=',', header=0).values

    length_arr = []

    with tqdm(total=len(articles), desc='Reading file %s' % file_path) as t:
        for article in articles:
            story_id = article[0]
            story_text = article[1].replace("''", '" ').replace("``", '" ')

            story_length = len(story_text.split())
            length_arr.append(story_length)


    logger.info('%d' % (story_length / len(articles)))
    t.close()
    logger.info('Save file %s into meta_data %s' % (file_path, save_path))
    torch.save(story_dict, save_path)

    return story_dict


def read_newsqa_examples(directory, corpus_type, recover=True):

    with open(os.path.join(directory, 'split_data/' + corpus_type + '.csv')) as csv_reader:
        articles = pd.read_csv(csv_reader, sep=',', header=0).values

    unique_id = 0
    examples = []
    with tqdm(total=len(articles), desc='Reading %s dataset in directory %s' % (corpus_type, directory)) as t:
        for article in articles:
            paragraph_text = para["context"].replace("''", '" ').replace("``", '" ')
            paragraph_stanza = stanza_nlp(paragraph_text)

            parsed_paragraph, paragraph_sentences = parse_text_with_stanza(paragraph_stanza.sentences)

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
                if answer_text in sentence.text.lower():
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
    t.close()
    return examples
