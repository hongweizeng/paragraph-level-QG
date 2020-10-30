import os
import json
import random
from tqdm import tqdm
import torch
import stanza
stanza_nlp = stanza.Pipeline('en', logging_level='WARN', processors='tokenize,mwt,pos,lemma,depparse,ner')


from utils.logging import logger
from datasets.common import setup_vocab, QgDataset, Example, UNK_TOKEN, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, \
    get_answer_tag, context2ids, question2ids, parse_text_with_stanza


def read_squad_qas_dict(file_path, save_path, recover=True):
    if recover and os.path.exists(save_path):
        logger.info('Load meta_data from %s' % save_path)
        qas_dict = torch.load(save_path)
        return qas_dict

    qas_dict = {}
    with open(file_path, mode='r', encoding='utf-8') as json_reader:
        articles  = json.load(json_reader)['data']

    with tqdm(total=len(articles), desc='Reading file %s' % file_path) as t:
        for article in articles:
            for para in article["paragraphs"]:
                paragraph_text = para["context"].replace("''", '" ').replace("``", '" ')
                paragraph_stanza = stanza_nlp(paragraph_text)

                parsed_paragraph, paragraph_sentences = parse_text_with_stanza(paragraph_stanza.sentences)

                for qa in para['qas']:
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

                    qas_id = qa['id']
                    qas_dict[qas_id] = meta_data
    t.close()

    logger.info('Save file %s into meta_data %s' % (file_path, save_path))
    torch.save(qas_dict, save_path)

    return qas_dict


def read_squad_examples(directory, corpus_type, qas_id_dict):
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
