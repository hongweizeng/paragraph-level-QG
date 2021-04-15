import os
import json
import random
from tqdm import tqdm
import torch
import stanza
stanza_nlp = stanza.Pipeline('en', logging_level='WARN', processors='tokenize,mwt,pos,lemma,depparse,ner')


from utils.logging import logger
from datasets.common import Example, get_answer_tag, parse_text_with_stanza


def read_hotpotqa_examples(corpus_type):

    if corpus_type == 'train':
        src_path, ans_path, trg_path = tuple(os.path.expanduser('data/hotpotqa/train' + x) for x in ['.src.txt', '.ans.txt', '.tgt.txt'])
    else:
        src_path, ans_path, trg_path = tuple(os.path.expanduser('data/hotpotqa/valid' + x) for x in ['.src.txt', '.ans.txt', '.tgt.txt'])

    src_lines = open(src_path, encoding='utf-8').readlines()
    ans_lines = open(ans_path, mode='r', encoding='utf-8').readlines()
    trg_lines = open(trg_path, mode='r', encoding='utf-8').readlines()

    examples = []
    unique_id = 0

    with tqdm(total=len(src_lines), desc='Reading file from para=%s, ans=%s, ques=%s' % (src_path, ans_path, trg_path)) as t:
        for ex_id, (src_line, ans_line, trg_line) in enumerate(zip(src_lines, ans_lines, trg_lines)):
            paragraph_text = src_line.strip().replace("''", '" ').replace("``", '" ')
            paragraph_stanza = stanza_nlp(paragraph_text)

            parsed_paragraph, paragraph_sentences = parse_text_with_stanza(paragraph_stanza.sentences)

            question_text = trg_line.strip().replace("''", '" ').replace("``", '" ')
            question_stanza = stanza_nlp(question_text)
            parsed_question, _ = parse_text_with_stanza(question_stanza.sentences)

            answer_text = ans_line.strip().replace("''", '" ').replace("``", '" ')
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
            t.update()

    t.close()

    # logger.info('Save meta file %s into meta_data %s' % save_path)
    # torch.save(qas_dict, save_path)
    return examples
