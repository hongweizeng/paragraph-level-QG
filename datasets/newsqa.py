import os
import re
import random
from tqdm import tqdm
import pandas as pd
import stanza
stanza_nlp = stanza.Pipeline('en', logging_level='WARN', processors='tokenize,mwt,pos,lemma,depparse,ner')

from utils.logging import logger
from datasets.common import Example, get_answer_tag, parse_text_with_stanza


def read_newsqa_examples(directory, corpus_type, recover=True):

    with open(os.path.join(directory, 'split_data/' + corpus_type + '.csv')) as csv_reader:
        articles = pd.read_csv(csv_reader, sep=',', header=0).values

    unique_id = 0
    examples = []
    with tqdm(total=len(articles), desc='Reading %s dataset in directory %s' % (corpus_type, directory)) as t:
        for article in articles:
            # Paragraph
            story_text = article[1]
            story_tokens = story_text.split()
            answer_token_ranges = [int(t) for t in re.split(',|:', article[3])]
            if answer_token_ranges[0] < 150:
                paragraph_text = " ".join(story_tokens[:300])
            else:
                paragraph_text = " ".join(story_tokens[answer_token_ranges[0]-150:answer_token_ranges[0]+150])
            paragraph_text = paragraph_text.replace("''", '" ').replace("``", '" ')
            paragraph_stanza = stanza_nlp(paragraph_text)
            parsed_paragraph, paragraph_sentences = parse_text_with_stanza(paragraph_stanza.sentences)

            # Question
            question_text = article[2].replace("''", '" ').replace("``", '" ')
            question_stanza = stanza_nlp(question_text)
            parsed_question, _ = parse_text_with_stanza(question_stanza.sentences)

            # Answer
            answer_text = " ".join(story_tokens[answer_token_ranges[0]:answer_token_ranges[-1]])
            answer_stanza = stanza_nlp(answer_text)
            parsed_answer, _ = parse_text_with_stanza(answer_stanza.sentences)
            answer_text = " ".join(parsed_answer['tokens'])

            paragraph_ans_tag = get_answer_tag(parsed_paragraph['tokens'], parsed_answer['tokens'])

            # Evidences
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
    return examples
