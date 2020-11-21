import os
from tqdm import tqdm

from datasets.common import Example, get_answer_tag, parse_text_with_stanza


def read_squad_examples_without_ids(corpus_type, qas_id_dict):
    if corpus_type == 'train':
        this_qas_dict = list(qas_id_dict['train'].values())
    elif corpus_type == 'dev':
        this_qas_dict = list(qas_id_dict['dev'].values())
        num_dev = len(this_qas_dict) // 2
        this_qas_dict = this_qas_dict[:num_dev]
    else:
        this_qas_dict = list(qas_id_dict['dev'].values())
        num_dev = len(this_qas_dict) // 2
        this_qas_dict = this_qas_dict[num_dev:]

    qas_list = this_qas_dict

    unique_id = 0
    examples = []
    with tqdm(total=len(qas_list), desc='Reading %s examples' % corpus_type) as t:
        for qas in qas_list:

            meta_data = qas

            examples.append(Example(unique_id=unique_id, meta_data=meta_data))
            unique_id += 1

            t.update()

    t.close()
    return examples