import os
from tqdm import tqdm

from datasets.common import Example, get_answer_tag, parse_text_with_stanza


def read_squad_examples_without_ids(directory, corpus_type, qas_id_dict):
    id_path = os.path.join(directory, corpus_type + '.txt.id')
    with open(id_path, mode='r', encoding='utf-8') as txt_reader:
        qas_ids = txt_reader.readlines()

    unique_id = 0
    examples = []
    with tqdm(total=len(qas_ids), desc='Reading %s examples' % corpus_type) as t:
        for qas_id in qas_ids:
            qas_id = qas_id.strip()

            meta_data = qas_id_dict[qas_id]

            examples.append(Example(unique_id=unique_id, meta_data=meta_data))
            unique_id += 1

            t.update()

    t.close()
    return examples