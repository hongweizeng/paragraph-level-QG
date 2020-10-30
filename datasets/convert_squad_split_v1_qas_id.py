import os
import json
import argparse

def convert_to_ids(directory, corpus_type):
    file_path = os.path.join(directory, corpus_type + '.json')
    with open(file_path, mode='r', encoding='utf-8') as json_reader:
        articles = json.load(json_reader)

    save_path = os.path.join(directory, corpus_type + '.txt.id')
    with open(save_path, mode='w', encoding='utf-8') as txt_writer:
        for ex_id, article in enumerate(articles):

            for para in article["paragraphs"]:
                for qa in para["qas"]:
                    qas_id = qa['id']
                    txt_writer.write(qas_id + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SSR')
    parser.add_argument('--data_dir', '-data_dir', type=str, default='../data/squad_split_v1')
    args = parser.parse_args()

    data_directory = args.data_dir

    convert_to_ids(data_directory, corpus_type='train')
    convert_to_ids(data_directory, corpus_type='dev')
    convert_to_ids(data_directory, corpus_type='test')