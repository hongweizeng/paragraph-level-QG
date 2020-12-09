import torch
from utils.logging import init_logger

logger = init_logger()

def stat(file_name_list):
    example_num = 0
    paragraph_length = 0
    evidences_length = 0
    question_length = 0
    answer_length = 0

    for file_name in file_name_list:
        logger.info('Load meta_data from %s' % file_name)
        story_dict = torch.load(file_name)

        for key, value in story_dict.items():
            for example in value:

                paragraph_length += len(example['paragraph']['tokens'])
                evidences_length += len(example['evidences']['tokens'])
                question_length += len(example['question']['tokens'])
                answer_length += len(example['answer']['tokens'])

                example_num += 1

    avg_paragraph_length =  paragraph_length * 1.0 / example_num
    avg_evidences_length = evidences_length * 1.0 / example_num
    avg_question_length = question_length * 1.0 / example_num
    avg_answer_length = answer_length * 1.0 / example_num

    logger.info('Average length: para=%.2f, evid=%.2f, ques=%.2f, answ=%.2f' % (
        avg_paragraph_length, avg_evidences_length, avg_question_length, avg_answer_length))

stat(['data/newsqa.train.meta', 'data/newsqa.dev.meta', 'data/newsqa.test.meta'])

# stat(['data/squad.train.meta_list', 'data/squad.dev.meta_list'])


train_data = torch.load('data/newsqa/train.pt')
dev_data = torch.load('data/newsqa/dev.pt')
test_data = torch.load('data/newsqa/test.pt')

avg_para_length = (sum([len(ex.paragraph_ids) for ex in train_data.examples]) +
                   sum([len(ex.paragraph_ids) for ex in dev_data.examples]) + \
                   sum([len(ex.paragraph_ids) for ex in test_data.examples])) * 1.0 / \
                  (len(train_data.examples) + len(dev_data.examples) + len(test_data.examples))

avg_evid_length = (sum([len(ex.evidences_ids) for ex in train_data.examples]) +
                   sum([len(ex.evidences_ids) for ex in dev_data.examples]) + \
                   sum([len(ex.evidences_ids) for ex in test_data.examples])) * 1.0 / \
                  (len(train_data.examples) + len(dev_data.examples) + len(test_data.examples))

avg_ques_length = (sum([len(ex.question_ids) for ex in train_data.examples]) +
                   sum([len(ex.question_ids) for ex in dev_data.examples]) + \
                   sum([len(ex.question_ids) for ex in test_data.examples])) * 1.0 / \
                  (len(train_data.examples) + len(dev_data.examples) + len(test_data.examples))

avg_answ_length = (sum([len(ex.answer_ids) for ex in train_data.examples]) +
                   sum([len(ex.answer_ids) for ex in dev_data.examples]) + \
                   sum([len(ex.answer_ids) for ex in test_data.examples])) * 1.0 / \
                  (len(train_data.examples) + len(dev_data.examples) + len(test_data.examples))

logger.info('Average length: para=%.2f, evid=%.2f, ques=%.2f, answ=%.2f' % (
    avg_para_length, avg_evid_length, avg_ques_length, avg_answ_length))