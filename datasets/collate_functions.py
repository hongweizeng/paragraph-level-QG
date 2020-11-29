import torch
from easydict import EasyDict


def master_collate_function(examples, pad_token_id=1, device=None):
    batch_size = len(examples)

    examples.sort(key=lambda x: (len(x.paragraph_ids), len(x.question_ids)), reverse=True)
    meta_data = [ex.meta_data for ex in examples]
    paragraph_oov_lst = [ex.paragraph_oov_lst for ex in examples]
    evidences_oov_lst = [ex.evidences_oov_lst for ex in examples]

    max_src_len = max(len(ex.paragraph_ids) for ex in examples)
    max_trg_len = max(len(ex.question_ids) for ex in examples)

    paragraph_ids = torch.LongTensor(batch_size, max_src_len).fill_(pad_token_id).cuda(device)
    paragraph_extended_ids = torch.LongTensor(batch_size, max_src_len).fill_(pad_token_id).cuda(device)
    # paragraph_ans_tag_ids = torch.LongTensor(batch_size, max_src_len).cuda(device)
    paragraph_ans_tag_ids = torch.ones([batch_size, max_src_len], dtype=torch.long, device=device)
    paragraph_pos_tag_ids = torch.ones([batch_size, max_src_len], dtype=torch.long, device=device)
    paragraph_ner_tag_ids = torch.ones([batch_size, max_src_len], dtype=torch.long, device=device)
    paragraph_dep_tag_ids = torch.ones([batch_size, max_src_len], dtype=torch.long, device=device)
    paragraph_cas_tag_ids = torch.ones([batch_size, max_src_len], dtype=torch.long, device=device)

    max_evd_len = max(len(ex.evidences_ids) for ex in examples)
    evidences_ids = torch.LongTensor(batch_size, max_evd_len).fill_(pad_token_id).cuda(device)
    # evidences_ans_tag_ids = torch.LongTensor(batch_size, max_src_len).fill_(pad_token_id).cuda(device)
    evidences_ans_tag_ids = torch.ones([batch_size, max_evd_len], dtype=torch.long, device=device)
    evidences_pos_tag_ids = torch.ones([batch_size, max_evd_len], dtype=torch.long, device=device)
    evidences_ner_tag_ids = torch.ones([batch_size, max_evd_len], dtype=torch.long, device=device)
    evidences_dep_tag_ids = torch.ones([batch_size, max_evd_len], dtype=torch.long, device=device)
    evidences_cas_tag_ids = torch.ones([batch_size, max_evd_len], dtype=torch.long, device=device)

    question_ids = torch.LongTensor(batch_size, max_trg_len).fill_(pad_token_id).cuda(device)
    question_extended_ids_para = torch.LongTensor(batch_size, max_trg_len).fill_(pad_token_id).cuda(device)
    question_extended_ids_evid = torch.LongTensor(batch_size, max_trg_len).fill_(pad_token_id).cuda(device)

    for idx, example in enumerate(examples):
        assert len(example.paragraph_ids) == len(example.paragraph_extended_ids) == len(example.paragraph_ans_tag_ids)
        paragraph_ids[idx, :len(example.paragraph_ids)] = torch.tensor(example.paragraph_ids)
        paragraph_extended_ids[idx, :len(example.paragraph_extended_ids)] = torch.tensor(example.paragraph_extended_ids)
        paragraph_ans_tag_ids[idx, :len(example.paragraph_ans_tag_ids)] = torch.tensor(example.paragraph_ans_tag_ids)
        paragraph_pos_tag_ids[idx, :len(example.paragraph_pos_tag_ids)] = torch.tensor(example.paragraph_pos_tag_ids)
        paragraph_ner_tag_ids[idx, :len(example.paragraph_ner_tag_ids)] = torch.tensor(example.paragraph_ner_tag_ids)
        paragraph_dep_tag_ids[idx, :len(example.paragraph_dep_tag_ids)] = torch.tensor(example.paragraph_dep_tag_ids)
        paragraph_cas_tag_ids[idx, :len(example.paragraph_cas_tag_ids)] = torch.tensor(example.paragraph_cas_tag_ids)

        evidences_ids[idx, :len(example.evidences_ids)] = torch.tensor(example.evidences_ids)
        evidences_ans_tag_ids[idx, :len(example.evidences_ans_tag_ids)] = torch.tensor(example.evidences_ans_tag_ids)
        evidences_pos_tag_ids[idx, :len(example.evidences_pos_tag_ids)] = torch.tensor(example.evidences_pos_tag_ids)
        evidences_ner_tag_ids[idx, :len(example.evidences_ner_tag_ids)] = torch.tensor(example.evidences_ner_tag_ids)
        evidences_dep_tag_ids[idx, :len(example.evidences_dep_tag_ids)] = torch.tensor(example.evidences_dep_tag_ids)
        evidences_cas_tag_ids[idx, :len(example.evidences_cas_tag_ids)] = torch.tensor(example.evidences_cas_tag_ids)

        assert len(example.question_ids) == len(example.question_extended_ids_para) == len(example.question_extended_ids_evid)
        question_ids[idx, :len(example.question_ids)] = torch.tensor(example.question_ids)
        question_extended_ids_para[idx, :len(example.question_extended_ids_para)] = torch.tensor(
            example.question_extended_ids_para)
        question_extended_ids_evid[idx, :len(example.question_extended_ids_evid)] = torch.tensor(
            example.question_extended_ids_evid)


    return EasyDict({'paragraph_ids': paragraph_ids, 'paragraph_extended_ids': paragraph_extended_ids,
                     'paragraph_ans_tag_ids': paragraph_ans_tag_ids,
                     'paragraph_pos_tag_ids': paragraph_pos_tag_ids, 'paragraph_ner_tag_ids': paragraph_ner_tag_ids,
                     'paragraph_dep_tag_ids': paragraph_dep_tag_ids, 'paragraph_cas_tag_ids': paragraph_cas_tag_ids,

                     'evidences_ids': evidences_ids,
                     'evidences_ans_tag_ids': evidences_ans_tag_ids,
                     'evidences_pos_tag_ids': evidences_pos_tag_ids, 'evidences_ner_tag_ids': evidences_ner_tag_ids,
                     'evidences_dep_tag_ids': evidences_dep_tag_ids, 'evidences_cas_tag_ids': evidences_cas_tag_ids,

                     'question_ids': question_ids,
                     'question_extended_ids_para': question_extended_ids_para,
                     'question_extended_ids_evid': question_extended_ids_evid,

                     'paragraph_oov_lst': paragraph_oov_lst, 'evidences_oov_lst': evidences_oov_lst,

                     'pad_token_id': pad_token_id,

                     'meta_data': meta_data, 'batch_size': batch_size})


def eanqg_collate_function(examples, pad_token_id=1, device=None):
    batch_size = len(examples)

    examples.sort(key=lambda x: (len(x.paragraph_ids), len(x.question_ids)), reverse=True)
    meta_data = [ex.meta_data for ex in examples]
    paragraph_oov_lst = [ex.paragraph_oov_lst for ex in examples]
    evidences_oov_lst = [ex.evidences_oov_lst for ex in examples]

    max_src_len = max(len(ex.paragraph_ids) for ex in examples)
    max_trg_len = max(len(ex.question_ids) for ex in examples)

    paragraph_ids = torch.LongTensor(batch_size, max_src_len).fill_(pad_token_id).cuda(device)
    paragraph_extended_ids = torch.LongTensor(batch_size, max_src_len).fill_(pad_token_id).cuda(device)
    # paragraph_ans_tag_ids = torch.LongTensor(batch_size, max_src_len).cuda(device)
    paragraph_ans_tag_ids = torch.ones([batch_size, max_src_len], dtype=torch.long, device=device)
    paragraph_pos_tag_ids = torch.ones([batch_size, max_src_len], dtype=torch.long, device=device)
    paragraph_ner_tag_ids = torch.ones([batch_size, max_src_len], dtype=torch.long, device=device)
    paragraph_dep_tag_ids = torch.ones([batch_size, max_src_len], dtype=torch.long, device=device)
    paragraph_cas_tag_ids = torch.ones([batch_size, max_src_len], dtype=torch.long, device=device)

    evidences_ids = torch.LongTensor(batch_size, max_src_len).fill_(pad_token_id).cuda(device)
    # evidences_ans_tag_ids = torch.LongTensor(batch_size, max_src_len).fill_(pad_token_id).cuda(device)
    evidences_ans_tag_ids = torch.ones([batch_size, max_src_len], dtype=torch.long, device=device)
    evidences_pos_tag_ids = torch.ones([batch_size, max_src_len], dtype=torch.long, device=device)
    evidences_ner_tag_ids = torch.ones([batch_size, max_src_len], dtype=torch.long, device=device)
    evidences_dep_tag_ids = torch.ones([batch_size, max_src_len], dtype=torch.long, device=device)
    evidences_cas_tag_ids = torch.ones([batch_size, max_src_len], dtype=torch.long, device=device)

    question_ids = torch.LongTensor(batch_size, max_trg_len).fill_(pad_token_id).cuda(device)
    question_extended_ids_para = torch.LongTensor(batch_size, max_trg_len).fill_(pad_token_id).cuda(device)
    question_extended_ids_evid = torch.LongTensor(batch_size, max_trg_len).fill_(pad_token_id).cuda(device)

    for idx, example in enumerate(examples):
        assert len(example.paragraph_ids) == len(example.paragraph_extended_ids) == len(example.paragraph_ans_tag_ids)
        paragraph_ids[idx, :len(example.paragraph_ids)] = torch.tensor(example.paragraph_ids)
        paragraph_extended_ids[idx, :len(example.paragraph_extended_ids)] = torch.tensor(example.paragraph_extended_ids)
        paragraph_ans_tag_ids[idx, :len(example.paragraph_ans_tag_ids)] = torch.tensor(example.paragraph_ans_tag_ids)
        paragraph_pos_tag_ids[idx, :len(example.paragraph_pos_tag_ids)] = torch.tensor(example.paragraph_pos_tag_ids)
        paragraph_ner_tag_ids[idx, :len(example.paragraph_ner_tag_ids)] = torch.tensor(example.paragraph_ner_tag_ids)
        paragraph_dep_tag_ids[idx, :len(example.paragraph_dep_tag_ids)] = torch.tensor(example.paragraph_dep_tag_ids)
        paragraph_cas_tag_ids[idx, :len(example.paragraph_cas_tag_ids)] = torch.tensor(example.paragraph_cas_tag_ids)

        evidences_ids[idx, :len(example.evidences_ids)] = torch.tensor(example.evidences_ids)
        evidences_ans_tag_ids[idx, :len(example.evidences_ans_tag_ids)] = torch.tensor(example.evidences_ans_tag_ids)
        evidences_pos_tag_ids[idx, :len(example.evidences_pos_tag_ids)] = torch.tensor(example.evidences_pos_tag_ids)
        evidences_ner_tag_ids[idx, :len(example.evidences_ner_tag_ids)] = torch.tensor(example.evidences_ner_tag_ids)
        evidences_dep_tag_ids[idx, :len(example.evidences_dep_tag_ids)] = torch.tensor(example.evidences_dep_tag_ids)
        evidences_cas_tag_ids[idx, :len(example.evidences_cas_tag_ids)] = torch.tensor(example.evidences_cas_tag_ids)

        assert len(example.question_ids) == len(example.question_extended_ids_para) == len(example.question_extended_ids_evid)
        question_ids[idx, :len(example.question_ids)] = torch.tensor(example.question_ids)
        question_extended_ids_para[idx, :len(example.question_extended_ids_para)] = torch.tensor(
            example.question_extended_ids_para)
        question_extended_ids_evid[idx, :len(example.question_extended_ids_evid)] = torch.tensor(
            example.question_extended_ids_evid)


    return EasyDict({'paragraph_ids': paragraph_ids, 'paragraph_extended_ids': paragraph_extended_ids,
                     'paragraph_ans_tag_ids': paragraph_ans_tag_ids,
                     'paragraph_pos_tag_ids': paragraph_pos_tag_ids, 'paragraph_ner_tag_ids': paragraph_ner_tag_ids,
                     'paragraph_dep_tag_ids': paragraph_dep_tag_ids, 'paragraph_cas_tag_ids': paragraph_cas_tag_ids,

                     'evidences_ids': evidences_ids,
                     'evidences_ans_tag_ids': evidences_ans_tag_ids,
                     'evidences_pos_tag_ids': evidences_pos_tag_ids, 'evidences_ner_tag_ids': evidences_ner_tag_ids,
                     'evidences_dep_tag_ids': evidences_dep_tag_ids, 'evidences_cas_tag_ids': evidences_cas_tag_ids,

                     'question_ids': question_ids,
                     'question_extended_ids_para': question_extended_ids_para,
                     'question_extended_ids_evid': question_extended_ids_evid,

                     'paragraph_oov_lst': paragraph_oov_lst, 'evidences_oov_lst': evidences_oov_lst,

                     'pad_token_id': pad_token_id,

                     'meta_data': meta_data, 'batch_size': batch_size})
