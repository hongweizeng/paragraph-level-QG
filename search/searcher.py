import os
import math
import torch
import torch.nn.functional as F
import functools
from tqdm import tqdm
from easydict import EasyDict
from torchtext.data.metrics import bleu_score

from utils import logger
from datasets.common import UNK_TOKEN, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, setup_iterator
from datasets.collate_functions import master_collate_function

def outputids2words(id_list, idx2word, article_oovs=None, UNK_ID=0):
    """
    :param id_list: list of indices
    :param idx2word: dictionary mapping idx to word
    :param article_oovs: list of oov words
    :return: list of words
    """

    words = []
    for idx in id_list:
        try:
            word = idx2word[idx]
        except KeyError:
            if article_oovs is not None:
                article_oov_idx = idx - len(idx2word)
                try:
                    word = article_oovs[article_oov_idx]
                except IndexError:
                    print("there's no such a word in extended vocab")
            else:
                word = idx2word[UNK_ID]
        words.append(word)

    return words


def sort_hypotheses(hypotheses):
    return sorted(hypotheses, key=lambda h: h.score, reverse=True)


class Hypothesis(object):
    def __init__(self, tokens, log_probs, state, context=None, coverage=None, uncertainty_scores=None, beta=0.):
        self.tokens = tokens
        self.log_probs = log_probs
        self.state = state
        self.context = context
        self.coverage = coverage
        self.uncertainty_scores = uncertainty_scores

        self.beta = beta

    def extend(self, token, log_prob, state, context=None, coverage=None, uncertainty_scores=None):
        h = Hypothesis(tokens=self.tokens + [token],
                       log_probs=self.log_probs + [log_prob],
                       state=state,
                       context=context,
                       coverage=coverage,
                       uncertainty_scores=self.uncertainty_scores + [uncertainty_scores])
        return h

    @property
    def latest_token(self):
        return self.tokens[-1]

    @property
    def avg_log_prob(self):
        return sum(self.log_probs) / len(self.tokens)

    @property
    def uncertainty_aware_score(self):
        return 0.885 * sum(self.log_probs) / len(self.tokens) - 0.115 * math.log((sum(self.uncertainty_scores) / len(self.tokens)))

    @property
    def length(self):
        return len(self.tokens)

    @property
    def score(self):
        if self.beta == 0.0:
            return sum(self.log_probs) / len(self.tokens)
        else:
            return (1 - self.beta) * sum(self.log_probs) / self.length - self.beta * math.log(sum(self.uncertainty_scores) / self.length)



class Searcher(object):
    def __init__(self, vocabularies, test_dataset, model, output_dir, config):
        self.beam_size = config.beam_size
        self.min_decode_step = config.min_decode_step
        self.max_decode_step = config.max_decode_step

        self.beta = config.beta

        self.output_dir = output_dir

        self.vocabularies = vocabularies
        self.tok2idx = vocabularies['token'].stoi
        self.idx2tok = {idx: tok for idx, tok in enumerate(vocabularies['token'].itos)}

        self.PAD_INDEX = vocabularies['token'].stoi[PAD_TOKEN]
        self.TRG_UNK_INDEX = vocabularies['token'].stoi[UNK_TOKEN]
        self.TRG_SOS_INDEX = vocabularies['token'].stoi[SOS_TOKEN]
        self.TRG_EOS_INDEX = vocabularies['token'].stoi[EOS_TOKEN]
        self.TRG_VOCAB_SIZE = len(vocabularies['token'])

        self.device = next(model.parameters()).device
        collate_fn = functools.partial(master_collate_function, pad_token_id=self.PAD_INDEX, device=self.device)

        self.data_loader = setup_iterator(dataset=test_dataset, collate_fn=collate_fn, batch_size=1, random=False)
        self.test_data = test_dataset

        self.model = model
        # self.device = next(model.parameters()).device

        self.pred_dir = os.path.join(output_dir, "generated.txt")
        self.golden_dir = os.path.join(output_dir, "golden.txt")
        self.src_file = os.path.join(output_dir, "src.txt")

        # if not os.path.exists(output_dir):
        #     os.makedirs(output_dir)
        # dummy file for evaluation
        with open(self.src_file, "w") as f:
            for i in range(len(self.data_loader)):
                f.write(str(i) + "\n")


    def search(self):
        logger.info(' * Dataset size: test = %d' % len(self.data_loader))
        logger.info(' * Output directory: %s' % self.output_dir)

        references = []
        hypothesis = []

        pred_fw = open(self.pred_dir, "w")
        golden_fw = open(self.golden_dir, "w")
        t = tqdm(total=len(self.data_loader), unit='q', desc='Writing Questions:')
        for i, batch_data in enumerate(self.data_loader):
            # best_question = self.search_batch(batch_data) #TODO. reduce replicated codes.
            best_question = self.model.beam_search(
                batch_data=batch_data, beam_size=self.beam_size,
                tok2idx=self.tok2idx,
                TRG_SOS_INDEX=self.TRG_SOS_INDEX, TRG_UNK_INDEX=self.TRG_UNK_INDEX, TRG_EOS_INDEX=self.TRG_EOS_INDEX,
                min_decode_step=self.min_decode_step, max_decode_step=self.max_decode_step, device=self.device)

            # discard START  token
            output_indices = [int(idx) for idx in best_question.tokens[1:-1]]
            decoded_words = outputids2words(
                output_indices, self.idx2tok, batch_data.paragraph_oov_lst[0], UNK_ID=self.TRG_UNK_INDEX)
            try:
                fst_stop_idx = decoded_words.index(self.TRG_EOS_INDEX)
                decoded_words = decoded_words[:fst_stop_idx]
            except ValueError:
                decoded_words = decoded_words

            golden_question = self.test_data[i].meta_data['question']['tokens']
            references.append([golden_question])
            hypothesis.append(decoded_words)

            decoded_words = " ".join(decoded_words)
            golden_question = " ".join(golden_question)
            # print("write {}th question\r".format(i))
            pred_fw.write(decoded_words + "\n")
            golden_fw.write(golden_question + "\n")

            t.update()

        t.close()
        pred_fw.close()
        golden_fw.close()

        from torchtext.data.metrics import bleu_score
        score = bleu_score(hypothesis, references) * 100
        logger.info('Bleu score = %.3f' % score)

        # return references, hypothesis
        return score

    def search_batch(self, batch_data):
        raise NotImplementedError

    def greedy_search(self):
        references = []
        hypothesis = []

        for i, batch_data in enumerate(self.data_loader):
            # best_question = self.search_batch(batch_data) #TODO. reduce replicated codes.
            best_question = self.model.beam_search(
                batch_data=batch_data, beam_size=1,
                tok2idx=self.tok2idx,
                TRG_SOS_INDEX=self.TRG_SOS_INDEX, TRG_UNK_INDEX=self.TRG_UNK_INDEX, TRG_EOS_INDEX=self.TRG_EOS_INDEX,
                min_decode_step=self.min_decode_step, max_decode_step=self.max_decode_step, device=self.device)

            # discard START  token
            output_indices = [int(idx) for idx in best_question.tokens[1:-1]]
            decoded_words = outputids2words(
                output_indices, self.idx2tok, batch_data.paragraph_oov_lst[0], UNK_ID=self.TRG_UNK_INDEX)
            try:
                fst_stop_idx = decoded_words.index(self.TRG_EOS_INDEX)
                decoded_words = decoded_words[:fst_stop_idx]
            except ValueError:
                decoded_words = decoded_words

            golden_question = self.test_data[i].meta_data['question']['tokens']
            references.append([golden_question])
            hypothesis.append(decoded_words)

        score = bleu_score(hypothesis, references) * 100
        return score