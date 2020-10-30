import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from easydict import EasyDict

# import maxout_pointer_gated_self_attention.config as config
# from maxout_pointer_gated_self_attention.data_utils import outputids2words, UNK_TOKEN, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN
from data import UNK_TOKEN, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN

from utils import logger, count_params, Statistics


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


class Hypothesis(object):
    def __init__(self, tokens, log_probs, state, context=None):
        self.tokens = tokens
        self.log_probs = log_probs
        self.state = state
        self.context = context

    def extend(self, token, log_prob, state, context=None):
        h = Hypothesis(tokens=self.tokens + [token],
                       log_probs=self.log_probs + [log_prob],
                       state=state,
                       context=context)
        return h

    @property
    def latest_token(self):
        return self.tokens[-1]

    @property
    def avg_log_prob(self):
        return sum(self.log_probs) / len(self.tokens)


class Searcher(object):
    def __init__(self, vocabularies, data_loader, model, output_dir, device, config:EasyDict):
        self.config = config
        self.beam_size = config['beam_size']
        self.min_decode_step = config['min_decode_step']
        self.max_decode_step = config['max_decode_step']

        self.output_dir = output_dir

        self.vocabularies = vocabularies
        self.tok2idx = vocabularies['token'].stoi
        self.idx2tok = {idx: tok for idx, tok in enumerate(vocabularies['token'].itos)}

        self.PAD_INDEX = vocabularies['token'].stoi[PAD_TOKEN]
        self.TRG_UNK_INDEX = vocabularies['token'].stoi[UNK_TOKEN]
        self.TRG_SOS_INDEX = vocabularies['token'].stoi[SOS_TOKEN]
        self.TRG_EOS_INDEX = vocabularies['token'].stoi[EOS_TOKEN]
        self.TRG_VOCAB_SIZE = len(vocabularies['token'])

        self.data_loader = data_loader
        self.test_data = data_loader.dataset

        self.model = model
        self.device = device

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
        logger.info(' * Dataset size: valid = %d' % len(self.data_loader))
        logger.info(' * Output directory: %s' % self.output_dir)

        references = []
        hypothesis = []

        pred_fw = open(self.pred_dir, "w")
        golden_fw = open(self.golden_dir, "w")
        t = tqdm(total=len(self.data_loader), unit='q', desc='Writing Questions:')
        for i, batch_data in enumerate(self.data_loader):
            best_question = self.search_batch(batch_data)
            # discard START  token
            output_indices = [int(idx) for idx in best_question.tokens[1:-1]]
            decoded_words = outputids2words(
                output_indices, self.idx2tok, batch_data.oov_lst[0], UNK_ID=self.TRG_UNK_INDEX)
            try:
                fst_stop_idx = decoded_words.index(self.TRG_EOS_INDEX)
                decoded_words = decoded_words[:fst_stop_idx]
            except ValueError:
                decoded_words = decoded_words

            golden_question = self.test_data[i].meta_data['trg']
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

        return references, hypothesis

    def search_batch(self, batch_data):
        raise NotImplementedError


class BeamSearcher(Searcher):

    @staticmethod
    def sort_hypotheses(hypotheses):
        return sorted(hypotheses, key=lambda h: h.avg_log_prob, reverse=True)


    def search_batch(self, batch_data):
        src_seq, ext_src_seq, tag_seq = batch_data.src_seq, batch_data.ext_src_seq, batch_data.tag_seq
        src_padding_idx = self.PAD_INDEX

        enc_mask = (src_seq != src_padding_idx)

        enc_outputs, enc_states = self.model.encoder(batch_data)

        prev_context = torch.zeros(1, 1, enc_outputs.size(-1)).to(dtype=next(self.parameters()).dtype)  # fp16 compatibility.cuda(device=self.device)

        h, c = enc_states  # [2, b, d] but b = 1
        hypotheses = [Hypothesis(tokens=[self.tok2idx[SOS_TOKEN]],
                                 log_probs=[0.0],
                                 state=(h[:, 0, :], c[:, 0, :]),
                                 context=prev_context[0]) for _ in range(self.beam_size)]
        # tile enc_outputs, enc_mask for beam search
        ext_src_seq = ext_src_seq.repeat(self.beam_size, 1)
        enc_outputs = enc_outputs.repeat(self.beam_size, 1, 1)
        enc_features = self.model.decoder.get_encoder_features(enc_outputs)
        enc_mask = enc_mask.repeat(self.beam_size, 1)
        num_steps = 0
        results = []
        while num_steps < self.config.max_decode_step and len(results) < self.beam_size:
            latest_tokens = [h.latest_token for h in hypotheses]
            latest_tokens = [idx if idx < len(
                self.tok2idx) else self.TRG_UNK_INDEX for idx in latest_tokens]
            prev_y = torch.tensor(latest_tokens, dtype=torch.long).view(-1).to(dtype=next(self.parameters()).dtype)  # fp16 compatibility, device=self.device)

            # if config.use_gpu:
            #     prev_y = prev_y.to(self.device)

            # make batch of which size is beam size
            all_state_h = []
            all_state_c = []
            all_context = []
            for h in hypotheses:
                state_h, state_c = h.state  # [num_layers, d]
                all_state_h.append(state_h)
                all_state_c.append(state_c)
                all_context.append(h.context)

            prev_h = torch.stack(all_state_h, dim=1)  # [num_layers, beam, d]
            prev_c = torch.stack(all_state_c, dim=1)  # [num_layers, beam, d]
            prev_context = torch.stack(all_context, dim=0)
            prev_states = (prev_h, prev_c)
            # [beam_size, |V|]
            logits, states, context_vector = self.model.decoder.decode(prev_y, ext_src_seq,
                                                                       prev_states, prev_context,
                                                                       enc_features, enc_mask)
            h_state, c_state = states
            log_probs = F.log_softmax(logits, dim=1)
            top_k_log_probs, top_k_ids \
                = torch.topk(log_probs, self.beam_size * 2, dim=-1)

            all_hypotheses = []
            num_orig_hypotheses = 1 if num_steps == 0 else len(hypotheses)
            for i in range(num_orig_hypotheses):
                h = hypotheses[i]
                state_i = (h_state[:, i, :], c_state[:, i, :])
                context_i = context_vector[i]
                for j in range(self.beam_size * 2):
                    new_h = h.extend(token=top_k_ids[i][j].item(),
                                     log_prob=top_k_log_probs[i][j].item(),
                                     state=state_i,
                                     context=context_i)
                    all_hypotheses.append(new_h)

            hypotheses = []
            for h in self.sort_hypotheses(all_hypotheses):
                if h.latest_token == self.TRG_EOS_INDEX:
                    if num_steps >= self.config.min_decode_step:
                        results.append(h)
                else:
                    hypotheses.append(h)

                if len(hypotheses) == self.beam_size or len(results) == self.beam_size:
                    break
            num_steps += 1
        if len(results) == 0:
            results = hypotheses
        h_sorted = self.sort_hypotheses(results)

        return h_sorted[0]


class GreedySearcher(Searcher):
    def search_batch(self, batch_data):
        raise NotImplementedError