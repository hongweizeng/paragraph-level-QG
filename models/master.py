from __future__ import division

from easydict import EasyDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack

from torch_scatter import scatter_max, scatter_sum
from torch.nn.init import xavier_uniform_

from utils import logger, freeze_module
from searcher import Hypothesis, sort_hypotheses
from models.modules.stacked_rnn import StackedGRU
from models.modules.concat_attention import ConcatAttention
from models.modules.maxout import MaxOut

INF = 1e12

class Embeddings(nn.Module):
    """Construct the embeddings from word, pos_tag, ner_tag, dep_tag, cas_tag and answer_tag embeddings.
    """

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=config.pad_token_id)
        self.feature_tag_embeddings = nn.Embedding(config.feature_tag_vocab_size, config.feature_tag_embedding_size)
        self.answer_tag_embeddings = nn.Embedding(config.answer_tag_vocab_size, config.answer_tag_embedding_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file

        # concatenated_embedding_size = config.embedding_size + config.feature_tag_embedding_size * config.feature_num\
        #                             + config.answer_tag_embedding_size

        # self.LayerNorm = nn.LayerNorm(config.embedding_size, eps=config.layer_norm_eps)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids=None, feature_tag_ids_dict=None, answer_tag_ids=None):
        embeddings = self.word_embeddings(input_ids)
        # embeddings = self.LayerNorm(embeddings)

        if feature_tag_ids_dict is not None:

            for feature_tag_ids in feature_tag_ids_dict.values():
                feature_tag_embeddings = self.feature_tag_embeddings(feature_tag_ids)

                embeddings = torch.cat([embeddings, feature_tag_embeddings], dim=2)

        if answer_tag_ids is not None:
            answer_tag_embeddings = self.answer_tag_embeddings(answer_tag_ids)

            embeddings = torch.cat([embeddings, answer_tag_embeddings], dim=2)

        # embeddings = inputs_embeds + feature_tag_embeddings + answer_tag_embeddings
        # embeddings = self.dropout(embeddings)
        return embeddings


class BilinearSeqAttn(nn.Module):
    """A bilinear attention layer over a sequence X w.r.t y:
    * o_i = softmax(x_i'Wy) for x_i in X.
    Optionally don't normalize output weights.
    """

    def __init__(self, x_size, y_size, identity=False, normalize=True):
        super(BilinearSeqAttn, self).__init__()
        self.normalize = normalize
        self.ysize = y_size
        self.xsize = x_size
        # If identity is true, we just use a dot product without transformation.
        if not identity:
            self.linear = nn.Linear(y_size, x_size)
        else:
            self.linear = None

    def forward(self, x, y, x_mask=None):
        """
        Args:
            x: batch * len * hdim1
            y: batch * len_sc * hdim2
            x_mask: batch * len (1 for padding, 0 for true)
        Output:
            alpha = batch * len
            xWy = batch * len_sc * len
        """

        batch_size = y.size(0)
        ly = y.size(1)
        y = y.view(-1, self.ysize)
        Wy = self.linear(y) if self.linear is not None else y
        Wy = Wy.view(batch_size, ly, self.ysize)
        Wy = Wy.permute(0, 2, 1)
        xWy = x.bmm(Wy)
        xWy = xWy.permute(0, 2, 1)
        if x_mask is not None:
            # xWy.data.masked_fill_(x_mask.data.unsqueeze(1).repeat(1, ly, 1), -float('inf'))
            x_mask = x_mask.unsqueeze(1).repeat(1, ly, 1)
            xWy = xWy * (1 - x_mask) + x_mask * (-100000)
        alpha = F.softmax(xWy, dim=-1)  # batch * len_sc * len
        return alpha


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        rnn_input_size = config.embedding_size + config.feature_tag_embedding_size * config.feature_num + \
                          config.answer_tag_embedding_size

        self.num_directions = 2 if config.brnn else 1
        assert config.enc_rnn_size % self.num_directions == 0
        self.hidden_size = config.enc_rnn_size
        rnn_hidden_size = self.hidden_size // self.num_directions
        # self.hidden_size = config.enc_rnn_size // self.num_directions
        # rnn_hidden_size = self.hidden_size
        self.rnn = nn.GRU(rnn_input_size, rnn_hidden_size,
                          num_layers=config.enc_num_layers,
                          dropout=config.dropout,
                          bidirectional=config.brnn, batch_first=True)

        self.wf =  nn.Linear(2 * self.hidden_size, self.hidden_size, bias=False)
        self.wg = nn.Linear(2 * self.hidden_size, self.hidden_size, bias=False)
        self.attn = BilinearSeqAttn(self.hidden_size, self.hidden_size)



    def gated_self_attn(self, queries, memories, mask):
        # queries: [b,t,d]
        # memories: [b,t,d]
        # mask: [b,t]
        energies = torch.matmul(queries, memories.transpose(1, 2))  # [b, t, t]
        mask = mask.unsqueeze(1)
        energies = energies.masked_fill(mask == 0, value=-1e12)

        scores = F.softmax(energies, dim=2)
        context = torch.matmul(scores, queries)
        inputs = torch.cat([queries, context], dim=2)
        f_t = torch.tanh(self.update_layer(inputs))
        g_t = torch.sigmoid(self.gate(inputs))
        updated_output = g_t * f_t + (1 - g_t) * queries

        return updated_output

    def forward(self, paragraph_embedded, paragraph_mask, evidences_embedded=None, evidences_mask=None):

        paragraph_lengths = paragraph_mask.sum(1).tolist()
        paragraph_packed = pack_padded_sequence(paragraph_embedded, paragraph_lengths,
                                                batch_first=True, enforce_sorted=False)
        paragraph_outputs, paragraph_state = self.rnn(paragraph_packed)
        paragraph_outputs, _ = pad_packed_sequence(paragraph_outputs, batch_first=True,
                                                   total_length=paragraph_mask.size(1))

        evidences_lengths = evidences_mask.sum(1).tolist()
        evidences_packed = pack_padded_sequence(evidences_embedded, evidences_lengths,
                                                batch_first=True, enforce_sorted=False)
        evidences_outputs, hidden_t = self.rnn(evidences_packed)
        evidences_outputs, _ = pad_packed_sequence(evidences_outputs, batch_first=True,
                                                   total_length=evidences_mask.size(1))

        # paragraph_outputs = paragraph_outputs.permute(1, 0, 2).contiguous()
        # evidences_outputs = evidences_outputs.permute(1, 0, 2).contiguous()

        batch_size = paragraph_outputs.size(0)
        # T = outputs2.size(1)  # context sentence length (word level)
        J = paragraph_outputs.size(1)  # source sentence length   (word level)
        # para_pad_mask = Variable(parainput[0].eq(s2s.Constants.PAD).float(), requires_grad=False,
        #                          volatile=False).transpose(0, 1)

        # para_pad_mask = Variable(input[0].eq(s2s.Constants.PAD).float(), requires_grad=False,
        #                          volatile=False).transpose(0, 1)

        # this_paragraph_mask = 1.0 - paragraph_mask.long()
        # scores = self.attn(paragraph_outputs, evidences_outputs, this_paragraph_mask)  # batch * len_sc * len_para
        this_evidences_mask = 1.0 - evidences_mask.long()
        scores = self.attn(evidences_outputs, paragraph_outputs, this_evidences_mask)  # batch * len_sc * len_para
        # context = scores.unsqueeze(1).bmm(source_hiddens).squeeze(1)

        # shape = (batch_size, T, J, self.hidden_size)  # (N, T, J, 2d)
        # embd_context = outputs2.unsqueeze(2)  # (N, T, 1, 2d)
        # embd_context = embd_context.expand(shape)  # (N, T, J, 2d)
        # embd_source = outputs.unsqueeze(1)  # (N, 1, J, 2d)
        # embd_source = embd_source.expand(shape)  # (N, T, J, 2d)
        # a_elmwise_mul_b = torch.mul(embd_context, embd_source)  # (N, T, J, 2d)
        # # cat_data = torch.cat((embd_context_ex, embd_source_ex, a_elmwise_mul_b), 3) # (N, T, J, 6d), [h;u;h◦u]
        # S = self.W(torch.cat((embd_context, embd_source, a_elmwise_mul_b), 3)).view(batch_size, T, J) # (N, T, J)
        #
        # para_pad_mask = para_pad_mask.unsqueeze(2).repeat(1, 1, J)
        # S = S*(1-para_pad_mask) + para_pad_mask*(-1000000)
        # self_att = F.softmax(S, dim=-2).permute(0, 2, 1)
        #
        q2c = torch.bmm(scores, evidences_outputs)  # (N, J, 2d) = bmm( (N, J, T), (N, T, 2d) )
        # emb2 = pack(torch.cat((q2c.permute(1, 0, 2), outputs.permute(1, 0, 2)), dim=-1), lengths)
        # outputs_f, hidden_t_f = self.rnn2(emb2, hidden)
        # if isinstance(input, tuple):
        #     outputs_f = unpack(outputs_f)[0]

        f_sc = torch.tanh(self.wf(torch.cat((paragraph_outputs, q2c), dim=-1).view(-1, self.hidden_size * 2)))
        g_sc = torch.sigmoid(self.wg(torch.cat((paragraph_outputs, q2c), dim=-1).view(-1, self.hidden_size * 2)))
        x = g_sc * f_sc + (1 - g_sc) * (paragraph_outputs.view(-1, self.hidden_size))
        # x = x.view(batch_size, J, 2 * self.hidden_size)
        x = x.view(batch_size, J, self.hidden_size)


        _, b, d = paragraph_state.size()
        h = paragraph_state.view(2, 2, b, d)  # [n_layers, bi, b, d]
        h = torch.cat((h[:, 0, :, :], h[:, 1, :, :]), dim=-1)

        return x, h


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vocab_size = config.vocab_size

        hidden_size = config.dec_rnn_size

        self.encoder_trans = nn.Linear(hidden_size, hidden_size)
        self.reduce_layer = nn.Linear(
            config.embedding_size + hidden_size, config.embedding_size)
        self.lstm = nn.LSTM(config.embedding_size, hidden_size, batch_first=True,
                            num_layers=config.dec_num_layers, bidirectional=False, dropout=config.dropout)
        self.concat_layer = nn.Linear(2 * hidden_size, hidden_size)
        self.logit_layer = nn.Linear(hidden_size, config.vocab_size)

        self.use_pointer = config.use_pointer
        self.UNK_ID = config.unk_token_id

    @staticmethod
    def attention(query, memories, mask):
        # query : [b, 1, d]
        energy = torch.matmul(query, memories.transpose(1, 2))  # [b, 1, t]
        energy = energy.squeeze(1).masked_fill(mask == 0, value=-1e12)
        attn_dist = F.softmax(energy, dim=1).unsqueeze(dim=1)  # [b, 1, t]
        context_vector = torch.matmul(attn_dist, memories)  # [b, 1, d]

        return context_vector, energy

    def get_encoder_features(self, encoder_outputs):
        return self.encoder_trans(encoder_outputs)

    def forward(self, trg_seq_embedded, ext_src_seq, init_states, encoder_outputs, encoder_mask):
        # trg_seq : [b,t]
        # init_states : [2,b,d]
        # encoder_outputs : [b,t,d]
        # init_states : a tuple of [2, b, d]
        # device = trg_seq.device
        # batch_size, max_len = trg_seq.size()
        device = trg_seq_embedded.device
        batch_size, max_len, _ = trg_seq_embedded.size()

        hidden_size = encoder_outputs.size(-1)
        memories = self.get_encoder_features(encoder_outputs)
        logits = []
        # init decoder hidden states and context vector
        prev_states = init_states
        prev_context = torch.zeros((batch_size, 1, hidden_size))
        prev_context = prev_context.to(device)

        coverage_sum = None
        coverage_output = []
        attention_output = []

        for i in range(max_len):
            # y_i = trg_seq[:, i].unsqueeze(1)  # [b, 1]
            # embedded = self.embedding(y_i)  # [b, 1, d]
            embedded = trg_seq_embedded[:, i, :].unsqueeze(1)  # [b, 1, d]
            lstm_inputs = self.reduce_layer(
                torch.cat([embedded, prev_context], 2))
            output, states = self.lstm(lstm_inputs, prev_states)
            # encoder-decoder attention
            context, energy = self.attention(output, memories, encoder_mask)
            concat_input = torch.cat((output, context), dim=2).squeeze(dim=1)
            logit_input = torch.tanh(self.concat_layer(concat_input))
            logit = self.logit_layer(logit_input)  # [b, |V|]

            # maxout pointer network
            if self.use_pointer:
                num_oov = max(torch.max(ext_src_seq - self.vocab_size + 1), 0)
                # zeros = torch.zeros((batch_size, num_oov)).type(dtype=logit.dtype)      #TODO:
                zeros = logit.data.new_zeros(size=(batch_size, num_oov))
                extended_logit = torch.cat([logit, zeros], dim=1)
                out = torch.zeros_like(extended_logit) - INF
                out, _ = scatter_max(energy, ext_src_seq, out=out)      #TODO: scatter_sum.
                out = out.masked_fill(out == -INF, 0)
                logit = extended_logit + out
                logit = logit.masked_fill(logit == 0, -INF)

            logits.append(logit)
            # update prev state and context
            prev_states = states
            prev_context = context

            if coverage_sum is None:
                # coverage_sum = context.data.new(encoder_outputs.size(0), encoder_outputs.size(1)).zero_().clone().detach()
                coverage_sum = context.data.new_zeros(size=(encoder_outputs.size(0), encoder_outputs.size(1)))

            # avg_tmp_coverage = coverage_sum / (i + 1)
            # coverage_loss = torch.sum(torch.min(energy, avg_tmp_coverage), dim=1)
            coverage_output.append(coverage_sum.data.clone())
            attn_dist = F.softmax(energy, dim=1)  # [b, t]
            attention_output.append(attn_dist)

            coverage_sum += attn_dist

        logits = torch.stack(logits, dim=1)  # [b, t, |V|]

        return logits, attention_output, coverage_output

    def decode(self, embedded_y, ext_x, prev_states, prev_context, encoder_features, encoder_mask):
        # forward one step lstm
        # y : [b]
        # embedded = self.embedding(y.unsqueeze(1))
        embedded = embedded_y.unsqueeze(1)
        lstm_inputs = self.reduce_layer(torch.cat([embedded, prev_context], 2))
        output, states = self.lstm(lstm_inputs, prev_states)

        context, energy = self.attention(output,
                                         encoder_features,
                                         encoder_mask)
        concat_input = torch.cat((output, context), 2).squeeze(1)
        logit_input = torch.tanh(self.concat_layer(concat_input))
        logit = self.logit_layer(logit_input)  # [b, |V|]

        if self.use_pointer:
            # batch_size = y.size(0)
            batch_size = embedded_y.size(0)
            num_oov = max(torch.max(ext_x - self.vocab_size + 1), 0)
            # zeros = torch.zeros((batch_size, num_oov)).type(dtype=logit.dtype)
            zeros = logit.data.new_zeros(size=(batch_size, num_oov))
            extended_logit = torch.cat([logit, zeros], dim=1)
            out = torch.zeros_like(extended_logit) - INF
            out, _ = scatter_max(energy, ext_x, out=out)
            out = out.masked_fill(out == -INF, 0)
            logit = extended_logit + out
            logit = logit.masked_fill(logit == -INF, 0)
            # forcing UNK prob 0
            logit[:, self.UNK_ID] = -INF

        return logit, states, context


class DecoderV2(nn.Module):
    def __init__(self, config):
        super(DecoderV2, self).__init__()

        self.vocab_size = config.vocab_size

        self.layers = config.dec_num_layers
        self.input_feed = config.input_feed
        input_size = config.embedding_size
        if self.input_feed:
            input_size += config.enc_rnn_size


        # num_directions = 2 if config.brnn else 1
        # self.enc_dec_transformer = nn.Linear(config.enc_rnn_size // num_directions, config.dec_rnn_size)
        self.enc_dec_transformer = nn.Linear(config.enc_rnn_size, config.dec_rnn_size)

        self.rnn = StackedGRU(config.dec_num_layers, input_size, config.dec_rnn_size, config.dropout)

        self.attn = ConcatAttention(config.enc_rnn_size, config.dec_rnn_size, config.ctx_attn_size)

        self.dropout = nn.Dropout(config.dropout)

        self.readout = nn.Linear((config.enc_rnn_size + config.dec_rnn_size + config.embedding_size), config.dec_rnn_size)
        self.maxout = MaxOut(config.maxout_pool_size)
        self.maxout_pool_size = config.maxout_pool_size

        self.copySwitch_l1 = nn.Linear(config.enc_rnn_size + config.dec_rnn_size, 1)
        # self.copySwitch2 = nn.Linear(opt.ctx_rnn_size + opt.dec_rnn_size, 1)
        self.hidden_size = config.dec_rnn_size
        # self.cover = []

        self.logit_layer = nn.Linear(config.dec_rnn_size // config.maxout_pool_size, config.vocab_size)

        self.use_pointer = config.use_pointer
        self.UNK_ID = config.unk_token_id

    def init_rnn_hidden(self, enc_hidden):
        # enc_hidden = torch.cat([enc_states[0], enc_states[-1]], dim=-1)
        dec_hidden = self.enc_dec_transformer(enc_hidden)
        return dec_hidden

    def forward(self, trg_seq_embedded, ext_src_seq, enc_states, encoder_outputs, encoder_mask):

        device = trg_seq_embedded.device
        batch_size, max_len, _ = trg_seq_embedded.size()

        hidden_size = encoder_outputs.size(-1)
        # memories = self.get_encoder_features(encoder_outputs)
        memories = encoder_outputs

        logits = []

        # init decoder hidden states and context vector
        pre_hidden = self.init_rnn_hidden(enc_states)
        pre_context = torch.zeros((batch_size, hidden_size))
        pre_context = pre_context.to(device)
        pre_compute = None
        self.attn.applyMask(encoder_mask)

        coverage_output = []
        attention_output = []
        coverage = memories.data.new_zeros(size=(encoder_outputs.size(0), encoder_outputs.size(1)))

        for i in range(max_len):
            # Embedding
            embedded = trg_seq_embedded[:, i, :]  # [b, d]
            input_emb = embedded
            if self.input_feed:
                input_emb = torch.cat([embedded, pre_context], 1)

            # Decoder
            output, hidden = self.rnn(input_emb, pre_hidden)

            # Encoder-Decoder Attention
            context, attn_dist, pre_compute, energy = self.attn(output, memories, coverage, pre_compute)

            # Maxout
            readout = self.readout(torch.cat((embedded, output, context), dim=1))
            maxout = self.maxout(readout)
            output = self.dropout(maxout)
            logit = self.logit_layer(output)  # [b, |V|]

            # maxout pointer network
            if self.use_pointer:
                num_oov = max(torch.max(ext_src_seq - self.vocab_size + 1), 0)
                # zeros = torch.zeros((batch_size, num_oov)).type(dtype=logit.dtype)      #TODO:
                zeros = logit.data.new_zeros(size=(batch_size, num_oov))
                extended_logit = torch.cat([logit, zeros], dim=1)
                out = torch.zeros_like(extended_logit) - INF
                out, _ = scatter_max(energy, ext_src_seq, out=out)      #TODO: scatter_sum.
                out = out.masked_fill(out == -INF, 0)
                logit = extended_logit + out
                logit = logit.masked_fill(logit == 0, -INF)

            logits.append(logit)

            pre_context = context
            pre_hidden = hidden


            coverage_output.append(coverage.data.clone())
            attention_output.append(attn_dist)

            coverage = coverage + attn_dist

        logits = torch.stack(logits, dim=1)  # [b, t, |V|]

        return logits, attention_output, coverage_output




class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embeddings = Embeddings(config)
        self.encoder = Encoder(config)
        # self.decoder = Decoder(config)
        self.decoder = DecoderV2(config)

    def run_enc_embeddings(self, batch_data):
        paragraph_feature_tag_ids_dict = {
            'ner': batch_data.paragraph_ner_tag_ids,
            'pos': batch_data.paragraph_pos_tag_ids,
            # 'dep': batch_data.paragraph_dep_tag_ids,
            # 'cas': batch_data.paragraph_cas_tag_ids,
        }

        paragraph_mask = (batch_data.paragraph_ids != batch_data.pad_token_id)
        paragraph_embeddings = self.embeddings(input_ids=batch_data.paragraph_ids,
                                               feature_tag_ids_dict=paragraph_feature_tag_ids_dict,
                                               answer_tag_ids=batch_data.paragraph_ans_tag_ids)
        evidences_feature_tag_ids_dict = {
            'ner': batch_data.evidences_ner_tag_ids,
            'pos': batch_data.evidences_pos_tag_ids,
            # 'dep': batch_data.paragraph_dep_tag_ids,
            # 'cas': batch_data.paragraph_cas_tag_ids,
        }
        evidences_mask = (batch_data.evidences_ids != batch_data.pad_token_id)
        evidences_embeddings = self.embeddings(input_ids=batch_data.evidences_ids,
                                               feature_tag_ids_dict=evidences_feature_tag_ids_dict,
                                               answer_tag_ids=batch_data.evidences_ans_tag_ids)
        return paragraph_embeddings, paragraph_mask, evidences_embeddings, evidences_mask

    def forward(self, batch_data):
        """
        Arguments:
            attention_mask: torch.Tensor with 1 indicating tokens to ATTEND to
        """

        paragraph_embeddings, paragraph_mask, evidences_embeddings, evidences_mask = self.run_enc_embeddings(batch_data)

        enc_outputs, enc_states = self.encoder(paragraph_embeddings, paragraph_mask,
                                               evidences_embeddings, evidences_mask)
        sos_trg = batch_data.question_ids[:, :-1].contiguous()

        embedding_output_for_decoder = self.embeddings(input_ids=sos_trg)
        logits, attention_output, coverage_output = self.decoder(
            embedding_output_for_decoder, batch_data.paragraph_extended_ids, enc_states, enc_outputs, paragraph_mask)

        model_output = {
            'logits': logits,
            'attentions': attention_output,
            'coverages': coverage_output
        }

        return model_output


    def beam_search(self, batch_data, beam_size,
                    tok2idx, TRG_SOS_INDEX, TRG_UNK_INDEX, TRG_EOS_INDEX,
                    min_decode_step, max_decode_step, device):

        paragraph_embeddings, paragraph_mask, evidences_embeddings, evidences_mask = self.run_enc_embeddings(batch_data)
        attention_mask = paragraph_mask

        enc_outputs, enc_states = self.encoder(paragraph_embeddings, paragraph_mask,
                                               evidences_embeddings, evidences_mask)

        prev_context = torch.zeros(1, 1, enc_outputs.size(-1)).cuda(device=device)

        h, c = enc_states  # [2, b, d] but b = 1
        hypotheses = [Hypothesis(tokens=[TRG_SOS_INDEX],
                                 log_probs=[0.0],
                                 state=(h[:, 0, :], c[:, 0, :]),
                                 context=prev_context[0]) for _ in range(beam_size)]
        # tile enc_outputs, enc_mask for beam search
        ext_src_seq = batch_data.paragraph_extended_ids.repeat(beam_size, 1)
        enc_outputs = enc_outputs.repeat(beam_size, 1, 1)
        enc_features = self.decoder.get_encoder_features(enc_outputs)
        enc_mask = attention_mask.repeat(beam_size, 1)
        num_steps = 0
        results = []
        while num_steps < max_decode_step and len(results) < beam_size:
            latest_tokens = [h.latest_token for h in hypotheses]
            latest_tokens = [idx if idx < len(
                tok2idx) else TRG_UNK_INDEX for idx in latest_tokens]
            prev_y = torch.tensor(latest_tokens, dtype=torch.long, device=device).view(-1)

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

            embedded_prev_y = self.embeddings(prev_y)
            logits, states, context_vector = self.decoder.decode(embedded_prev_y, ext_src_seq,
                                                                       prev_states, prev_context,
                                                                       enc_features, enc_mask)
            h_state, c_state = states
            log_probs = F.log_softmax(logits, dim=1)
            top_k_log_probs, top_k_ids \
                = torch.topk(log_probs, beam_size * 2, dim=-1)

            all_hypotheses = []
            num_orig_hypotheses = 1 if num_steps == 0 else len(hypotheses)
            for i in range(num_orig_hypotheses):
                h = hypotheses[i]
                state_i = (h_state[:, i, :], c_state[:, i, :])
                context_i = context_vector[i]
                for j in range(beam_size * 2):

                    score = top_k_log_probs[i][j].item()


                    vocab_entropy = torch.sum(-1 * log_probs[i] * torch.log(log_probs[i]), 1) / torch.log(len(tok2idx))
                    copy_entropy = torch.sum(-1 * context_i * torch.log(context_i), 1) / torch.log(paragraph_embeddings.size(1))

                    new_h = h.extend(token=top_k_ids[i][j].item(),
                                     log_prob=top_k_log_probs[i][j].item(),
                                     state=state_i,
                                     context=context_i)
                    all_hypotheses.append(new_h)

            hypotheses = []
            for h in sort_hypotheses(all_hypotheses):
                if h.latest_token == TRG_EOS_INDEX:
                    if num_steps >= min_decode_step:
                        results.append(h)
                else:
                    hypotheses.append(h)

                if len(hypotheses) == beam_size or len(results) == beam_size:
                    break
            num_steps += 1
        if len(results) == 0:
            results = hypotheses
        h_sorted = sort_hypotheses(results)

        return h_sorted[0]


def collate_function(examples, pad_token_id=1, device=None):
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
