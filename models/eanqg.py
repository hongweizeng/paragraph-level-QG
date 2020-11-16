import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch_scatter import scatter_max
from torch.nn.init import xavier_uniform_

from utils import logger, freeze_module

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

        self.LayerNorm = nn.LayerNorm(config.embedding_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids=None, feature_tag_ids_dict=None, answer_tag_ids=None):
        embeddings = self.word_embeddings(input_ids)
        embeddings = self.LayerNorm(embeddings)

        if feature_tag_ids_dict is not None:

            for k, v in feature_tag_ids_dict:
                feature_tag_embeddings = self.feature_tag_embeddings(v)

                embeddings = torch.cat([embeddings, feature_tag_embeddings], dim=2)

        if answer_tag_ids is not None:
            answer_tag_embeddings = self.answer_tag_embeddings(answer_tag_ids)

            embeddings = torch.cat([embeddings, answer_tag_embeddings], dim=2)

        # embeddings = inputs_embeds + feature_tag_embeddings + answer_tag_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        lstm_input_size = config.embedding_size + config.feature_tag_embedding_size * config.feature_num + \
                          config.answer_tag_embedding_size

        self.num_layers = config.num_layers
        if self.num_layers == 1:
            dropout = 0.0
        self.lstm = nn.LSTM(lstm_input_size, config.hidden_size, dropout=config.dropout,
                            num_layers=config.num_layers, bidirectional=True, batch_first=True)
        self.linear_trans = nn.Linear(2 * config.hidden_size, 2 * config.hidden_size)
        self.update_layer = nn.Linear(
            4 * config.hidden_size, 2 * config.hidden_size, bias=False)
        self.gate = nn.Linear(4 * config.hidden_size, 2 * config.hidden_size, bias=False)
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

    def forward(self, embedded, enc_mask):
        # total_length = src_seq.size(1)
        total_length = embedded.size(1)
        src_len = enc_mask.sum(1).tolist()

        # embedded = self.embedding(src_seq)
        # tag_embedded = self.tag_embedding(tag_seq)
        # embedded = torch.cat((embedded, tag_embedded), dim=2)
        embedded = embedded
        packed = pack_padded_sequence(embedded,
                                      src_len,
                                      batch_first=True,
                                      enforce_sorted=False)
        outputs, states = self.lstm(packed)  # states : tuple of [4, b, d]
        outputs, _ = pad_packed_sequence(outputs,
                                         batch_first=True,
                                         total_length=total_length)  # [b, t, d]
        h, c = states

        # self attention
        # mask = torch.sign(src_seq)
        mask = enc_mask
        memories = self.linear_trans(outputs)
        outputs = self.gated_self_attn(outputs, memories, mask)

        _, b, d = h.size()
        h = h.view(2, 2, b, d)  # [n_layers, bi, b, d]
        h = torch.cat((h[:, 0, :, :], h[:, 1, :, :]), dim=-1)

        c = c.view(2, 2, b, d)
        c = torch.cat((c[:, 0, :, :], c[:, 1, :, :]), dim=-1)
        concat_states = (h, c)

        return outputs, concat_states


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vocab_size = config.vocab_size

        hidden_size = 2 * config.hidden_size
        if config.num_layers == 1:
            dropout = 0.0
        self.encoder_trans = nn.Linear(hidden_size, hidden_size)
        self.reduce_layer = nn.Linear(
            config.embedding_size + hidden_size, config.embedding_size)
        self.lstm = nn.LSTM(config.embedding_size, hidden_size, batch_first=True,
                            num_layers=config.num_layers, bidirectional=False, dropout=config.dropout)
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
                out, _ = scatter_max(energy, ext_src_seq, out=out)
                out = out.masked_fill(out == -INF, 0)
                logit = extended_logit + out
                logit = logit.masked_fill(logit == 0, -INF)

            logits.append(logit)
            # update prev state and context
            prev_states = states
            prev_context = context

        logits = torch.stack(logits, dim=1)  # [b, t, |V|]

        return logits

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


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embeddings = Embeddings(config)
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def forward(self, batch_data):
        """
        Arguments:
            attention_mask: torch.Tensor with 1 indicating tokens to ATTEND to
        """

        attention_mask = (batch_data.paragraph_ids != batch_data.pad_token_id)

        assert batch_data.paragraph_ids.lt(45004).all(), 'source vocab out'
        assert batch_data.paragraph_ans_tag_ids.lt(5).all(), 'answer vocab out'

        embedding_output_for_encoder = self.embeddings(input_ids=batch_data.paragraph_ids, feature_tag_ids_dict=None,
                                                       answer_tag_ids=batch_data.paragraph_ans_tag_ids)
        enc_outputs, enc_states = self.encoder(embedding_output_for_encoder, attention_mask)
        sos_trg = batch_data.question_ids[:, :-1].contiguous()

        assert batch_data.question_ids.lt(45004).all(), 'target vocab out'
        embedding_output_for_decoder = self.embeddings(input_ids=sos_trg)
        logits = self.decoder(embedding_output_for_decoder, batch_data.paragraph_extended_ids,
                              enc_states, enc_outputs, attention_mask)
        return logits


def setup_model(config, vocabularies, device=None, checkpoint=None):
    # Set up model configuration.
    model = Model(config)

    # Load the model states from checkpoint or initialize them.
    if checkpoint is not None:
        logger.info('Loading checkpoint from %s' % checkpoint)
        model.load_state_dict(torch.load(checkpoint))
    else:
        if config.param_init != 0.0:
            for p in model.parameters():
                p.data.uniform_(-config.param_init, config.param_init)
        if config.param_init_glorot:
            for p in model.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)

        logger.info('Initializing shared source and target embedding with glove.6B.300d.')
        vocabularies['token'].load_vectors('glove.6B.300d')
        model.encoder.embedding.weight.data.copy_(vocabularies['token'].vectors)
        # logger.info('Freezing embeddings.')
        # freeze_module(model.encoder.embedding)

    model.to(device)
    logger.info('Model = %s' % model)
    return model