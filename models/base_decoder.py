import torch
import torch.nn as nn

from easydict import EasyDict

class DecoderBase(nn.Module):
    def __init__(self, config):
        super(DecoderBase, self).__init__()

        self.attention_network = None
        # self.coverage = config.coverage

    def forward(self, embeddings, memory_bank, memory_state, memory_mask):

        # Initial hidden_state & context
        hidden_state = self.init_hidden_state(memory_state)
        context = 0

        # Forward Loop
        batch_size, max_steps, _ = embeddings.size()

        attn_energy_list = []
        attn_dist_list = []
        logit_list = []

        for step_index in range(max_steps):
            embedding = embeddings[:, step_index, :]  # [b, d]

            hidden_state, context, attn_energy, attn_dist, logit = self.step(
                embedding, hidden_state, context, memory_bank, memory_mask)

            attn_energy_list.append(attn_energy)
            attn_dist_list.append(attn_dist)
            logit_list.append(logit)

        return EasyDict(dict(logits=logit_list, attn_dists=attn_dist_list, attn_energies=attn_energy_list))

    def init_hidden_state(self, memory_state):
        if isinstance(memory_state, tuple):
            # LSTM
            raise NotImplementedError
        else:
            # GRU
            raise NotImplementedError

    def attention(self, query, key_value, key_mask, coverage=None):
        raise NotImplementedError

    def step(self, embedding, hidden_state, context, memory_bank, memory_mask):
        # Enhanced RNN input.
        rnn_input = embedding
        if self.input_feed:
            rnn_input = torch.cat([embedding, context], dim=1)

        # Decoder
        rnn_output, hidden_state = self.rnn(rnn_input, hidden_state)

        # Encoder-Decoder Attention
        context, attn_energy, attn_dist = self.attention(
            query=hidden_state, key_value=memory_bank, key_mask=memory_mask)

        # Maxout
        readout = self.readout(torch.cat((embedding, rnn_output, context), dim=1))
        maxout = self.maxout(readout)
        output = self.dropout(maxout)
        logit = self.logit_layer(output)  # [b, |V|]

        return hidden_state, context, attn_energy, attn_dist, logit