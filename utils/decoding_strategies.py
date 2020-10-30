import torch
import numpy as np

class DecodingStrategies(object):
    def __init__(self, model, memory_bank, memory_mask, memory_proj,
                 batch_size, max_len, device):
        self.model = model

        self.memory_bank = memory_bank
        self.memory_mask = memory_mask
        self.memory_proj = memory_proj

        self.max_len = max_len
        self.is_finished = torch.zeros([batch_size], dtype=torch.bool, device=device)
        self.device = device

    def step(self, previous_y, hidden_state):
        raise NotImplementedError


class GreedyDecoding(DecodingStrategies):
    def __init__(self, model, memory_bank, memory_mask, memory_proj,
                 batch_size, max_len,
                 sos_index=2, eos_index=3, device=None):
        self.batched_sequences = torch.full([batch_size, 1], sos_index, dtype=torch.long, device=device)

        self.sos_index = sos_index
        self.eos_index = eos_index

        super(GreedyDecoding, self).__init__(
            model, memory_bank, memory_mask, memory_proj, batch_size, max_len, device)

    def run(self, hidden_state):

        attention_scores = []
        for i in range(self.max_len):
            previous_y = self.batched_sequences[:, -1:]

            probabilities, hidden_state = self.model.step(
                previous_y, hidden_state, self.memory_bank, self.memory_mask, self.memory_proj)

            _, next_word = torch.max(probabilities, dim=1, keepdim=True)

            self.batched_sequences = torch.cat((self.batched_sequences, next_word), dim=-1)

            attention_scores.append(self.model.decoder.attention.alphas.cpu().numpy())

            self.is_finished |= next_word.eq(self.eos_index).squeeze(1)
            if self.is_finished.all():
                break

        output = self.batched_sequences[:, 1:]

        return output, np.concatenate(attention_scores, axis=1)


class BeamDecoding(DecodingStrategies):
    def step(self, previous_y, hidden_state):
        pass
