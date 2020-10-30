# https://github.com/abisee/pointer-generator/blob/master/beam_search.py
from typing import List


class Hypothesis(object):
    """Class to represent a hypothesis during beam search. Holds all the information needed for the hypothesis."""

    def __init__(self, unique_id, indexes, log_probs, hidden_state=None,
                 encoder_hidden=None, src_mask=None, proj_key=None):
        # Decoder states
        self.unique_id = unique_id
        self.indexes = indexes
        self.log_probs = log_probs
        self.hidden_state = hidden_state

        # Encoder states
        self.encoder_hidden = encoder_hidden
        self.src_mask = src_mask
        self.proj_key = proj_key

    def update(self, unique_id, index, log_prob, hidden_state=None):
        return Hypothesis(unique_id=unique_id,
                          indexes=self.indexes + [index],
                          log_probs=self.log_probs + [log_prob],
                          hidden_state=hidden_state,
                          encoder_hidden=self.encoder_hidden,
                          src_mask=self.src_mask,
                          proj_key=self.proj_key)


    @property
    def latest_index(self):
        return self.indexes[-1]

    @property
    def log_prob(self):
        # the log probability of the hypothesis so far is the sum of the log probabilities of the tokens so far
        return sum(self.log_probs)

    @property
    def avg_log_prob(self):
        # normalize log probability by number of tokens (otherwise longer sequences always have lower probability)
        return self.log_prob / len(self.indexes)

    # @property
    # def tokens(self):
    #     # remove the init_token.
    #     return self.indexes[1:]


def sort_hyps(hyps: List[Hypothesis]) -> List[Hypothesis]:
    """Return a list of Hypothesis objects, sorted by descending average log probability"""
    return sorted(hyps, key=lambda h: h.avg_log_prob, reverse=True)


def run_beam_search(batch, model, batch_size, SOS_INDEX, ):
    # (1) Run the encoder to get the encoder hidden states and mask
    encoder_hidden, encoder_final = model.encode(batch.src, batch.src_mask, batch.src_lengths)

    # Initialize batch_size * beam_size-many hypothesis
    hyps = [Hypothesis(unique_id=idx, indexes=[SOS_INDEX], log_probs=[0.0], hidden_state=None)
            for idx in range(batch_size)]