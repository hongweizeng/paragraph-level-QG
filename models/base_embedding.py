from __future__ import division

import torch.nn as nn


class EmbeddingBase(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=config.pad_token_id)

    def forward(self, input_ids=None):
        pass