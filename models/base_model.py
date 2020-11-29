from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from torch_scatter import scatter_max

from search.searcher import Hypothesis, sort_hypotheses
from models.modules.stacked_rnn import StackedGRU
from models.modules.concat_attention import ConcatAttention
from models.modules.maxout import MaxOut
from models.base_embedding import EmbeddingBase
from models.base_encoder import EncoderBase
from models.base_decoder import DecoderBase


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embedding = EmbeddingBase(config)
        self.encoder = EncoderBase(config)
        self.decoder = DecoderBase(config)

    def forward(self, batch_data):
        embedding_output_for_encoder = self.embedding(batch_data)
        encoder_outputs = self.encoder(batch_data, embedding_output_for_encoder)

        embedding_output_for_decoder = self.embeddings(batch_data)
        decoder_outputs = self.decoder(batch_data, embedding_output_for_decoder, encoder_outputs)

        return decoder_outputs
