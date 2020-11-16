import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_
from torchtext.vocab import Vocab
from transformers.modeling_utils import ModuleUtilsMixin


class TrainedModel(nn.Module, ModuleUtilsMixin):
    pass


def init_parameters(model, config):
    if config.param_init != 0.0:
        for p in model.parameters():
            p.data.uniform_(-config.param_init, config.param_init)
    if config.param_init_glorot:
        for p in model.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


def init_embeddings(embedding_module: nn.Embedding, vocabulary: Vocab, vectors='glove.6B.300d'):
    vocabulary.load_vectors(vectors)
    embedding_module.weight.data.copy_(vocabulary.vectors)