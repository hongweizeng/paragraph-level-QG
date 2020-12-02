import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_
from torchtext.vocab import Vocab
from transformers.modeling_utils import ModuleUtilsMixin

import numpy


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


def init_embeddings(embedding_module: nn.Embedding, vocabulary: Vocab, vectors='glove.6B.300d', freeze=False):
    vocabulary.load_vectors(vectors)
    embedding_module.weight.data.copy_(vocabulary.vectors)

    if freeze:
        freeze_module(embedding_module)


def freeze_module(module):
    for param in module.parameters():
        param.requires_grad = False


def unfreeze_module(module):
    for param in module.parameters():
        param.requires_grad = True


def count_params(module):
    module_parameters = filter(lambda p: p.requires_grad, module.parameters())
    param_cnt = sum([numpy.prod(p.size()) for p in module_parameters])
    return param_cnt