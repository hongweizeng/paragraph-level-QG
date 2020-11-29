import torch
import torch.nn as nn

from easydict import EasyDict

class EncoderBase(nn.Module):
    def __init__(self, config):
        super(EncoderBase, self).__init__()

    def forward(self):
        pass
