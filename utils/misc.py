import numpy

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