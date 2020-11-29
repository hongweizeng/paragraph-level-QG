import functools
from math import sqrt


def make_learning_rate_decay_fn(opt):
    """Returns the learning decay function from options."""
    if opt.decay_method == 'noam':
        return functools.partial(noam_decay, warmup_steps=opt.n_warmup_steps, model_size=opt.d_model)
    elif opt.decay_method == 'noamwd':
        return functools.partial(noamwd_decay, warmup_steps=opt.n_warmup_steps, model_size=opt.d_model,
                                 rate=opt.learning_rate_decay, decay_steps=opt.decay_steps,
                                 start_step=opt.start_decay_steps)
    elif opt.decay_method == 'rsqrt':
        return functools.partial(rsqrt_decay, warmup_steps=opt.n_warmup_steps)
    elif opt.start_decay_steps is not None:
        return functools.partial(exponential_decay, rate=opt.learning_rate_decay, decay_steps=opt.decay_steps,
                                 start_step=opt.start_decay_steps)


def noam_decay(step, warmup_steps, model_size):
    """Learning rate schedule described in https://arxiv.org/pdf/1706.03762.pdf. """
    return (model_size ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5)))


def noamwd_decay(step, warmup_steps, model_size, rate, decay_steps, start_step=0):
    """Learning rate schedule optimized for huge batches"""
    return (model_size ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5)) *
            rate ** (max(step - start_step + decay_steps, 0) // decay_steps))


def exponential_decay(step, rate, decay_steps, start_step=0):
    """A standard exponential decay, scaling the learning rate by :obj:`rate` every :obj:`decay_steps` steps. """
    return rate ** (max(step - start_step + decay_steps, 0) // decay_steps)


def rsqrt_decay(step, warmup_steps):
    """Decay based on the reciprocal of the step square root."""
    return 1.0 / sqrt(max(step, warmup_steps))