import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from utils import logger
from train.lr_scheduler import make_learning_rate_decay_fn


def build_torch_optimizer(params, config):
    """Builds the PyTorch optimizer.
    Input:
        model: The model to optimize.
        opt: The dictionary of options.
    Output:
        A ``torch.optim.Optimizer`` instance.
    """
    if config.optimizer_name == 'sgd':
        optimizer = optim.SGD(params, lr=config.learning_rate, momentum=config.momentum, dampening=0,
                              weight_decay=config.weight_decay, nesterov=False)
    elif config.optimizer_name == 'adam':
        optimizer = optim.Adam(params, lr=config.learning_rate, betas=(config.adam_beta1, config.adam_beta2), eps=1e-9,
                               weight_decay=config.weight_decay, amsgrad=False)
    else:
        raise ValueError('Invalid optimizer type: ' + config.optimizer_name)

    return optimizer, params





class Optimizer(object):
    """
    Controller class for optimization. Mostly a thin wrapper for `optim`,
    but also useful for implementing rate scheduling beyond what is currently available.
    Also implements necessary methods for training RNNs such as grad manipulations.
    """

    def __init__(self, model, config):

        params = list(filter(lambda p: p.requires_grad, model.parameters()))
        # [p for p in model.parameters() if p.requires_grad]

        optimizer, params = build_torch_optimizer(params, config)

        self._optimizer = optimizer
        self._params = params

        self._learning_rate = config.learning_rate

        self._learning_rate_decay_method = config.decay_method
        self._learning_rate_decay = config.learning_rate_decay

        self._learning_rate_decay_fn = make_learning_rate_decay_fn(config)

        self._start_decay_steps = config.start_decay_steps
        self._decay_step = config.decay_steps

        self._bad_cnt = 0
        self._decay_bad_cnt = config.decay_bad_cnt

        self._training_step = 0

        self._max_grad_norm = config.max_grad_norm or 0
        self._max_weight_value = config.max_weight_value

    @property
    def training_step(self):
        """The current training step."""
        return self._training_step

    # @property
    def get_learning_rate(self):
        return self._learning_rate

    def learning_rate(self, is_improving):
        """Returns the current learning rate."""

        if is_improving:
            self._bad_cnt = 0
        else:
            self._bad_cnt += 1

        if self._training_step % self._decay_step == 0 and self._training_step > self._start_decay_steps:

            if self._bad_cnt >= self._decay_bad_cnt and self._learning_rate >= 1e-5:

                if self._learning_rate_decay_method:
                    scale = self._learning_rate_decay_fn(self._decay_step)
                    self._decay_step += 1
                    self._learning_rate *= scale
                else:
                    self._learning_rate *= self._learning_rate_decay

                self._bad_cnt = 0

        return self._learning_rate

    def state_dict(self):
        # return {
        #     'training_step': self._training_step,
        #     'decay_step': self._decay_step,
        #     'optimizer': self._optimizer.state_dict()
        # }
        return self._optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self._training_step = state_dict['training_step']
        # State can be partially restored.
        if 'decay_step' in state_dict:
            self._decay_step = state_dict['decay_step']
        if 'optimizer' in state_dict:
            self._optimizer.load_state_dict(state_dict['optimizer'])

    def zero_grad(self):
        """Zero the gradients of optimized parameters."""
        self._optimizer.zero_grad()

    def backward_and_step_and_update(self):
        pass

    def backward(self, loss):
        """Wrapper for backward pass. Some optimizer requires ownership of the backward pass."""
        loss.backward()

    def step(self):
        """Update the model parameters based on current gradients. """
        learning_rate = self._learning_rate
        for group in self._optimizer.param_groups:
            group['lr'] = learning_rate
            if self._max_grad_norm > 0:
                clip_grad_norm_(group['params'], self._max_grad_norm)
        self._optimizer.step()
        if self._max_weight_value:
            for p in self._params:
                p.data.clamp_(0 - self._max_weight_value, self._max_weight_value)
        self._training_step += 1

    def update_learning_rate(self, is_improving):
        old_learning_rate = self._learning_rate
        new_learning_rate = self.learning_rate(is_improving)

        if old_learning_rate != new_learning_rate:
            logger.info("Update the learning rate from %.5f to %.5f" % (old_learning_rate, new_learning_rate))
