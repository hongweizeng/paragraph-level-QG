import torch
import torch.nn as nn


class ConcatAttention(nn.Module):  #generate current state via attention and hi
    def __init__(self, attend_dim, query_dim, att_dim, use_coverage=False):
        super(ConcatAttention, self).__init__()
        self.attend_dim = attend_dim
        self.query_dim = query_dim
        self.att_dim = att_dim
        self.linear_pre = nn.Linear(attend_dim, att_dim, bias=True)
        self.linear_q = nn.Linear(query_dim, att_dim, bias=False)
        self.linear_v = nn.Linear(att_dim, 1, bias=False)
        self.linear_c = nn.Linear(1, att_dim, bias=False)
        self.sm = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        self.mask = None

        self.use_coverage = use_coverage

    def applyMask(self, mask):
        self.mask = mask.long()

    def forward(self, input, context, coverage=None, precompute=None, encoder_mask=None):
        """
        input: batch x dim
        context: batch x sourceL x dim
        """
        if precompute is None:
            precompute = self.linear_pre(context)       # batch x sourceL x att_dim
        targetT = self.linear_q(input).unsqueeze(1)  # batch x 1 x att_dim

        tmp_sum = precompute + targetT.expand_as(precompute)  # batch x sourceL x att_dim

        if self.use_coverage:
            Wc = self.linear_c(coverage.contiguous().unsqueeze(2))
            # Wc = Wc.view(coverage.size(0), coverage.size(1), -1)
            tmp_sum += Wc

        tmp_activated = self.tanh(tmp_sum)  # batch x sourceL x att_dim
        energy = self.linear_v(tmp_activated).view(tmp_sum.size(0), tmp_sum.size(1))  # batch x sourceL

        # tmp10 = precompute + targetT.expand_as(precompute)  # batch x sourceL x att_dim
        # tmp10 = Wc + precompute + targetT.expand_as(precompute)  # batch x sourceL x att_dim
        # tmp20 = self.tanh(tmp10)  # batch x sourceL x att_dim
        # energy = self.linear_v(tmp20.view(-1, tmp20.size(2))).view(tmp20.size(0), tmp20.size(1))  # batch x sourceL
        if encoder_mask is not None:
            # energy.data.masked_fill_(self.mask, -float('inf'))
            # energy.masked_fill_(self.mask, -float('inf'))   # TODO: might be wrong
            # energy = energy.squeeze(1).masked_fill(self.mask == 0, value=-1e12)
            energy = energy * encoder_mask.long() + (1 - encoder_mask.long()) * (-1000000)      # 1 for true token, 0 for padding token.
        score = self.sm(energy)
        score_m = score.view(score.size(0), 1, score.size(1))  # batch x 1 x sourceL

        weightedContext = torch.bmm(score_m, context).squeeze(1)  # batch x dim

        return weightedContext, score, precompute, energy

    # def extra_repr(self):
    #     return self.__class__.__name__ + '(' + str(self.att_dim) + ' * ' + '(' \
    #            + str(self.attend_dim) + '->' + str(self.att_dim) + ' + ' \
    #            + str(self.query_dim) + '->' + str(self.att_dim) + ')' + ')'
    #
    # def __repr__(self):
    #     return self.__class__.__name__ + '(' + str(self.att_dim) + ' * ' + '(' \
    #            + str(self.attend_dim) + '->' + str(self.att_dim) + ' + ' \
    #            + str(self.query_dim) + '->' + str(self.att_dim) + ')' + ')'
