import math
from overrides import overrides


def sort_hypotheses(hypotheses):
    return sorted(hypotheses, key=lambda h: h.score, reverse=True)


class Hypothesis(object):
    def __init__(self, tokens, log_probs, state, context=None, coverage=None, uncertainty_scores=None, beta=0.):
        self.tokens = tokens
        self.log_probs = log_probs
        self.state = state
        self.context = context
        self.coverage = coverage
        self.uncertainty_scores = uncertainty_scores

        self.beta = beta

    def extend(self, token, log_prob, state, context=None, coverage=None, uncertainty_scores=None):
        h = Hypothesis(tokens=self.tokens + [token],
                       log_probs=self.log_probs + [log_prob],
                       state=state,
                       context=context,
                       coverage=coverage,
                       uncertainty_scores=self.uncertainty_scores + [uncertainty_scores])
        return h

    @property
    def latest_token(self):
        return self.tokens[-1]

    @property
    def avg_log_prob(self):
        return sum(self.log_probs) / len(self.tokens)

    @property
    def uncertainty_aware_score(self):
        return 0.885 * sum(self.log_probs) / len(self.tokens) - 0.115 * math.log((sum(self.uncertainty_scores) / len(self.tokens)))

    @property
    def length(self):
        return len(self.tokens)

    @property
    def score(self):
        # return sum(self.log_probs) / len(self.tokens)
        return (1 - self.beta) * sum(self.log_probs) / self.length - self.beta * math.log(sum(self.uncertainty_scores) / self.length)
