class Scorer(object):
    def __init__(self, best_score, name):
        self.best_score = best_score
        self.name = name

    def is_improving(self, stats):
        raise NotImplementedError()

    def is_decreasing(self, stats):
        raise NotImplementedError()

    def update(self, stats):
        self.best_score = self._caller(stats)

    def __call__(self, stats, **kwargs):
        return self._caller(stats)

    def _caller(self, stats):
        raise NotImplementedError()


class PPLScorer(Scorer):

    def __init__(self):
        super(PPLScorer, self).__init__(float("inf"), "ppl")

    def is_improving(self, stats):
        return stats.ppl() < self.best_score

    def is_decreasing(self, stats):
        return stats.ppl() > self.best_score

    def _caller(self, stats):
        return stats.ppl()


class AccuracyScorer(Scorer):

    def __init__(self):
        super(AccuracyScorer, self).__init__(float("-inf"), "acc")

    def is_improving(self, stats):
        return stats.accuracy() > self.best_score

    def is_decreasing(self, stats):
        return stats.accuracy() < self.best_score

    def _caller(self, stats):
        return stats.accuracy()


class XentScorer(Scorer):

    def __init__(self):
        super(XentScorer, self).__init__(float("inf"), "xent")

    def is_improving(self, stats):
        return stats.xent() < self.best_score

    def is_decreasing(self, stats):
        return stats.xent() > self.best_score

    def _caller(self, stats):
        return stats.xent()


class BLEUScorer(Scorer):

    def __init__(self):
        super(BLEUScorer, self).__init__(float("-inf"), "bleu")

    def is_improving(self, stats):
        return stats.bleu > self.best_score

    def is_decreasing(self, stats):
        return stats.bleu < self.best_score

    def _caller(self, stats):
        return stats.bleu


DEFAULT_SCORERS = [PPLScorer(), AccuracyScorer(), XentScorer(), BLEUScorer()]

SCORER_BUILDER = {
    "ppl": PPLScorer,
    "acc": AccuracyScorer,
    "xent": XentScorer,
    "bleu": BLEUScorer
}


def build_scorers(criteria):
    if criteria is None:
        return DEFAULT_SCORERS
    else:
        scorers = []
        for criterion in set(criteria):
            assert criterion in SCORER_BUILDER.keys(), "Criterion {} not found".format(criterion)
            scorers.append(SCORER_BUILDER[criterion]())
        return scorers
