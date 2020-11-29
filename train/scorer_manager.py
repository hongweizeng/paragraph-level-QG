# https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/utils/earlystopping.py

from enum import Enum
from utils.logging import logger

from train.scorers import build_scorers

class PatienceEnum(Enum):
    IMPROVING = 0
    DECREASING = 1
    STOPPED = 2


class ScorerManager(object):

    def __init__(self, config):
        """
            Callable class to keep track of early stopping.
            Args:
                tolerance(int): number of validation steps without improving
                scorer(fn): list of scorers to validate performance on dev
        """

        tolerance = config['tolerance']
        scorers = build_scorers(config['criteria'])

        self.tolerance = tolerance
        self.stalled_tolerance = self.tolerance
        self.current_tolerance = self.tolerance
        self.early_stopping_scorers = scorers
        self.status = PatienceEnum.IMPROVING
        self.current_step_best = 0

    def __call__(self, valid_stats, step):
        """
            Update the internal state of early stopping mechanism, whether to
        continue training or stop the train procedure.
            Checks whether the scores from all pre-chosen scorers improved. If
        every metric improve, then the status is switched to improving and the
        tolerance is reset. If every metric deteriorate, then the status is
        switched to decreasing and the tolerance is also decreased; if the
        tolerance reaches 0, then the status is changed to stopped.
        Finally, if some improved and others not, then it's considered stalled;
        after tolerance number of stalled, the status is switched to stopped.
        :param valid_stats: Statistics of dev set
        """

        if self.status == PatienceEnum.STOPPED:
            # Don't do anything
            return

        if all([scorer.is_improving(valid_stats) for scorer
                in self.early_stopping_scorers]):
            self._update_increasing(valid_stats, step)

        elif all([scorer.is_decreasing(valid_stats) for scorer
                  in self.early_stopping_scorers]):
            self._update_decreasing()

        else:
            self._update_stalled()

    def _update_stalled(self):
        self.stalled_tolerance -= 1

        logger.info(
            "Stalled patience: {}/{}".format(self.stalled_tolerance,
                                             self.tolerance))

        if self.stalled_tolerance == 0:
            logger.info(
                "Training finished after stalled validations. Early Stop!"
            )
            self._log_best_step()

        self._decreasing_or_stopped_status_update(self.stalled_tolerance)

    def _update_increasing(self, valid_stats, step):
        self.current_step_best = step
        for scorer in self.early_stopping_scorers:
            logger.info(
                "Model is improving {}: {:g} --> {:g}.".format(
                    scorer.name, scorer.best_score, scorer(valid_stats))
            )
            # Update best score of each criteria
            scorer.update(valid_stats)

        # Reset tolerance
        self.current_tolerance = self.tolerance
        self.stalled_tolerance = self.tolerance

        # Update current status
        self.status = PatienceEnum.IMPROVING

    def _update_decreasing(self):
        # Decrease tolerance
        self.current_tolerance -= 1

        # Log
        logger.info(
            "Decreasing patience: {}/{}".format(self.current_tolerance,
                                                self.tolerance)
        )
        # Log
        if self.current_tolerance == 0:
            logger.info("Training finished after not improving. Early Stop!")
            self._log_best_step()

        self._decreasing_or_stopped_status_update(self.current_tolerance)

    def _log_best_step(self):
        logger.info("Best model found at step {}".format(
            self.current_step_best))

    def _decreasing_or_stopped_status_update(self, tolerance):
        self.status = PatienceEnum.DECREASING \
            if tolerance > 0 \
            else PatienceEnum.STOPPED

    def is_improving(self):
        return self.status == PatienceEnum.IMPROVING

    def early_stop(self):
        return self.status == PatienceEnum.STOPPED
