"""
==============================
quantnn.models.pytorch.logging
==============================

This module contains training logger that are specific for the
PyTorch backend.
"""
from torch.utils.tensorboard.writer import SummaryWriter

from quantnn.logging import TrainingLogger


class TensorboardLogger(TrainingLogger):
    """
    Logger that also logs information to tensor board.
    """
    def __init__(self,
                 n_epochs,
                 log_rate=100,
                 log_directory=None,
                 epoch_begin_callback=None):
        """
        Create a new logger instance.

        Args:
            n_epochs: The number of epochs for which the training will last.
            log_rate: The message rate for output to standard out.
            log_directory: The directory to use for tensorboard output.
            epoch_begin_callback: Callback function the will be called with
                arguments ``writer, model``, where ``writer`` is the current
                ``torch.utils.tensorboard.writer.SummaryWriter`` object used
                used to write output and ``model`` is the model that is being
                in its current state.
        """
        super().__init__(n_epochs, log_rate)
        self.writer = SummaryWriter(log_dir=log_directory)
        self.step_training = 0
        self.step_validation = 0
        self.epoch_begin_callback = epoch_begin_callback

    def epoch_begin(self, model):
        """
        Called at the beginning of each epoch.

        Args:
            The model that is trained in its current state.
        """
        TrainingLogger.epoch_begin(self, model)
        if self.epoch_begin_callback:
            self.epoch_begin_callback(self.writer, model)

    def training_step(self, loss, n_samples, of=None):
        """
        Log processing of a training batch. This method should be called
        after each batch is processed so that the logger can keep track
        of training progress.

        Args:
            loss: The loss of the current batch.
            n_samples: The number of samples in the batch.
            of: If available the number of batches in the epoch.
        """
        TrainingLogger.training_step(self, loss, n_samples, of=of)
        self.writer.add_scalar("Training loss", loss, self.step_training)
        self.step_training += 1

    def validation_step(self,
                        loss,
                        n_samples,
                        of=None):
        """
        Log processing of a validation batch.

        Args:
            i: The index of the current batch.
            loss: The loss of the current batch.
            n_samples: The number of samples in the batch.
            of: If available the number of batches in the epoch.
        """
        TrainingLogger.validation_step(self, loss, n_samples, of=None)
        self.writer.add_scalar("Validation loss", loss, self.step_validation)
        self.step_validation += 1

    def epoch(self, learning_rate=None):
        """
        Log processing of epoch.

        Args:
            learning_rate: If available the learning rate of the optimizer.
        """
        TrainingLogger.epoch(self, learning_rate)
