"""
==============================
quantnn.models.pytorch.logging
==============================

This module contains training logger that are specific for the
PyTorch backend.
"""
from torch.utils.tensorboard.writer import SummaryWriter

from quantnn.logging import TrainingLogger


class TensorBoardLogger(TrainingLogger):
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
        self.epoch_begin_callback = epoch_begin_callback

        self.step_training = 0
        self.step_validation = 0
        self.step_epoch = 0

    def epoch_begin(self, model):
        """
        Called at the beginning of each epoch.

        Args:
            The model that is trained in its current state.
        """
        TrainingLogger.epoch_begin(self, model)
        if self.epoch_begin_callback:
            self.epoch_begin_callback(self.writer, model, self.step_epoch)
        self.step_epoch += 1

    def training_step(self, loss, n_samples, of=None, losses=None):
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
                        of=None,
                        losses=None):
        """
        Log processing of a validation batch.

        Args:
            i: The index of the current batch.
            loss: The loss of the current batch.
            n_samples: The number of samples in the batch.
            of: If available the number of batches in the epoch.
        """
        TrainingLogger.validation_step(self, loss, n_samples, of=of)
        self.writer.add_scalar("Validation loss", loss, self.step_validation)
        self.step_validation += 1

    def epoch(self, learning_rate=None, metrics=None):
        """
        Log processing of epoch.

        Args:
            learning_rate: If available the learning rate of the optimizer.
        """
        self.writer.add_scalar("Learning rate", learning_rate)

        TrainingLogger.epoch(self, learning_rate)

        if metrics is not None:
            for m in metrics:
                if hasattr(m, "get_figures"):
                    figures = m.get_figures()
                    if isinstance(figures, dict):
                        for target in figures.keys():
                            f = figures[target]
                            self.writer.add_figure(f"{m.name} ({target})", f, self.i_epoch)
                    else:
                        self.writer.add_figure(f"{m.name}", figures, self.i_epoch)
