"""
===============
quantnn.logging
===============

This module provides a generic training logger to handle the logging of
training information.
"""
from datetime import datetime

class TrainingLogger:
    """
    Logger class that prints training statistics to standard out.
    """
    def __init__(self, n_epochs, log_rate=100):
        """
        Create a new logger instance.

        Args:
            n_epochs: The number of epochs for which the training will last.
            log_rate: The number of training steps to perform between
                 subsequent messages.
        """
        self.i_epoch = 0
        self.i_train_batch = 0
        self.i_val_batch = 0
        self.n_epochs = n_epochs

        self.train_loss = 0.0
        self.train_samples = 0
        self.val_loss = 0.0
        self.val_samples = 0

        self.log_rate = log_rate
        self.epoch_start_time = datetime.now()

    def epoch_begin(self, model):
        """
        Called at the beginning of each epoch.

        Args:
            The model that is trained in its current state.
        """
        self.epoch_start_time = datetime.now()
        self.train_loss = 0.0
        self.train_samples = 0

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
        self.train_loss += n_samples * loss
        self.train_samples += n_samples

        if (self.i_train_batch % self.log_rate) == self.log_rate - 1:

            if of is None:
                of = "?"
            else:
                of = f"{of:2}"

            avg_loss = self.train_loss / self.train_samples
            msg = f"Batch {self.i_train_batch:2} / {of}: train. loss = {avg_loss:.4f}"
            print(msg, end="\r")
        self.i_train_batch += 1

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
        if (self.i_val_batch == 0):
            self.val_loss = 0.0
            self.val_samples = 0

        self.val_loss += n_samples * loss
        self.val_samples += n_samples
        self.i_val_batch += 1

    def epoch(self, learning_rate=None):
        """
        Log processing of epoch.

        Args:
            learning_rate: If available the learning rate of the optimizer.
        """
        train_loss = self.train_loss / self.train_samples
        of = self.n_epochs

        msg = f"Epoch {self.i_epoch + 1:2} / {of:2}: train. loss = {train_loss:.4f}"
        if self.val_samples > 0:
            val_loss = self.val_loss / self.val_samples
            msg += f", val. loss = {val_loss:.4f}"
        if learning_rate:
            msg += f", lr. = {learning_rate:.4f}"

        dt = (datetime.now() - self.epoch_start_time).total_seconds()
        msg += f", time = {dt} s"
        print(msg)

        self.i_epoch += 1
        self.i_train_batch = 0
        self.i_val_batch = 0
