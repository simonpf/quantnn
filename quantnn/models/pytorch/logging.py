"""
==============================
quantnn.models.pytorch.logging
==============================

This module contains training logger that are specific for the
PyTorch backend.
"""
import torch
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.tensorboard.summary import hparams
import xarray as xr


from quantnn.logging import TrainingLogger


class SummaryWriter(SummaryWriter):
    """
    Specialization of torch original SummaryWriter that overrides 'add_params'
    to avoid creating a new directory to store the hyperparameters.

    Source: https://github.com/pytorch/pytorch/issues/32651
    """

    def add_hparams(self, hparam_dict, metric_dict, epoch):
        torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError("hparam_dict and metric_dict should be dictionary.")
        exp, ssi, sei = hparams(hparam_dict, metric_dict)

        logdir = self._get_file_writer().get_logdir()

        with SummaryWriter(log_dir=logdir) as w_hp:
            w_hp.file_writer.add_summary(exp)
            w_hp.file_writer.add_summary(ssi)
            w_hp.file_writer.add_summary(sei)
            for k, v in metric_dict.items():
                w_hp.add_scalar(k, v, epoch)


class TensorBoardLogger(TrainingLogger):
    """
    Logger that also logs information to tensor board.
    """

    def __init__(
        self, n_epochs, log_rate=100, log_directory=None, epoch_begin_callback=None
    ):
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
        self.attributes = None

    def set_attributes(self, attributes):
        """
        Stores attributes that describe the training in the logger.
        These will be stored in the logger history.

        Args:
            Dictionary of attributes to store in the history of the
            logger.
        """
        super().set_attributes(attributes)

    def epoch_begin(self, model):
        """
        Called at the beginning of each epoch.

        Args:
            The model that is trained in its current state.
        """
        TrainingLogger.epoch_begin(self, model)
        if self.epoch_begin_callback:
            self.epoch_begin_callback(self.writer, model, self.i_epoch)

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
        super().training_step(loss, n_samples, of=of, losses=losses)

    def validation_step(self, loss, n_samples, of=None, losses=None):
        """
        Log processing of a validation batch.

        Args:
            i: The index of the current batch.
            loss: The loss of the current batch.
            n_samples: The number of samples in the batch.
            of: If available the number of batches in the epoch.
        """
        super().validation_step(loss, n_samples, of=of, losses=losses)

    def epoch(self, learning_rate=None, metrics=None):
        """
        Log processing of epoch.

        Args:
            learning_rate: If available the learning rate of the optimizer.
        """
        TrainingLogger.epoch(self, learning_rate, metrics=metrics)
        self.writer.add_scalar("Learning rate", learning_rate, self.i_epoch)

        for name, v in self.history.variables.items():
            if name == "epochs":
                continue
            if len(v.dims) == 1:
                value = v.data[-1]
                self.writer.add_scalar(name, value, self.i_epoch)

        if metrics is not None:
            for m in metrics:
                if hasattr(m, "get_figures"):
                    figures = m.get_figures()
                    if isinstance(figures, dict):
                        for target in figures.keys():
                            f = figures[target]
                            self.writer.add_figure(
                                f"{m.name} ({target})", f, self.i_epoch
                            )
                    else:
                        self.writer.add_figure(f"{m.name}", figures, self.i_epoch)

    def training_end(self):
        """
        Called to signal the end of the training to the logger.
        """
        if self.attributes is not None:
            if self.i_epoch >= self.n_epochs:
                metrics = {}
                for name, v in self.history.variables.items():
                    if name == "epochs":
                        continue
                    if len(v.dims) == 1:
                        metrics[name + "_final"] = v.data[-1]
                self.writer.add_hparams(self.attributes, {}, self.i_epoch)
                self.writer.flush()

    def __del__(self):
        # Extract metric values for hyper parameters.
        super().__del__()
