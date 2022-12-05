"""
quantnn.models.pytorch.ligthning
================================

Interface for PyTorch lightning.
"""
import sys
import pickle


import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn


from quantnn.packed_tensor import PackedTensor


def combine_outputs_list(y_preds, ys):
    y_pred_c = []
    y_c = []
    for y_pred, y in zip(y_preds, ys):
        comb = combine_outputs(y_pred, y)
        if comb is None:
            continue
        y_pred, y = comb
        y_pred_c.append(y_pred)
        y_c.append(y)
    if len(y_pred_c) == 0:
        return None
    return y_pred_c, y_c


def combine_outputs(y_pred, y):

    if isinstance(y_pred, list):
        return combine_outputs_list(y_pred, y)

    if isinstance(y_pred, PackedTensor):
        if isinstance(y, PackedTensor):
            y_pred_v, y_v = y_pred.intersection(y_pred, y)
            if y_pred_v is None:
                return None
            return y_pred_v._t, y_v._t
        else:
            if len(y_pred.batch_indices) == 0:
                return None
            return y_pred._t, y[y_pred.batch_indices]
    else:
        if isinstance(y, PackedTensor):
            if len(y.batch_indices) == 0:
                return None
            return y_pred[y.batch_indices], y._t
    return y_pred, y


def to_device(x, device=None, dtype=None):
    if isinstance(x, tuple):
        return tuple([to_device(x_i, device=device, dtype=dtype) for x_i in x])
    elif isinstance(x, list):
        return [to_device(x_i, device=device, dtype=dtype) for x_i in x]
    elif isinstance(x, dict):
        return {k: to_device(x_i, device=device, dtype=dtype) for k, x_i in x.items()}
    elif isinstance(x, PackedTensor):
        return x.to(device=device, dtype=dtype)
    return x



class QuantnnLightning(pl.LightningModule):
    """
    Pytorch Lightning module for quantnn pytorch models.
    """
    def __init__(self,
                 qrnn,
                 loss,
                 name=None,
                 optimizer=None,
                 scheduler=None,
                 metrics=None,
                 mask=None,
                 transformation=None):
        super().__init__()
        self.qrnn = qrnn
        self.model = qrnn.model
        self.loss = loss

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.metrics = metrics
        if self.metrics is None:
            self.metrics = []
        for metric in self.metrics:
            metric.model = self.qrnn
            metric.mask = mask

        self.transformation = transformation
        self.tensorboard = pl.loggers.TensorBoardLogger(
            "lightning_logs",
            name=name
        )

    def training_step(self, batch, batch_idx):
        x, y = batch
        #x = to_device(x, device=self.device, dtype=self.dtype)
        y_pred = self.model(x)

        try:
            y_pred, y = combine_outputs(y_pred, y)
        except TypeError:
            return None

        avg_loss, tot_loss, losses, n_samples = self.model._train_step(
            y_pred, y, self.loss, None,
            metrics=None,
            transformation=self.transformation
        )

        #x = x.detach().cpu().numpy()
        #if isinstance(x, list):
        #    x = [x_i.detach().cpu().numpy() for x_i in x]
        #elif isinstance(x, dict):
        #    x = {k: x_k.detach().cpu().numpy() for k, x_k in x.items()}
        #else:
        #    x = x.detach().cpu().numpy()

        #if isinstance(y, list):
        #    y = [y_i.detach().cpu().numpy() for y_i in y]
        #elif isinstance(y, dict):
        #    y = {k: y_k.detach().cpu().numpy() for k, y_k in y.items()}
        #else:
        #    y = y.detach().cpu().numpy()

        if np.isnan(avg_loss.detach().cpu().numpy()):
            with open("x_prev.pckl", "wb") as output:
                pickle.dump(self.x_prev, output)
            with open("x.pckl", "wb") as output:
                pickle.dump(x, output)
            with open("y_prev.pckl", "wb") as output:
                pickle.dump(self.y_prev, output)
            with open("y.pckl", "wb") as output:
                pickle.dump(y, output)
            sys.exit()
        self.x_prev = x
        self.y_prev = y

        self.log(
            "Training loss",
            avg_loss,
            on_epoch=True,
            batch_size=n_samples,
            sync_dist=True
        )
        losses = {f"Training loss ({key})": loss for key, loss in losses.items()}
        self.log_dict(
            losses,
            on_epoch=True,
            batch_size=n_samples,
            sync_dist=True
        )
        return avg_loss


    def validation_step(self, batch, batch_idx):
        x, y = batch
        #x = to_device(x, device=self.device, dtype=self.dtype)
        y_pred = self.model(x)
        try:
            y_pred, y = combine_outputs(y_pred, y)
        except TypeError:
            return None

        avg_loss, tot_loss, losses, n_samples = self.model._train_step(
            y_pred, y, self.loss, None, metrics=self.metrics,
            transformation=self.transformation
        )
        self.log("Validation loss", avg_loss, on_epoch=True, batch_size=n_samples)
        losses = {f"Validation loss ({key})": loss for key, loss in losses.items()}
        self.log_dict(losses, on_epoch=True, batch_size=n_samples)

    def on_validation_epoch_start(self):
        for metric in self.metrics:
            metric.reset()

    def validation_epoch_end(self, validation_step_outputs):

        i_epoch = self.trainer.current_epoch
        writer = self.tensorboard.experiment

        i_epoch = self.trainer.current_epoch

        if self.trainer.is_global_zero:
            for m in self.metrics:
                # Log values.
                if hasattr(m, "get_values"):
                    values = m.get_values()
                    if isinstance(values, dict):
                        values = {
                            f"{m.name} ({key})": value for key, value in values.items()
                        }
                    else:
                        values = {m.name: values}
                    self.tensorboard.log_metrics(values, i_epoch)

                # Log figures.
                if hasattr(m, "get_figures"):
                    figures = m.get_figures()
                    if isinstance(figures, dict):
                        for target in figures.keys():
                            f = figures[target]
                            writer.add_figure(
                                f"{m.name} ({target})", f, i_epoch
                            )
                    else:
                        writer.add_figure(f"{m.name}", figures, i_epoch)


    def configure_optimizers(self):
        if self.optimizer is None:
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=1e-3
            )
        else:
            optimizer = self.optimizer

        if self.scheduler is None:
            return optimizer
        else:
            return [optimizer], [self.scheduler]
