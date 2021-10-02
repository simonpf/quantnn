"""
quantnn.models.pytorch.common
=============================

This module provides common functionality required to realize QRNNs in pytorch.
"""
from inspect import signature
from collections.abc import Mapping, Iterable
import os
import shutil
import tarfile
import tempfile

import torch
import numpy as np
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset

from quantnn.common import (ModelNotSupported,
                            InputDataError,
                            DatasetError,
                            ModelLoadError)
from quantnn.logging import TrainingLogger
import quantnn.data
from quantnn.backends.pytorch import PyTorch
from quantnn.generic import to_array
from quantnn.utils import apply

activations = {
    "elu": nn.ELU,
    "hardshrink": nn.Hardshrink,
    "hardtanh": nn.Hardtanh,
    "prelu": nn.PReLU,
    "relu": nn.ReLU,
    "selu": nn.SELU,
    "celu": nn.CELU,
    "sigmoid": nn.Sigmoid,
    "softplus": nn.Softplus,
    "softmin": nn.Softmin,
}


_ZERO_GRAD_ARGS = {}
major, minor, *_ = torch.__version__.split(".")
if int(major) >= 1 and int(minor) > 7:
    _ZERO_GRAD_ARGS = {"set_to_none": True}


def save_model(f, model):
    """
    Save pytorch model.

    Args:
        f(:code:`str` or binary stream): Either a path or a binary stream
            to store the data to.
        model(:code:`pytorch.nn.Moduel`): The pytorch model to save
    """
    path = tempfile.mkdtemp()
    filename = os.path.join(path, "module.h5")
    torch.save(model, filename)
    archive = tarfile.TarFile(fileobj=f, mode="w")
    archive.add(filename, arcname="module.h5")
    archive.close()
    shutil.rmtree(path)


def load_model(file):
    """
    Load pytorch model.

    Args:
        file(:code:`str` or binary stream): Either a path or a binary stream
            to read the model from
        quantiles(:code:`np.ndarray`): Array containing the quantiles
            that the model predicts.

    Returns:
        The loaded pytorch model.
    """
    path = tempfile.mkdtemp()
    tar_file = tarfile.TarFile(fileobj=file, mode="r")

    # Check that archive contains 'module.h5' file or
    # 'keras_model.h5' for backwards compatibility.
    names = tar_file.getnames()
    if "keras_model.h5" in names:
        tar_file.extract("keras_model.h5", path=path)
        filename = os.path.join(path, "keras_model.h5")
    elif "module.h5" in names:
        tar_file.extract("module.h5", path=path)
        filename = os.path.join(path, "module.h5")
    else:
        raise ModelLoadError(
            "Model archive does not contain the expected model file. It looks"
            "like your file is corrupted."
        )
    model = torch.load(filename, map_location=torch.device("cpu"))
    shutil.rmtree(path)
    return model


def handle_input(data, device=None):
    """
    Handle input data.

    This function handles data supplied

      - as tuple of :code:`np.ndarray`
      - a single :code:`np.ndarray`
      - torch :code:`dataloader`

    If a numpy array is provided it is converted to a torch tensor
    so that it can be fed into a pytorch model.
    """
    if type(data) == tuple:
        x, y = data

        dtype_y = torch.float
        if "int" in str(y.dtype):
            dtype_y = torch.long

        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(y, dtype=dtype_y)
        if device is not None:
            x = x.to(device)
            y = y.to(device)
        return x, y

    if type(data) == np.ndarray:
        x = torch.tensor(data, dtype=torch.float)
        if device is not None:
            x = x.to(device)
        return x

    return data


class BatchedDataset(quantnn.data.BatchedDataset):
    """
    Batches an un-batched dataset.
    """

    def __init__(self, training_data, batch_size=64):
        x, y = training_data
        super().__init__(x, y, batch_size, False, PyTorch)


def get_batch_size(x):
    """
    Get batch size of tensor or iterable of tensors.
    """
    if isinstance(x, Iterable):
        return x[0].shape[0]
    else:
        return x.shape[0]

################################################################################
# Quantile loss
################################################################################


class CrossEntropyLoss(nn.CrossEntropyLoss):
    """
    Cross entropy loss with optional masking.

    This loss function class calculates the mean cross entropy loss
    over the given inputs but applies an optional masking to the
    inputs, in order to allow the handling of missing values.
    """

    def __init__(self, bins, mask=None):
        """
        Args:
            mask: All values that are smaller than or equal to this value will
                 be excluded from the calculation of the loss.
        """
        self.bins = apply(lambda x: torch.Tensor(x).to(torch.float), bins)
        if mask is None:
            reduction = "mean"
            self.mask = mask
        else:
            reduction = "none"
            self.mask = np.float32(mask)
        super().__init__(reduction=reduction)

    def __call__(self, y_pred, y_true, key=None):
        """Evaluate the loss."""
        if not isinstance(self.bins, dict) or key is None:
            bins = self.bins.to(y_pred.device)
        else:
            bins = self.bins[key].to(y_pred.device)
        y_cat = torch.bucketize(y_true, bins[1:-1])

        if len(y_cat.shape) == len(y_pred.shape):
            y_cat = y_cat.squeeze(1)
            y_true = y_true.squeeze(1)

        if self.mask is None:
            return nn.CrossEntropyLoss.__call__(self, y_pred, y_cat)
        else:
            loss = nn.CrossEntropyLoss.__call__(
                self,
                y_pred,
                y_cat,
            )
            mask = (y_true > self.mask).to(dtype=y_pred.dtype)

            return (loss * mask).sum() / (mask.sum() + 1e-6)


class QuantileLoss(nn.Module):
    r"""
    The quantile loss function

    This function object implements the quantile loss defined as


    .. math::

        \mathcal{L}(y_\text{pred}, y_\text{true}) =
        \begin{cases}
        \tau \cdot |y_\text{pred} - y_\text{true}| & , y_\text{pred} < y_\text{true} \\
        (1 - \tau) \cdot |y_\text{pred} - y_\text{true}| & , \text{otherwise}
        \end{cases}


    as a training criterion for the training of neural networks. The loss criterion
    expects a vector :math:`\mathbf{y}_\tau` of predicted quantiles and the observed
    value :math:`y`. The loss for a single training sample is computed by summing the losses
    corresponding to each quantiles. The loss for a batch of training samples is
    computed by taking the mean over all samples in the batch.
    """

    def __init__(self, quantiles, mask=None, quantile_axis=1):
        """
        Create an instance of the quantile loss function with the given quantiles.

        Arguments:
            quantiles: Array or iterable containing the quantiles to be estimated.
        """
        super().__init__()
        self.quantiles = torch.tensor(quantiles).float()
        self.n_quantiles = len(quantiles)
        self.mask = mask
        if self.mask:
            self.mask = np.float32(mask)
        self.quantile_axis = quantile_axis

    def to(self, device):
        self.quantiles = self.quantiles.to(device)

    def __call__(self, y_pred, y_true, key=None):
        """
        Compute the mean quantile loss for given inputs.

        Arguments:
            y_pred: N-tensor containing the predicted quantiles along the last
                dimension
            y_true: (N-1)-tensor containing the true y values corresponding to
                the predictions in y_pred

        Returns:
            The mean quantile loss.
        """
        y_true = y_true.to(y_pred.dtype)
        dy = y_pred - y_true
        n = self.quantiles.size()[0]

        shape = [
            1,
        ] * len(dy.size())
        shape[self.quantile_axis] = self.n_quantiles
        qs = self.quantiles.reshape(shape)
        l = torch.where(dy >= 0.0, (1.0 - qs) * dy, (-qs) * dy)
        if self.mask:
            mask = (y_true > self.mask).to(y_true.dtype)
            return (l * mask).sum() / ((mask.sum() + 1e-6) * self.n_quantiles)
        return l.mean()


class MSELoss(nn.Module):
    r"""
    Mean-squared error loss with masking.
    """
    def __init__(self, mask=None):
        """
        Args:
            mask: Only values larger than the given mask value will be
                considered in the loss.
        """
        super().__init__()
        self.mask = mask
        if self.mask:
            self.mask = np.float32(mask)
        if self.mask is None:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.MSELoss(reduction="none")

    def to(self, device):
        pass

    def __call__(self, y_pred, y_true, key=None):
        """
        Compute mean-squared error loss.

        Arguments:
            y_pred: N-tensor containing the predicted quantities.
            y_true: N-tensor containing the true y values corresponding to
                the predictions in y_pred

        Returns:
            The MSE.
        """
        if self.mask is None:
            return self.loss(y_pred, y_true)
        dy_2 = self.loss(y_pred, y_true)
        mask = (y_true > self.mask).to(y_true.dtype)
        return (dy_2 * mask).sum() / (mask.sum() + 1e-6)

################################################################################
# Default scheduler and optimizer
################################################################################


def _get_default_optimizer(model):
    """
    The default optimizer. Currently set to Adam optimizer.
    """
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    return optimizer


def _get_default_scheduler(optimizer):
    """
    The default scheduler which reduces lr when training loss reaches a
    plateau.
    """
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5)
    return scheduler


def _has_channels_last_tensor(parameters):
    """
    Determine whether any of the tensors in the models parameters is
    in channels last format.
    """
    for p in parameters:
        if isinstance(p.data, torch.Tensor):
            t = p.data
            if (
                t.is_contiguous(memory_format=torch.channels_last)
                and not t.is_contiguous()
            ):
                return True
            elif isinstance(t, list) or isinstance(t, tuple):
                if _has_channels_last_tensor(list(t)):
                    return True
    return False


def _get_x_y(batch_data, keys):
    """
    Retrieve training input from batch data.

    This function checks whether the object returned as a batch from
    the training loader is an iterable or a mapping. If it is an
    iterable it will simply unpack the input- and target-data in the
    order ``x, y``. If ``batch_data`` is a mapping and keys is not
    ``None`` it will use the two arguments in keys to retrieve elements from
    ``batch_data``. If ``keys`` is ``None``, the first to keys will
    be used to retrieve inputs and targets, respectively.

    Args:
        batch_data: The object that is obtained when iterating o

    Returns:
        Tuple ``x, y`` of input data ``x`` and corresponding output data
        ``y``.
    """
    if isinstance(batch_data, Mapping):

        if keys is not None:
            try:
                x_key, y_key = keys
            except ValueError:
                raise DatasetError(
                    f"Could not unpack provided keys f{keys} into "
                    "variables 'x_key, y_key'"
                )

        else:
            try:
                x_key, y_key = batch_data.keys()
            except ValueError as v:
                raise DatasetError(
                    f"Could not unpack batch keys f{batch_data.keys()} into "
                    "variables 'x_key, y_key'"
                )

        try:
            x = batch_data[x_key]
            y = batch_data[y_key]
        except Exception as e:
            raise DatasetError(
                "The following error was encountered when trying to "
                f"retrieve the keys '{x_key}' and '{y_key} from a batch of  "
                f"training data.: {e}"
            )
    else:
        x, y = batch_data
    return x, y


###############################################################################
# QRNN
###############################################################################


class PytorchModel:
    """
    Quantile regression neural network (QRNN)

    This class implements QRNNs as a fully-connected network with
    a given number of layers.
    """

    @staticmethod
    def create(model):
        if not isinstance(model, torch.nn.Module):
            raise ModelNotSupported(
                f"The provided model ({model}) is not supported by the PyTorch"
                "backend"
            )
        if isinstance(model, PytorchModel):
            return model
        model.__class__ = type("__QuantnnMixin__", (PytorchModel, type(model)), {})
        PytorchModel.__init__(model)
        return model

    @property
    def channel_axis(self):
        """
        The index of the axis that contains the channel information in a batch
        of input data.
        """
        if _has_channels_last_tensor(self.parameters()):
            return -1
        return 1

    def _make_adversarial_samples(self, x, eps):
        """
        Recycles current gradients to perform an adversarial training
        step.

        Args:
            x: The current input.
            eps: Scaling factor for the fast gradient sign method.

        Returns:
            x_adv: Perturbed input tensor representing the adversarial
                example.
        """
        x_adv = x.detach() + eps * torch.sign(x.grad.detach())
        return x_adv

    def reset(self):
        """
        Reinitializes the weights of a model.
        """

        def reset_function(module):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.reset_parameters()

        self.apply(reset_function)

    def _train_step(
        self, x, y, loss, adversarial_training, metrics=None, transformation=None
    ):
        """
        Performs a single training step and returns a dictionary of
        losses for every output.

        Args:
            x: The input data for the current batch.
            y: The output data for the current batch.
            adversarial_training: Scaling factor for adversarial training or
                None if no adversarial training should be performed.

        Returns:
            A single loss or a dictionary of losses in the case of a multi-output
            network.
        """
        y_pred = self(x)

        # Keep track of x gradients if adversarial training
        # is to be performed.
        if adversarial_training is not None:
            x.requires_grad = True

        # Make sure both outputs are dicts.
        if not isinstance(y_pred, dict):
            y_pred = {"__loss__": y_pred}
        if not isinstance(y, dict):
            y = {next(iter(y_pred.keys())): y}
        if not isinstance(loss, dict):
            loss = {k: loss for k in y_pred}

        avg_loss = None
        tot_loss = None
        losses = {}

        # Loop over keys in prediction.
        for k in y_pred:

            loss_k = loss[k]
            y_pred_k = y_pred[k]

            if loss_k.mask is not None:
                mask = torch.tensor(loss_k.mask).to(
                    dtype=y_pred_k.dtype,
                    device=y_pred_k.device
                )
            else:
                mask = None

            if isinstance(transformation, dict):
                transform_k = transformation[k]
            else:
                transform_k = transformation

            try:
                y_k = y[k]
            except KeyError:
                raise DatasetError(f"No targets provided for ouput '{k}'.")

            if y_k.ndim < y_pred_k.ndim:
                y_k = torch.unsqueeze(y_k, 1)

            if transform_k is None:
                y_k_t = y_k
                y_pred_k_t = y_pred_k
            else:
                y_k_t = transform_k(y_k)
                if mask is not None:
                    y_k_t = torch.where(y_k > mask, y_k_t, mask)
                y_pred_k_t = transform_k.invert(y_pred_k)

            if k == "__loss__":
                l = loss_k(y_pred_k, y_k_t)
            else:
                l = loss_k(y_pred_k, y_k_t, k)

            cache = {}
            with torch.no_grad():
                if metrics is not None:
                    for m in metrics:
                        m.process_batch(k, y_pred_k_t, y_k, cache=cache)

            losses[k] = l.item()
            if not avg_loss:
                avg_loss = 0.0
                tot_loss = 0.0
                n_samples = 0
            avg_loss += l
            if mask is not None:
                n = (y_k > mask).sum().item()
            else:
                n = torch.numel(y_k)
            tot_loss += (l * n).item()
            n_samples += n

        return avg_loss, tot_loss, losses, n_samples

    def train(
        self,
        training_data,
        validation_data=None,
        loss=None,
        optimizer=None,
        scheduler="default",
        n_epochs=None,
        adversarial_training=None,
        batch_size=None,
        device="cpu",
        logger=None,
        metrics=None,
        keys=None,
        transformation=None,
    ):
        """
        Train the network.

        This trains the network for the given number of epochs using the
        provided training and validation data.

        If desired, the training can be augmented using adversarial training.
        In this case the network is additionally trained with an adversarial
        batch of examples in each step of the training.

        Arguments:
            training_data: pytorch dataloader providing the training data
            validation_data: pytorch dataloader providing the validation data
            n_epochs: the number of epochs to train the network for
            adversarial_training: whether or not to use adversarial training
            eps_adv: The scaling factor to use for adversarial training.
        """
        # Avoid nameclash with PyTorch train method.
        if type(training_data) == bool:
            return nn.Module.train(self, training_data)

        if logger is None:
            logger = TrainingLogger(n_epochs)

        # Determine device to use
        if torch.cuda.is_available() and device in ["gpu", "cuda"]:
            device = torch.device("cuda")
        elif device == "cpu":
            device = torch.device("cpu")
        else:
            device = torch.device(device)

        # Handle input data
        try:
            x, y = handle_input(training_data, device)
            training_data = BatchedDataset((x, y), batch_size=batch_size)
        except:
            pass

        try:
            x, y = handle_input(validation_data, device)
            validation_data = BatchedDataset((x, y), batch_size=batch_size)
        except Exception:
            pass

        # Optimizer
        if not optimizer:
            optimizer = _get_default_optimizer(self)
        self.optimizer = optimizer

        # Training scheduler
        if scheduler == "default":
            scheduler = _get_default_scheduler(optimizer)
        if scheduler is not None:
            scheduler_sig = signature(scheduler.step)
        else:
            scheduler_sig = None

        self.to(device)

        if isinstance(loss, dict):
            {k: loss[k].to(device) for k in loss}
        else:
            loss.to(device)

        # metrics
        if metrics is None:
            metrics = []

        training_losses = []
        validation_losses = []

        state = {}
        for m in self.modules():
            state[m] = m.training

        # Training loop
        with logger:
            for i in range(n_epochs):

                for m in metrics:
                    m.reset()

                epoch_error = 0.0
                n = 0

                logger.epoch_begin(self)

                for j, data in enumerate(training_data):

                    self.optimizer.zero_grad(**_ZERO_GRAD_ARGS)

                    x, y = _get_x_y(data, keys)

                    if not isinstance(x, torch.Tensor):
                        if isinstance(x, Iterable):
                            x = [x_i.float().to(device) for x_i in x]
                        else:
                            raise ValueError(
                                "Batch input 'x' should be a torch.Tensor or"
                                " an Iterable of tensors."
                            )
                    else:
                        x = x.float().to(device)

                    if isinstance(y, dict):
                        for k in y:
                            y[k] = y[k].to(device)
                    else:
                        y = y.to(device)
                    if adversarial_training is not None:
                        x.requires_grad = True

                    avg_loss, tot_loss, losses, n_samples = self._train_step(
                        x, y, loss, adversarial_training,
                        transformation=transformation
                    )
                    avg_loss.backward()
                    optimizer.step()

                    # Log training step.
                    if hasattr(training_data, "__len__"):
                        of = len(training_data)
                    else:
                        of = None
                    if n_samples > 0:
                        tot_loss /= n_samples
                    logger.training_step(
                        tot_loss, n_samples, of=of, losses=losses
                    )

                    # Track epoch error.
                    n += get_batch_size(x)
                    epoch_error += tot_loss * n

                    # Perform adversarial training step if required.
                    if adversarial_training is not None and x.requires_grad:
                        x_adv = self._make_adversarial_samples(x, adversarial_training)
                        self.optimizer.zero_grad(**_ZERO_GRAD_ARGS)
                        tot_loss, _ = self._train_step(
                            x_adv,
                            y,
                            loss,
                            adversarial_training,
                            transformation=transformation,
                        )
                        tot_loss.backward()
                        optimizer.step()

                # Save training error
                training_losses.append(epoch_error / n)

                lr = [group["lr"] for group in self.optimizer.param_groups][0]

                errors = {}

                if validation_data is not None:
                    n = 0
                    epoch_error = 0
                    self.eval()
                    with torch.no_grad():
                        for j, data in enumerate(validation_data):

                            self.optimizer.zero_grad(**_ZERO_GRAD_ARGS)

                            x, y = _get_x_y(data, keys)
                            if not isinstance(x, torch.Tensor):
                                if isinstance(x, Iterable):
                                    x = [x_i.float().to(device) for x_i in x]
                                else:
                                    raise ValueError(
                                        "Batch input 'x' should be a torch.Tensor or"
                                        " an Iterable of tensors."
                                    )
                            else:
                                x = x.float().to(device)
                            if isinstance(y, dict):
                                for k in y:
                                    y[k] = y[k].to(device)
                            else:
                                y = y.to(device)

                            avg_loss, tot_loss, losses, n_samples = self._train_step(
                                x,
                                y,
                                loss,
                                None,
                                metrics=metrics,
                                transformation=transformation,
                            )

                            # Log validation step.
                            if hasattr(validation_data, "__len__"):
                                of = len(validation_data)
                            else:
                                of = None
                            if n_samples > 0:
                                tot_loss /= n_samples

                            logger.validation_step(
                                tot_loss, n_samples, of=of, losses=losses
                            )

                            # Update running validation errors.
                            n += get_batch_size(x)
                            epoch_error = tot_loss * n

                        validation_losses.append(epoch_error / n)

                        for m in self.modules():
                            m.training = state[m]

                # Finally update scheduler.
                if scheduler:
                    if len(scheduler_sig.parameters) == 1:
                        scheduler.step()
                    else:
                        scheduler.step(epoch_error)

                logger.epoch(learning_rate=lr, metrics=metrics)
        logger.training_end()

        self.eval()

    def predict(self, x, device="cpu"):
        """
        Evaluate the model.

        Args:
            x: The input data for which to evaluate the data.
            device: The device on which to evaluate the prediction.

        Returns:
            The model prediction converted to numpy array.
        """
        # Determine device to use
        w = next(iter(self.parameters())).data
        if isinstance(x, torch.Tensor):
            x_torch = x
        else:
            x_torch = to_array(torch, x, like=w)
        self.to(x_torch.device)

        if x_torch.requires_grad:
            y = self(x_torch)
        else:
            with torch.no_grad():
                y = self(x_torch)

        return y

    def calibration(self, data, gpu=False):
        """
        Computes the calibration of the predictions from the neural network.

        Arguments:
            data: torch dataloader object providing the data for which to compute
                the calibration.

        Returns:
            (intervals, frequencies): Tuple containing the confidence intervals and
                corresponding observed frequencies.
        """

        if gpu and torch.cuda.is_available():
            dev = torch.device("cuda")
        else:
            dev = torch.device("cpu")
        self.to(dev)

        n_intervals = self.quantiles.size // 2
        qs = self.quantiles
        intervals = np.array([q_r - q_l for (q_l, q_r) in zip(qs, reversed(qs))])[
            :n_intervals
        ]
        counts = np.zeros(n_intervals)

        total = 0.0

        for x, y in iterator:
            x = x.to(dev).detach()
            y = y.to(dev).detach()
            shape = x.size()
            shape = (shape[0], 1) + shape[2:]
            y = y.reshape(shape)

            y_pred = self(x)
            y_pred = y_pred.cpu()
            y = y.cpu()

            for i in range(n_intervals):
                l = y_pred[:, [i]]
                r = y_pred[:, [-(i + 1)]]
                counts[i] += np.logical_and(y >= l, y < r).sum()

            total += np.prod(y.size())
        return intervals[::-1], (counts / total)[::-1]

    def save(self, path):
        """
        Save QRNN to file.

        Arguments:
            The path in which to store the QRNN.
        """
        torch.save(
            {
                "width": self.width,
                "depth": self.depth,
                "activation": self.activation,
                "network_state": self.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
            },
            path,
        )

    @staticmethod
    def load(self, path):
        """
        Load QRNN from file.

        Arguments:
            path: Path of the file where the QRNN was stored.
        """
        state = torch.load(path, map_location=torch.device("cpu"))
        keys = ["depth", "width", "activation"]
        qrnn = QRNN(*[state[k] for k in keys])
        qrnn.load_state_dict["network_state"]
        qrnn.optimizer.load_state_dict["optimizer_state"]
