"""
quantnn.models.pytorch.common
=============================

This module provides common functionality required to realize QRNNs in pytorch.
"""
from inspect import signature
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

from quantnn.common import ModelNotSupported

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


def save_model(f, model):
    """
    Save pytorch model.

    Args:
        f(:code:`str` or binary stream): Either a path or a binary stream
            to store the data to.
        model(:code:`pytorch.nn.Moduel`): The pytorch model to save
    """
    path = tempfile.mkdtemp()
    filename = os.path.join(path, "keras_model.h5")
    torch.save(model, filename)
    archive = tarfile.TarFile(fileobj=f, mode="w")
    archive.add(filename, arcname="keras_model.h5")
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
    tar_file.extract("keras_model.h5", path=path)
    filename = os.path.join(path, "keras_model.h5")
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
        if not device is None:
            x = x.to(device)
            y = y.to(device)
        return x, y
    if type(data) == np.ndarray:
        x = torch.tensor(data, dtype=torch.float)
        if not device is None:
            x = x.to(device)
        return x
    else:
        return data


class BatchedDataset(Dataset):
    """
    Batches an un-batched dataset.
    """
    def __init__(self, training_data, batch_size=None):
        x, y = training_data

        # x
        if isinstance(x, torch.Tensor):
            self.x = x.clone().detach().float()
        else:
            self.x = torch.tensor(x, dtype=torch.float)

        # y
        dtype_y = torch.float
        if "int" in str(y.dtype):
            dtype_y = torch.long
        if isinstance(y, torch.Tensor):
            self.y = y.clone().detach().to(dtype=dtype_y)
        else:
            self.y = torch.tensor(y, dtype=dtype_y)

        if batch_size:
            self.batch_size = batch_size
        else:
            self.batch_size = 256

    def __len__(self):
        # This is required because x and y are tensors and don't throw these
        # errors themselves.
        return self.x.shape[0] // self.batch_size

    def __getitem__(self, i):
        if i >= len(self):
            raise IndexError()
        i_start = i * self.batch_size
        i_end = (i + 1) * self.batch_size
        x = self.x[i_start:i_end]
        y = self.y[i_start:i_end]
        return (x, y)


################################################################################
# Quantile loss
################################################################################

class CrossEntropyLoss(nn.CrossEntropyLoss):

    def __init__(self):
        super().__init__()

    def __call__(self, y_pred, y):
        return nn.CrossEntropyLoss.__call__(
            self,
            y_pred,
            y.flatten()
        )

class QuantileLoss:
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

    def __init__(self,
                 quantiles,
                 mask=None,
                 quantile_axis=1):
        """
        Create an instance of the quantile loss function with the given quantiles.

        Arguments:
            quantiles: Array or iterable containing the quantiles to be estimated.
        """
        self.quantiles = torch.tensor(quantiles).float()
        self.n_quantiles = len(quantiles)
        self.mask = mask
        if self.mask:
            self.mask = np.float32(mask)
        self.quantile_axis = quantile_axis

    def to(self, device):
        self.quantiles = self.quantiles.to(device)

    def __call__(self, y_pred, y_true):
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
        dy = y_pred - y_true
        n = self.quantiles.size()[0]

        shape = [1,] * len(dy.size())
        shape[self.quantile_axis] = self.n_quantiles
        qs = self.quantiles.reshape(shape)
        l = torch.where(dy >= 0.0, (1.0 - qs) * dy, (-qs) * dy)
        if self.mask:
            mask = y_true > self.mask
            return (l * mask).sum() / (mask.sum() * self.n_quantiles)
        return l.mean()

################################################################################
# Default scheduler and optimizer
################################################################################

def _get_default_optimizer(model):
    """
    The default optimizer. Currently set to Adam optimizer.
    """
    optimizer = optim.Adam(model.parameters())
    return optimizer

def _get_default_scheduler(optimizer):
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     factor=0.1,
                                                     patience=5)
    return scheduler

################################################################################
# QRNN
################################################################################

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
                "backend")
        if isinstance(model, PytorchModel):
            return model
        new_model = PytorchModel()
        model.__class__ = type("__QuantnnMixin__", (PytorchModel, type(model)), {})
        PytorchModel.__init__(model)
        return model

    def __init__(self):
        """
        Arguments:
            input_dimension(int): The number of input features.
            quantiles(array): Array of the quantiles to predict.
        """
        self.training_errors = []
        self.validation_errors = []

    def _make_adversarial_samples(self, x, y, eps):
        self.zero_grad()
        x.requires_grad = True
        y_pred = self(x)
        c = self.criterion(y_pred, y)
        c.backward()
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

    def train(self,
              training_data,
              validation_data=None,
              loss=None,
              optimizer=None,
              scheduler=None,
              n_epochs=None,
              adversarial_training=None,
              batch_size=None,
              device='cpu'):
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
        # Avoid nameclash with Pytorch train method.
        if type(training_data) == bool:
            return nn.Module.train(self, training_data)

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

        # Optimizer
        if not optimizer:
            optimizer = _get_default_optimizer(self)
        self.optimizer = optimizer

        # Training scheduler
        if not scheduler:
            scheduler = _get_default_scheduler(optimizer)

        loss.to(device)
        self.to(device)
        scheduler_sig = signature(scheduler.step)
        training_errors = []
        validation_errors = []

        # Training loop
        for i in range(n_epochs):
            error = 0.0
            n = 0
            for j, (x, y) in enumerate(training_data):

                x = x.to(device)
                y = y.to(device)

                shape = x.size()
                shape = (shape[0], 1) + shape[2:]
                y = y.reshape(shape)

                self.optimizer.zero_grad()
                y_pred = self(x)
                c = loss(y_pred, y)
                c.backward()
                self.optimizer.step()

                error += c.item() * x.size()[0]
                n += x.size()[0]

                if adversarial_training:
                    self.optimizer.zero_grad()
                    x_adv = self._make_adversarial_samples(x, y, adversarial_training)
                    y_pred = self(x_adv)
                    c = loss(y_pred, y)
                    c.backward()
                    self.optimizer.step()

                if j % 100:
                    try:
                        print(
                            "Epoch {} / {}: Batch {} / {}, Training error: {:.3f}".format(
                                i, n_epochs, j, len(training_data), error / n
                            ),
                            end="\r",
                        )
                    except TypeError:
                        pass


            # Save training error
            training_errors.append(error / n)

            lr = [group["lr"] for group in self.optimizer.param_groups][0]

            validation_error = 0.0
            if not validation_data is None:
                n = 0
                self.eval()
                for x, y in validation_data:
                    x = x.to(device).detach()
                    y = y.to(device).detach()

                    shape = x.size()
                    shape = (shape[0], 1) + shape[2:]
                    y = y.reshape(shape)

                    y_pred = self(x)
                    c = loss(y_pred, y)

                    validation_error += c.item() * x.size()[0]
                    n += x.size()[0]
                validation_errors.append(validation_error / n)
                nn.Module.train(self, True)

                print(
                    f"Epoch {i} / {n_epochs}: "
                    f"Training error: {training_errors[-1]:.4f}, "
                    f"Validation error: {validation_errors[-1]:.4f}, "
                    f"Learning rate: {lr:.5f}"
                )

                if scheduler:
                    if len(scheduler_sig.parameters) == 1:
                        scheduler.step()
                    else:
                        if validation_data:
                            scheduler.step(validation_errors[-1])

            else:
                if scheduler:
                    if len(scheduler_sig.parameters) == 1:
                        scheduler.step()
                    else:
                        if validation_data:
                            scheduler.step(training_errors[-1])

                print(
                    f"Epoch {i} / {n_epochs}: "
                    f"Training error: {training_errors[-1]:.4f}, "
                    f"Learning rate: {lr:.5f}"
                )

        self.training_errors += training_errors
        self.validation_errors += validation_errors
        self.eval()
        return {
            "training_errors": self.training_errors,
            "validation_errors": self.validation_errors,
        }

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
        if torch.cuda.is_available() and device in ["gpu", "cuda"]:
            device = torch.device("cuda")
        elif device == "cpu":
            device = torch.device("cpu")
        else:
            device = torch.device(device)
        if torch.cuda.is_available() and device in ["cuda", "gpu"]:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        x = handle_input(x, device)
        self.to(device)
        return self(x.detach()).detach().numpy()

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
