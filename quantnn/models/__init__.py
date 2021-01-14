"""
quantnn.models.pytorch
======================

This moudles provides Pytorch neural network models that can be used a backend
for the :py:class:`quantnn.QRNN` class.
"""
from quantnn.models.pytorch.common import (
    BatchedDataset,
    save_model,
    load_model,
)
from torch.nn import CrossEntropyLoss
from quantnn.models.pytorch.fully_connected import FullyConnected
from quantnn.models.pytorch.unet import UNet
