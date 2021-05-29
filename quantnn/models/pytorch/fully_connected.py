"""
quantnn.models.pytorch.fully_connected
======================================

This module provides an implementation of a fully-connected feed forward
neural networks in pytorch.
"""
from torch import nn
import torch
from quantnn.models.pytorch.common import PytorchModel, activations

###############################################################################
# Fully-connected neural network model.
###############################################################################


class FullyConnectedBlock(nn.Sequential):
    """
    Building block for fully-connected network. Consists of fully-connected
    layer followed by an optional batch norm layer and the activation.
    """

    def __init__(self, n_inputs, n_outputs, activation, batch_norm=True):
        """
        Create block.

        Args:
             n_inputs: The number of input features of the block.
             n_outputs: The number of outputs of the block.
             activation: The activation function to use.
             batch_norm: Whether or not to include a batch norm layer
                         in the block.
        """
        modules = [nn.Linear(n_inputs, n_outputs)]
        if batch_norm:
            modules.append(nn.BatchNorm1d(n_outputs))
        modules.append(activation())
        super().__init__(*modules)


class FullyConnected(PytorchModel, nn.Module):
    """
    A fully-connected neural network model.
    """

    def __init__(
        self,
        n_inputs,
        n_outputs,
        n_layers,
        width,
        activation=nn.ReLU,
        batch_norm=False,
        skip_connections=False,
    ):
        """
        Create a fully-connect neural network model.

        Args:
            n_inputs: The number of input features to the network.
            n_outputs: The number of outputs of the model.
            layers: The number of hidden layers in the model.
            width: The number of neurons in the hidden layers.
            activation: The activation function to use in the hidden
                  layers.
            batch_norm: Whether to include a batch-norm layer after
                 each hidden layer.
        """
        self.skips = skip_connections

        super().__init__()
        nn.Module.__init__(self)

        if type(activation) == str:
            activation = activations[activation]

        nominal_width = width
        if self.skips:
            nominal_width = (nominal_width - n_inputs) // 2

        n_in = n_inputs
        n_out = nominal_width

        modules = []
        for i in range(n_layers):
            modules.append(
                FullyConnectedBlock(n_in, n_out, activation, batch_norm=batch_norm)
            )
            if self.skips:
                if i == 0:
                    n_in = n_out + n_inputs
                else:
                    n_in = 2 * n_out + n_inputs
            else:
                n_in = n_out

        modules.append(nn.Linear(n_in, n_outputs))
        self.mods = nn.ModuleList(modules)

    def forward(self, x):
        """ Propagate input through network. """

        y_p = []
        y_l = self.mods[0](x)

        for l in self.mods[1:]:
            if self.skips:
                y = torch.cat(y_p + [y_l, x], 1)
                y_p = [y_l]
            else:
                y = y_l
            y_l = l(y)
        return y_l
