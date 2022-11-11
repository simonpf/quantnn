"""
quantnn.models.pytorch.fully_connected
======================================

This module provides an implementation of fully-connected feed forward
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

        if isinstance(activation, str):
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
        """Propagate input through network."""

        y_p = []
        y_l = self.mods[0](x)

        for layer in self.mods[1:]:
            if self.skips:
                y = torch.cat(y_p + [y_l, x], 1)
                y_p = [y_l]
            else:
                y = y_l
            y_l = layer(y)
        return y_l


class MLPBlock(nn.Module):
    """
    A building block for a fully-connected (MLP) network.

    This block expects the features to be located along the
    last dimension of the tensor.
    """

    def __init__(
        self,
        features_in,
        features_out,
        activation_factory,
        norm_factory,
        residuals=None,
    ):
        """
        Args:
            features_in: The number of features of the block input.
            features_out: The number of features of the block output.
            activation_factory: A factory functional to create the activation
                layers in the block.
            norm_factory: A factory functional to create the normalization
                layers used in the block.
            residuals: The type of residuals to apply.
        """
        super().__init__()
        if residuals is not None:
            residuals = residuals.lower()
        self.residuals = residuals
        self.body = nn.Sequential(
            nn.Linear(features_in, features_out, bias=False),
            norm_factory(features_out),
            activation_factory(),
        )

    def forward(self, x):
        """
        Propagate input through the block.

        Args:
            x: The input tensor for residuals 'None' or 'simple'. Or a tuple
                ``(x, acc, n_acc)`` containing the input ``x``, the
                accumulation buffer ``acc`` and the accumulation counter
                ``n_acc``.

        Return:
            If residuals is 'None' or 'simple', the output is just the output
            tensor ``y``. If residuals is `hyper` the output is a tuple
            ``(y, acc, n_acc)`` containing the block output ``y``, the
            accumulation buffer ``acc`` and the accumulation counter
            ``n_acc``.
        """
        if self.residuals is None:
            return self.body(x)

        if self.residuals == "simple":
            y = self.body(x)
            n = min(x.shape[-1], y.shape[-1])
            y[..., :n] += x[..., :n]
            return y

        if isinstance(x, tuple):
            x, acc, n_acc = x
            acc = acc.clone()
            n = min(acc.shape[-1], x.shape[-1])
            acc[..., :n] += x[..., :n]
            n_acc += 1
        else:
            acc = x.clone()
            n_acc = 1
            n = x.shape[-1]
        y = self.body(x)
        n = min(y.shape[-1], n)
        y[..., :n] += acc[..., :n] / n_acc
        return y, acc, n_acc


class MLP(nn.Module):
    """
    A fully-connected feed-forward neural network.

    The MLP can be used both as a fully-connected on 2D data as well
    as a module in a CNN. When used with 4D output the input is
    automatically permuted so that features are oriented along the last
    dimension of the input tensor.
    """

    def __init__(
        self,
        features_in,
        n_features,
        features_out,
        n_layers,
        residuals=None,
        activation_factory=nn.ReLU,
        norm_factory=nn.BatchNorm1d,
        internal=False,
    ):
        """
        Create MLP module.

        Args:
            features_in: Number of features in the input.
            n_features: Number of features of the hidden layers.
            features_out: Number of features of the output.
            n_layers: The number of layers.
            residuals: The type of residual connections in the MLP:
                None, 'simple', or 'hyper'.
            activation_factory: Factory functional to instantiate the activation
                functions to use in the MLP.
            norm_factory: Factory functional to instantiate the normalization
                layers to use in the MLP.
            internal: If the module is not an 'internal' module no
                 normalization or activation function are applied to the
                 output.
        """
        super().__init__()
        self.n_layers = n_layers
        if residuals is not None:
            residuals = residuals.lower()
        self.residuals = residuals

        self.layers = nn.ModuleList()
        for _ in range(n_layers - 1):
            self.layers.append(
                MLPBlock(
                    features_in=features_in,
                    features_out=n_features,
                    activation_factory=activation_factory,
                    norm_factory=norm_factory,
                    residuals=self.residuals,
                )
            )
            features_in = n_features
        if n_layers > 0:
            if internal:
                self.output_layer = nn.Sequential(
                    MLPBlock(
                        features_in=n_features,
                        features_out=features_out,
                        activation_factory=activation_factory,
                        norm_factory=norm_factory,
                        residuals=self.residuals,
                    )
                )
            else:
                self.output_layer = nn.Linear(n_features, features_out)

    def forward(self, x):
        """
        Forward input through network.

        Args:
            x: The 4D or 2D input tensor to propagate through
                the network.

        Return:
            The output tensor.
        """
        needs_reshape = False
        if x.ndim == 4:
            needs_reshape = True
            x = torch.permute(x, (0, 2, 3, 1))
            old_shape = x.shape
            x = x.reshape(-1, old_shape[-1])

        if self.n_layers == 0:
            return x, None

        y = x
        for l in self.layers:
            y = l(y)
        if self.residuals == "hyper":
            y = y[0]
        y = self.output_layer(y)

        if needs_reshape:
            y = y.view(old_shape[:-1] + (-1,))
            y = torch.permute(y, (0, 3, 1, 2))
        return y
