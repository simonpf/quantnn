import torch

from quantnn.models.pytorch.fully_connected import MLP

def test_mlp():

    #
    # 2D Input
    #

    x = torch.rand(128, 8)

    # No residual connections.
    mlp = MLP(
        features_in=8,
        n_features=128,
        features_out=16,
        n_layers=4,
    )
    y = mlp(x)
    assert y.shape == (128, 16)

    # Standard residual connections.
    mlp = MLP(
        features_in=8,
        n_features=128,
        features_out=16,
        n_layers=4,
        residuals="simple"
    )
    y = mlp(x)
    assert y.shape == (128, 16)

    # Hyper residual connections.
    mlp = MLP(
        features_in=8,
        n_features=128,
        features_out=16,
        n_layers=4,
        residuals="hyper"
    )
    y = mlp(x)
    assert y.shape == (128, 16)

    #
    # 4D input
    #
    x = torch.rand(128, 8, 8, 8)

    # No residual connections.
    mlp = MLP(
        features_in=8,
        n_features=128,
        features_out=16,
        n_layers=4,
    )
    y = mlp(x)
    assert y.shape == (128, 16, 8, 8)

    # Standard residual connections.
    mlp = MLP(
        features_in=8,
        n_features=128,
        features_out=16,
        n_layers=4,
        residuals="simple"
    )
    y = mlp(x)
    assert y.shape == (128, 16, 8, 8)

    # Hyper residual connections.
    mlp = MLP(
        features_in=8,
        n_features=128,
        features_out=16,
        n_layers=4,
        residuals="hyper"
    )
    y = mlp(x)
    assert y.shape == (128, 16, 8, 8)
