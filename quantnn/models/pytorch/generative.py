import torch
from torch import nn
from torch.nn.functional import relu

from quantnn.models.pytorch.xception import SymmetricPadding


class DownBlock(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.norm = nn.BatchNorm2d(channels_in)
        self.body = nn.Sequential(
            nn.ReLU(),
            SymmetricPadding(1),
            nn.Conv2d(channels_in, channels_in, 3),
            nn.BatchNorm2d(channels_in),
            SymmetricPadding(1),
            nn.Conv2d(channels_in, channels_out, 3),
            nn.ReLU(),
            SymmetricPadding(1),
            nn.MaxPool2d(3, stride=2)
        )
        self.skip = nn.Sequential(
            SymmetricPadding(1),
            nn.Conv2d(channels_in, channels_out, 1),
            nn.MaxPool2d(3, stride=2)
        )

    def forward(self, x):
        x_n = self.norm(x)
        y = self.body(x_n)
        return y + self.skip(x_n)


class DownBlockSpectral(nn.Module):
    def __init__(self, channels_in, channels_out, relu_in=True):
        super().__init__()
        if relu_in:
            self.body = nn.Sequential(
                nn.ReLU(),
                SymmetricPadding([1, 2, 1, 2]),
                nn.utils.spectral_norm(nn.Conv2d(channels_in, channels_in, 4)),
                #nn.BatchNorm2d(channels_out),
                nn.ReLU(),
                SymmetricPadding([1, 2, 1, 2]),
                nn.utils.spectral_norm(nn.Conv2d(channels_in, channels_out, 4)),
                nn.AvgPool2d(2)
            )
        else:
            self.body = nn.Sequential(
                SymmetricPadding([1, 2, 1, 2]),
                nn.utils.spectral_norm(nn.Conv2d(channels_in, channels_in, 4)),
                #nn.BatchNorm2d(channels_out),
                nn.ReLU(),
                SymmetricPadding([1, 2, 1, 2]),
                nn.utils.spectral_norm(nn.Conv2d(channels_in, channels_out, 4)),
                nn.AvgPool2d(2)
            )

        self.skip = nn.Sequential(
            SymmetricPadding([0, 1, 0, 1]),
            nn.utils.spectral_norm(nn.Conv2d(channels_in, channels_out, 2)),
            nn.AvgPool2d(2)
        )

    def forward(self, x):
        y = self.body(x)
        return y + self.skip(x)


class GeneratorBlock(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.body = nn.Sequential(
            nn.BatchNorm2d(channels_in),
            nn.LeakyReLU(0.1),
            SymmetricPadding([1, 2, 1, 2]),
            nn.utils.spectral_norm(nn.Conv2d(channels_in, channels_out, 4)),
            nn.BatchNorm2d(channels_out),
            nn.LeakyReLU(0.1),
            SymmetricPadding([1, 2, 1, 2]),
            nn.utils.spectral_norm(nn.Conv2d(channels_out, channels_out, 4)),
        )
        self.skip = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 1),
        )

    def forward(self, x):
        return self.body(x) + self.skip(x)

class GeneratorBlockUp(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.body = nn.Sequential(
            nn.BatchNorm2d(channels_in),
            nn.LeakyReLU(0.1),
            nn.Upsample(scale_factor=2, mode="nearest"),
            SymmetricPadding([1, 2, 1, 2]),
            nn.utils.spectral_norm(nn.Conv2d(channels_in, channels_out, 4)),
            nn.BatchNorm2d(channels_out),
            nn.LeakyReLU(0.1),
            SymmetricPadding([1, 2, 1, 2]),
            nn.utils.spectral_norm(nn.Conv2d(channels_out, channels_out, 4)),
        )
        self.skip = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(channels_in, channels_out, 1),
        )

    def forward(self, x):
        return self.body(x) + self.skip(x)


class ConditionalGenerator(nn.Module):
    def __init__(self, channels_in, channels, n_lat=8, output_range=1):
        super().__init__()

        self.n_lat = n_lat
        self.output_range = output_range

        self.conditioner = nn.ModuleList([
            DownBlock(channels_in, channels),
            DownBlock(channels, channels),
            DownBlock(channels, channels),
            DownBlock(channels, channels)
        ])

        self.generator = nn.ModuleList([
            nn.Sequential(
                GeneratorBlock(channels + n_lat, channels),
                GeneratorBlockUp(channels, channels)
            ),
            nn.Sequential(
                GeneratorBlock(channels, channels),
                GeneratorBlockUp(channels, channels)
            ),
            nn.Sequential(
                GeneratorBlock(channels, channels),
                GeneratorBlockUp(channels, channels)
            ),
            nn.Sequential(
                GeneratorBlock(channels, channels),
                GeneratorBlockUp(channels, channels)
            ),
        ])

        self.output = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.1),
            nn.utils.spectral_norm(nn.Conv2d(channels, 1, 1))
        )

    def forward(self, x, z=None):

        y = x
        for layer in self.conditioner:
            y = layer(y)

        n = x.shape[0]

        if z is None:
            z = torch.normal(0, 1, size=(n, self.n_lat, 4, 4))
        else:
            z = z.reshape((n, self.n_lat, 4, 4))

        input = None
        for layer in self.generator:
            if input is None:
                input = layer(torch.cat([y, z], 1))
            else:
                input = layer(input)

        return self.output_range * torch.tanh(self.output(input))


class Discriminator(nn.Module):
    def __init__(self, channels_in, channels=64):
        super().__init__()

        self.body = nn.Sequential(
            DownBlockSpectral(channels_in + 1, channels, relu_in=False),
            DownBlockSpectral(channels, channels),
            DownBlockSpectral(channels, channels),
            DownBlockSpectral(channels, channels),
            DownBlockSpectral(channels, channels),
            nn.ReLU(0.1)
        )

        self.classifier = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(channels, 1)),
        )

    def forward(self, x, y):
        if y.ndim < x.ndim:
            y = y.unsqueeze(1)
        x_in = torch.cat([x, y], 1)
        y = self.body(x_in).sum((-2, -1))
        return self.classifier(y)


def hinge_loss(d_real, d_fake):
    return relu(1.0 - d_real).mean() + relu(1.0 + d_fake).mean()

def training_step(x, y, g, d, opt_g, opt_d):
    y = y.unsqueeze(1)

    batch_size = x.size(0)
    z = torch.randn(batch_size, g.n_lat, 4, 4).cuda()

    # Train discriminator
    opt_d.zero_grad()
    opt_g.zero_grad()
    d_real = d(x, y)
    d_fake = d(x, g(x, z))
    d_loss = hinge_loss(d_real, d_fake)
    d_loss.backward()
    opt_d.step()

    # Train generator
    opt_d.zero_grad()
    opt_g.zero_grad()
    z = torch.randn(batch_size, g.n_lat, 4, 4).cuda()
    fake = g(x, z)
    g_loss = -d(x, fake).mean() 
    mse_loss = ((fake - y) ** 2).mean()
    g_loss = g_loss #+ mse_loss

    g_loss.backward()
    opt_g.step()

    return g_loss.item(), d_loss.item(), mse_loss.item()
