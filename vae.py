"""Convolutional Variational Autoencoder for chest X-ray image generation.

Architecture (input 1x128x128, grayscale):
  Encoder: 5 conv blocks (stride 2) 1->32->64->128->256->512, spatial 128->4
           fc -> mu, logvar (latent_dim = 128)
  Decoder: fc -> 512*4*4, 5 transposed-conv blocks back to 1x128x128 (logits)
"""
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, in_ch=1, latent_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 4, 2, 1),    # 128 -> 64
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),       # 64 -> 32
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),      # 32 -> 16
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),     # 16 -> 8
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1),     # 8 -> 4
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
        )
        self.fc_mu = nn.Linear(512 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(512 * 4 * 4, latent_dim)

    def forward(self, x):
        h = self.conv(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    def __init__(self, out_ch=1, latent_dim=128):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 512 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),  # 4 -> 8
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 8 -> 16
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 16 -> 32
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),    # 32 -> 64
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, out_ch, 4, 2, 1),  # 64 -> 128 (logits)
        )

    def forward(self, z):
        h = self.fc(z).view(-1, 512, 4, 4)
        return self.deconv(h)


class VAE(nn.Module):
    def __init__(self, in_ch=1, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(in_ch, latent_dim)
        self.decoder = Decoder(in_ch, latent_dim)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    @torch.no_grad()
    def sample(self, n, device):
        z = torch.randn(n, self.latent_dim, device=device)
        return torch.sigmoid(self.decoder(z))

    @torch.no_grad()
    def encode(self, x):
        mu, logvar = self.encoder(x)
        return self.reparameterize(mu, logvar)
