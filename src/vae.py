"""
vae.py — Variational AutoEncoder for representation learning.
Group 4 — Challenge 6 — Universidad Distrital Francisco José de Caldas
"""
import torch
import torch.nn as nn
import numpy as np


class VAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, latent_dim: int):
        super().__init__()
        enc = []
        prev = input_dim
        for h in hidden_dims:
            enc += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.encoder_body = nn.Sequential(*enc)
        self.fc_mu     = nn.Linear(prev, latent_dim)
        self.fc_logvar = nn.Linear(prev, latent_dim)
        dec = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        dec.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec)

    def encode(self, x):
        h = self.encoder_body(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterise(self, mu, logvar):
        std = (0.5 * logvar).exp()
        return mu + std * torch.randn_like(std)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        return self.decoder(z), mu, logvar


def vae_loss(x_hat, x, mu, logvar, beta=1.0):
    """ELBO loss: reconstruction + beta * KL divergence."""
    recon = nn.functional.mse_loss(x_hat, x, reduction='sum')
    kld   = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total = (recon + beta * kld) / x.size(0)
    return total, recon.item() / x.size(0), kld.item() / x.size(0)


def train_vae(model, loader, epochs=100, lr=1e-3, beta=1.0, seed=42):
    """Train VAE with KL warmup; return (recon_losses, kl_losses)."""
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    recon_losses, kl_losses = [], []
    for epoch in range(epochs):
        model.train()
        epoch_recon, epoch_kl = 0.0, 0.0
        beta_t = min(1.0, epoch / 30) * beta  # warmup
        for (xb,) in loader:
            x_hat, mu, logvar = model(xb)
            loss, r, k = vae_loss(x_hat, xb, mu, logvar, beta=beta_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_recon += r * len(xb)
            epoch_kl    += k * len(xb)
        recon_losses.append(epoch_recon / len(loader.dataset))
        kl_losses.append(epoch_kl / len(loader.dataset))
    return recon_losses, kl_losses


def get_latent_means(model, X_tensor):
    """Return mu vectors and reconstruction errors for all samples."""
    model.eval()
    with torch.no_grad():
        x_hat, mu, logvar = model(X_tensor)
        errors = ((X_tensor - x_hat) ** 2).mean(dim=1).numpy()
    return mu.numpy(), errors
