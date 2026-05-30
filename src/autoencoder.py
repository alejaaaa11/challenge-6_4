"""
autoencoder.py — Standard AutoEncoder for anomaly detection.
Group 4 — Challenge 6 — Universidad Distrital Francisco José de Caldas
"""
import torch
import torch.nn as nn
import numpy as np


class AutoEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, latent_dim: int):
        super().__init__()
        enc, dec = [], []
        dims = [input_dim] + hidden_dims + [latent_dim]
        for i in range(len(dims) - 1):
            enc += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
        enc = enc[:-1]  # remove last ReLU from encoder
        for i in range(len(dims) - 1, 0, -1):
            if i > 1:
                dec += [nn.Linear(dims[i], dims[i - 1]), nn.ReLU()]
            else:
                dec += [nn.Linear(dims[i], dims[i - 1])]
        self.encoder = nn.Sequential(*enc)
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


def train_ae(model, loader, epochs=100, lr=1e-3, seed=42):
    """Train AE and return list of epoch losses."""
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    losses = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for (xb,) in loader:
            x_hat, _ = model(xb)
            loss = criterion(x_hat, xb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        losses.append(epoch_loss / len(loader.dataset))
    return losses


def compute_errors(model, X_tensor):
    """Return per-sample MSE reconstruction errors."""
    model.eval()
    with torch.no_grad():
        x_hat, z = model(X_tensor)
        errors = ((X_tensor - x_hat) ** 2).mean(dim=1).numpy()
    return errors, z.numpy()


def select_threshold(train_errors, method='percentile95'):
    """Select anomaly threshold from training errors."""
    if method == 'percentile95':
        return float(np.percentile(train_errors, 95))
    elif method == 'percentile99':
        return float(np.percentile(train_errors, 99))
    elif method == '3sigma':
        return float(train_errors.mean() + 3 * train_errors.std())
    else:
        raise ValueError(f"Unknown method: {method}")
