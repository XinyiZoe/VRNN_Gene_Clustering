### File: models.py

import torch
import torch.nn as nn


class VRNNCell(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim):
        super(VRNNCell, self).__init__()
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim

        self.lstm = nn.LSTMCell(x_dim + z_dim, h_dim)

        self.encoder = nn.Sequential(nn.Linear(h_dim + x_dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.enc_mean = nn.Linear(64, z_dim)
        self.enc_logvar = nn.Linear(64, z_dim)

        self.prior = nn.Sequential(nn.Linear(h_dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.prior_mean = nn.Linear(64, z_dim)
        self.prior_logvar = nn.Linear(64, z_dim)

        self.decoder = nn.Sequential(
            nn.Linear(h_dim + z_dim, 64),
            nn.ReLU(),
            nn.Linear(64, x_dim),
        )

    def forward(self, x_t, h_tm1, c_tm1):
        enc_input = torch.cat([h_tm1, x_t], dim=1)
        enc_hidden = self.encoder(enc_input)
        mu_enc = self.enc_mean(enc_hidden)
        logvar_enc = self.enc_logvar(enc_hidden)
        z_t = self.reparameterize(mu_enc, logvar_enc)

        prior_hidden = self.prior(h_tm1)
        mu_prior = self.prior_mean(prior_hidden)
        logvar_prior = self.prior_logvar(prior_hidden)

        lstm_input = torch.cat([x_t, z_t], dim=1)
        h_t, c_t = self.lstm(lstm_input, (h_tm1, c_tm1))

        x_mean = self.decoder(torch.cat([h_t, z_t], dim=1))
        x_logvar = torch.zeros_like(x_mean)

        return x_mean, x_logvar, z_t, mu_enc, logvar_enc, mu_prior, logvar_prior, h_t, c_t

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class VRNN(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim):
        super(VRNN, self).__init__()
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.cell = VRNNCell(x_dim, h_dim, z_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        h_t = torch.zeros(batch_size, self.h_dim, device=x.device)
        c_t = torch.zeros(batch_size, self.h_dim, device=x.device)

        x_means, x_logvars, z_samples = [], [], []
        mu_z_encs, logvar_z_encs, mu_z_pris, logvar_z_pris = [], [], [], []

        for t in range(seq_len):
            x_t = x[:, t, :]
            x_mean, x_logvar, z_t, mu_enc, logvar_enc, mu_prior, logvar_prior, h_t, c_t = self.cell(x_t, h_t, c_t)
            x_means.append(x_mean)
            x_logvars.append(x_logvar)
            z_samples.append(z_t)
            mu_z_encs.append(mu_enc)
            logvar_z_encs.append(logvar_enc)
            mu_z_pris.append(mu_prior)
            logvar_z_pris.append(logvar_prior)

        return {
            "x_mean": torch.stack(x_means, dim=1),
            "x_logvar": torch.stack(x_logvars, dim=1),
            "z": torch.stack(z_samples, dim=1),
            "mu_z_enc": torch.stack(mu_z_encs, dim=1),
            "logvar_z_enc": torch.stack(logvar_z_encs, dim=1),
            "mu_z_pri": torch.stack(mu_z_pris, dim=1),
            "logvar_z_pri": torch.stack(logvar_z_pris, dim=1),
        }


def vrnn_loss_function(outputs, x, epoch, beta=1e-6, warmup_steps=1):
    x_mean, x_logvar = outputs["x_mean"], outputs["x_logvar"]
    mu_z_enc, logvar_z_enc = outputs["mu_z_enc"], outputs["logvar_z_enc"]
    mu_z_pri, logvar_z_pri = outputs["mu_z_pri"], outputs["logvar_z_pri"]

    recon_nll = 0.5 * x_logvar + (x - x_mean) ** 2 / (2.0 * torch.exp(x_logvar))
    recon_nll = torch.sum(recon_nll, dim=2)

    kl = (
        0.5 * (logvar_z_pri - logvar_z_enc)
        + (torch.exp(logvar_z_enc) + (mu_z_enc - mu_z_pri) ** 2) / (2.0 * torch.exp(logvar_z_pri))
        - 0.5
    )
    kl = torch.sum(kl, dim=2)

    if warmup_steps > 0:
        recon_nll = recon_nll[:, warmup_steps:]
        kl = kl[:, warmup_steps:]

    loss_t = recon_nll + beta * kl
    return torch.mean(torch.sum(loss_t, dim=1))
