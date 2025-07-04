import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierEmbedding(torch.nn.Module):
    def __init__(self, num_channels, scale=16):
        super().__init__()
        self.register_buffer('freqs', torch.randn(num_channels // 2) * scale)

    def forward(self, x):
        x = torch.outer(x, (2 * np.pi * self.freqs).to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x
        
class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels
        freqs = torch.linspace(0, 0.5, num_channels//2)
        freqs = (1 / 10000) ** freqs
        self.register_buffer('freqs', freqs)

    def forward(self, x):
        x = torch.outer(x, self.freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x
        
class ToyNet(nn.Module):
    def __init__(self, data_dim, cond_dim=0, traj_mode=False, hidden_dims=[32, 64, 64, 32]):
        super().__init__()
        self.layers = torch.nn.ModuleList([nn.Linear(data_dim, hidden_dims[0])])
        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
        self.layers.append(nn.Linear(hidden_dims[-1], data_dim))
        self.t_emb_layers = torch.nn.ModuleList([PositionalEmbedding(hidden_dims[0])])
        for i in range(len(hidden_dims) - 1):
            self.t_emb_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
        if traj_mode:
            self.s_emb_layers = torch.nn.ModuleList([PositionalEmbedding(hidden_dims[0])])
            for i in range(len(hidden_dims) - 1):
                self.s_emb_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
        if cond_dim > 0:
            self.cond_layers = torch.nn.ModuleList([nn.Linear(cond_dim, hidden_dims[0])])
            for i in range(len(hidden_dims) - 1):
                self.cond_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))

    def forward(self, x, t, cond=None, s=None):
        t_emb = t
        s_emb = s if s is not None else None
        cond_emb = cond if cond is not None else None
        for i in range(len(self.layers)-1):
            x = self.layers[i](x)
            t_emb = self.t_emb_layers[i](t_emb)
            s_emb = self.s_emb_layers[i](s_emb) if s is not None else None
            cond_emb = self.cond_layers[i](cond_emb) if cond is not None else None
            emb = t_emb + s_emb if s is not None else t_emb
            emb = emb + cond_emb if cond is not None else emb
            x = F.silu(x + emb)
        x = self.layers[-1](x)
        return x
