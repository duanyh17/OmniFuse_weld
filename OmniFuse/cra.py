# cra.py
import torch
import torch.nn as nn

class ResidualAdapter(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim)
        )
    def forward(self, x):
        return self.net(x)

class CRA(nn.Module):
    """Cascade Residual Autoencoder"""
    def __init__(self, dim, K=5):
        super().__init__()
        self.K = K
        self.adapters = nn.ModuleList([ResidualAdapter(dim) for _ in range(K)])

    def forward_impute(self, v):
        deltas = []
        for k in range(self.K):
            inp = v + (sum(deltas) if deltas else 0)
            delta = self.adapters[k](inp)
            deltas.append(delta)
        v_prime = v + sum(deltas)
        return v_prime

    def backward_impute(self, v_prime):
        deltas = []
        for k in range(self.K):
            inp = v_prime + (sum(deltas) if deltas else 0)
            delta = self.adapters[k](inp)
            deltas.append(delta)
        v_double = v_prime + sum(deltas)
        return v_double
