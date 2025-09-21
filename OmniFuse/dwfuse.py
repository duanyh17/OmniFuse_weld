# dwfuse.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ModalityClassifier(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, v):
        return self.fc(v)

class DWFuse(nn.Module):
    """Dynamic Weighted Fusion"""
    def __init__(self, modal_dims, num_classes, beta=0.5):
        super().__init__()
        self.M = len(modal_dims)
        self.g_mod = nn.ModuleList([ModalityClassifier(d, num_classes) for d in modal_dims])
        self.beta = beta
        total_dim = sum(modal_dims)
        self.joint_clf = nn.Sequential(
            nn.Linear(total_dim, 256), nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, vs, labels=None):
        per_logits = [g(v) for g, v in zip(self.g_mod, vs)]
        per_probs = [F.softmax(l, dim=1) for l in per_logits]
        if labels is not None:
            p_true = [p.gather(1, labels.view(-1,1)).squeeze(1) for p in per_probs]
        else:
            p_true = [p.max(dim=1).values for p in per_probs]

        p_stack = torch.stack(p_true, dim=1)  # [B, M]
        omegas = []
        for m in range(self.M):
            others = 1.0 - torch.cat([p_stack[:, :m], p_stack[:, m+1:]], dim=1)
            prod = torch.prod(others, dim=1)
            omega = prod ** (self.beta / (self.M - 1))
            omegas.append(omega)
        omegas = torch.stack(omegas, dim=1)

        joint = torch.cat(vs, dim=1)
        logits_joint = self.joint_clf(joint)
        return logits_joint, per_logits, omegas
