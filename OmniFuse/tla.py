# tla.py
import torch
import torch.nn.functional as F
import itertools
import numpy as np

def enumerate_masks(M):
    masks = []
    for i in range(1, 2**M):
        mask = tuple(int((i >> j) & 1) for j in range(M))
        masks.append(mask)
    return masks

class TLAMiner:
    """Traceable Laziness Activation"""
    def __init__(self, M):
        self.M = M
        self.masks = enumerate_masks(M)

    def mine_lazy_mask(self, model_forward_fn, batch_inputs, labels):
        total_dist = []
        for mask in self.masks:
            logits = model_forward_fn(mask, batch_inputs)
            probs = F.softmax(logits, dim=1)
            p_true = probs.gather(1, labels.view(-1,1)).squeeze(1)
            dist = (1.0 - p_true).sum().item()
            total_dist.append(dist)
        idx = int(np.argmax(total_dist))
        return self.masks[idx]
