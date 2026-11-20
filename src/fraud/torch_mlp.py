"""PyTorch MLP with focal loss, wrapped as a scikit-learn classifier (used when torch is installed).

Focal loss (Lin et al. 2017) down-weights easy negatives, a common alternative to resampling for
extreme imbalance.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class TorchMLP(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden=(128, 64), epochs=15, lr=1e-3, batch=1024, focal=True, gamma=2.0, alpha=0.75, seed=0):
        self.hidden, self.epochs, self.lr, self.batch = hidden, epochs, lr, batch
        self.focal, self.gamma, self.alpha, self.seed = focal, gamma, alpha, seed

    def fit(self, X, y):
        import torch
        from torch import nn

        torch.manual_seed(self.seed)
        X = torch.tensor(np.asarray(X), dtype=torch.float32)
        y = torch.tensor(np.asarray(y), dtype=torch.float32)
        layers, d = [], X.shape[1]
        for h in self.hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.2)]
            d = h
        self.net_ = nn.Sequential(*layers, nn.Linear(d, 1))
        opt = torch.optim.AdamW(self.net_.parameters(), lr=self.lr, weight_decay=1e-4)
        bce = nn.BCEWithLogitsLoss(reduction="none")
        self.net_.train()
        for _ in range(self.epochs):
            perm = torch.randperm(len(y))
            for i in range(0, len(y), self.batch):
                b = perm[i:i + self.batch]
                logits = self.net_(X[b]).squeeze(1)
                loss = bce(logits, y[b])
                if self.focal:
                    p = torch.sigmoid(logits)
                    pt = torch.where(y[b] == 1, p, 1 - p)
                    at = torch.where(y[b] == 1, torch.full_like(p, self.alpha), torch.full_like(p, 1 - self.alpha))
                    loss = at * (1 - pt) ** self.gamma * loss
                opt.zero_grad()
                loss.mean().backward()
                opt.step()
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        import torch

        self.net_.eval()
        with torch.no_grad():
            p = torch.sigmoid(self.net_(torch.tensor(np.asarray(X), dtype=torch.float32))).squeeze(1).numpy()
        return np.stack([1 - p, p], axis=1)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
