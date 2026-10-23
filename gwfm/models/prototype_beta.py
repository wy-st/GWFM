"""Prototype consistency during pretraining; beta is learned by a separate projection."""

import torch
from torch import nn
from torch.nn import functional as F

from .model import PDFMEncoder


def scale_mean(values, kind, eligible):
    terms = [
        values[(kind == k) & eligible].mean()
        for k in (0, 1)
        if ((kind == k) & eligible).any()
    ]
    return torch.stack(terms).mean() if terms else values.sum() * 0


class PrototypeBetaEncoder(PDFMEncoder):
    def __init__(
        self,
        feature_modality,
        strategy,
        hidden=64,
        block_dim=16,
        sage_backbone="max1_distance",
    ):
        if (
            strategy != "proto_self_beta"
            or block_dim != 16
            or len(feature_modality.unique()) != 3
        ):
            raise ValueError(
                "Expected a known prototype-beta strategy and three 16D blocks"
            )
        super().__init__(
            feature_modality, hidden, block_dim, sage_backbone=sage_backbone
        )
        self.strategy = strategy
        self.prototypes = nn.Parameter(F.normalize(torch.randn(16, 48), dim=1))
        self.learned = True
        self.log_scale = nn.Parameter(torch.zeros(()))
        self.metric_logits = nn.Parameter(torch.zeros(48))
        self.register_buffer("initial_scale", torch.ones(()))
        self.register_buffer("position_mean", torch.zeros(2))
        self.register_buffer("position_std", torch.ones(2))
        with torch.random.fork_rng():
            self.beta_map = nn.Linear(48, 48)
            nn.init.zeros_(self.beta_map.weight)
            nn.init.zeros_(self.beta_map.bias)

    def similarity(self, z, known):
        """Compare embeddings with 16 prototypes using only supported modality blocks."""
        if known.shape != (len(z), 3) or known.dtype != torch.bool:
            raise ValueError("A boolean modality support mask is required")
        support = known.repeat_interleave(16, 1).to(z.dtype)
        weight = self.metric_logits.softmax(0) * 48
        support = support * weight
        numerator = (z * support) @ self.prototypes.T
        norm_z = (z.square() * support).sum(1, keepdim=True).clamp_min(1e-12).sqrt()
        norm_p = (support @ self.prototypes.square().T).clamp_min(1e-12).sqrt()
        return numerator / (norm_z * norm_p)

    def assignment(self, z, known, teacher=False, center=None):
        cosine = self.similarity(z, known)
        centered = cosine if center is None else cosine - center
        logits = centered * self.log_scale.exp()
        return logits.softmax(1), cosine, logits

    def encode_graph(self, g, visible):
        _, z, _ = self.representations(g, visible)
        return z, self.decode(z)


class PrototypeBetaSSL(nn.Module):

    def reference_support(self, observed):
        eligible = observed.any(1)
        return observed, eligible

    def consistency(self, bases, teacher_base, observed, kind):
        """Match masked student assignments to the EMA teacher and prevent collapsed similarities."""
        known, eligible = self.reference_support(observed)
        with torch.no_grad():
            q, teacher_cosine, _ = self.teacher.assignment(
                teacher_base, known, teacher=True, center=self.center
            )
            values = [
                teacher_cosine[(kind == k) & eligible].mean(0)
                for k in (0, 1)
                if ((kind == k) & eligible).any()
            ]
            self._center_update = torch.stack(values).mean(0) if values else None
        consistency, variance = [], []
        for z in bases:
            _, cosine, logits = self.encoder.assignment(z, known)
            consistency.append(
                scale_mean(-(q * logits.log_softmax(1)).sum(1), kind, eligible)
            )
            terms = []
            for k in (0, 1):
                rows = (kind == k) & eligible
                if rows.sum() >= 2:
                    terms.append(
                        F.relu(
                            0.1 - (cosine[rows].var(0, unbiased=False) + 1e-4).sqrt()
                        ).mean()
                    )
            variance.append(torch.stack(terms).mean() if terms else cosine.sum() * 0)
        return torch.stack(consistency).mean(), torch.stack(variance).mean()
