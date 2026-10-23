"""Modality-pure spatial encoders followed by explicit cross-modal fusion.

The retained specific/shared fusion is inspired by ShaSpec; it is task-adapted.
"""

from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from .visible_graph import VisibleGraphEncoder


ARCHITECTURES = ("specific_shared",)


def available_mean(values, available):
    clean = torch.where(available[..., None], values, 0.0)
    return clean.sum(1) / available.sum(1).clamp_min(1)[:, None]


class ModalityFusion(nn.Module):
    """Combine modality-specific tokens with the mean of supported shared tokens."""

    def __init__(self, architecture):
        super().__init__()
        if architecture not in ARCHITECTURES:
            raise ValueError("Unknown modality architecture")
        self.architecture = architecture
        self.readout = nn.ModuleList(nn.Linear(35, 16) for _ in range(3))
        for layer in self.readout:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
            with torch.no_grad():
                layer.weight[:, :16].copy_(torch.eye(16))
        self.shared = nn.Linear(16, 16)

    def forward(self, tokens, visible, available):
        if (
            tokens.shape != (len(visible), 3, 16)
            or visible.shape != available.shape
            or visible.shape[1:] != (3,)
            or visible.dtype != torch.bool
            or available.dtype != torch.bool
            or (visible & ~available).any()
        ):
            raise ValueError(
                "Expected three 16D tokens and explicit node/context availability"
            )
        tokens = torch.where(available[..., None], tokens, 0.0)
        flags = visible.to(tokens.dtype)
        # Only modalities with node or neighbour support contribute to the shared mean.
        pooled = available_mean(self.shared(tokens), available)
        output = torch.stack(
            [
                pooled + layer(torch.cat([tokens[:, m], pooled, flags], 1))
                for m, layer in enumerate(self.readout)
            ],
            1,
        )
        info = dict(shared=pooled)
        return output.flatten(1), info


class ModalityStructuredEncoder(VisibleGraphEncoder):
    """Encode three four-feature modalities into explicit 16D blocks in both GWFM branches."""

    def __init__(
        self,
        feature_modality,
        architecture,
        hidden=64,
        block_dim=16,
        neighbors=8,
        bandwidths=(14.402076958647303, 4.3206784710364285),
    ):
        if architecture not in ARCHITECTURES or torch.bincount(
            feature_modality
        ).tolist() != [
            4,
            4,
            4,
        ]:
            raise ValueError(
                "Expected a known architecture and three four-feature modalities"
            )
        super().__init__(
            feature_modality,
            "visible_relations",
            hidden,
            block_dim,
            neighbors,
            bandwidths,
            post_sage_gate=False,
            spatial_sum=True,
        )
        self.modality_arch = architecture
        with torch.random.fork_rng():
            stems = nn.ModuleList(nn.Linear(5, hidden) for _ in range(3))
            for m, layer in enumerate(stems):
                cols = torch.cat(
                    [
                        (feature_modality == m).nonzero().flatten(),
                        torch.tensor([12 + m]),
                    ]
                )
                with torch.no_grad():
                    layer.weight.copy_(self.input.weight[:, cols])
                    layer.bias.copy_(self.input.bias)
            fusion = ModalityFusion(architecture)
        self.modality_input = stems
        self.modality_fusion = fusion
        self.k0.modality_input = deepcopy(stems)
        self.k0.modality_fusion = deepcopy(fusion)
        self.input = None
        self.k0.input = None

    @staticmethod
    def context_available(visible, topology):
        spatial, up, down = topology
        result = visible.clone()
        for m, (edges, _) in enumerate(spatial):
            for edge in (edges, up, down):
                source, target = edge
                counts = torch.zeros(
                    len(visible), dtype=torch.long, device=visible.device
                )
                counts.index_add_(0, target, visible[source, m].long())
                result[:, m] |= counts > 0
        return result

    def modality_tokens(self, trunk, g, visible, topology):
        """Aggregate spatial, child-to-parent, and parent-to-child messages for each modality."""
        spatial, up, down = topology
        tokens = []
        for m, (edges, delta) in enumerate(spatial):
            x = torch.where(
                visible[:, m : m + 1], g["x"][:, self.feature_modality == m], 0.0
            )
            h = F.gelu(
                trunk.modality_input[m](
                    torch.cat([x, visible[:, m : m + 1].to(x.dtype)], 1)
                )
            )
            projected = F.relu(trunk.sage.lin(h))
            weights = self.spatial_weights(trunk, g, m, edges, delta)
            pools = [
                self.pool(
                    projected, edges, weights, trunk.sage.aggr_module.aggrs[1], True
                )
            ]
            pools += [
                self.pool(
                    projected,
                    e,
                    h.new_ones(e.shape[1]),
                    trunk.sage.aggr_module.aggrs[1],
                    True,
                )
                for e in (up, down)
            ]
            hidden = F.relu(trunk.sage.lin_l(torch.cat(pools, 1)) + trunk.sage.lin_r(h))
            block = slice(m * 16, (m + 1) * 16)
            tokens.append(
                F.linear(
                    hidden, trunk.embedding.weight[block], trunk.embedding.bias[block]
                )
            )
        return torch.stack(tokens, 1)

    def trunk_embedding(self, trunk, g, visible, topology):
        tokens = self.modality_tokens(trunk, g, visible, topology)
        available = self.context_available(visible, topology)
        return trunk.modality_fusion(tokens, visible, available)[0]
