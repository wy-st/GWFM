"""Three final component removals, with checkpoint-compatible parameter names."""

import torch
from torch.nn import functional as F

from ..models.model import encoder_from_config, reconstruction_losses
from ..models.modality_structure import ModalityStructuredEncoder
from ..models.visible_graph import VisibleGraphEncoder, VisibleGraphSSL

VARIANTS = ("no_weighting", "no_modality", "no_prototype", "full")


class MixedInputEncoder(VisibleGraphEncoder):
    """Remove the modality-specific stems and fusion in both branches."""

    def __init__(self, feature_modality, cfg):
        super().__init__(
            feature_modality,
            "visible_relations",
            cfg["hidden"],
            cfg["block_dim"],
            cfg["spatial_neighbors"],
            cfg["distance_bandwidths"],
            post_sage_gate=False,
            spatial_sum=True,
        )

    def trunk_embedding(self, trunk, g, visible, topology):
        x = torch.where(visible[:, self.feature_modality], g["x"], 0.0)
        h = F.gelu(trunk.input(torch.cat([x, visible.to(x.dtype)], 1)))
        projected = F.relu(trunk.sage.lin(h))
        spatial, up, down = topology
        hierarchy = [
            self.pool(
                projected,
                e,
                h.new_ones(e.shape[1]),
                trunk.sage.aggr_module.aggrs[1],
                True,
            )
            for e in (up, down)
        ]
        blocks = []
        for m, (edges, delta) in enumerate(spatial):
            weights = self.spatial_weights(trunk, g, m, edges, delta)
            local = self.pool(
                projected, edges, weights, trunk.sage.aggr_module.aggrs[1], True
            )
            hidden = F.relu(
                trunk.sage.lin_l(torch.cat([local, *hierarchy], 1))
                + trunk.sage.lin_r(h)
            )
            block = slice(m * 16, (m + 1) * 16)
            blocks.append(
                F.linear(
                    hidden, trunk.embedding.weight[block], trunk.embedding.bias[block]
                )
            )
        return torch.cat(blocks, 1)


class AblationEncoder(ModalityStructuredEncoder):
    def __init__(self, feature_modality, cfg):
        super().__init__(
            feature_modality,
            "specific_shared",
            cfg["hidden"],
            cfg["block_dim"],
            cfg["spatial_neighbors"],
            cfg["distance_bandwidths"],
        )
        self.ablation = cfg["ablation"]
        if self.ablation not in ("no_weighting", "no_prototype"):
            raise ValueError(self.ablation)
        if self.ablation == "no_prototype":
            self.has_prototype = False
            self.prototypes = self.log_scale = self.metric_logits = None
            self.learned = False

    def representations(self, g, visible):
        if self.ablation != "no_weighting":
            return super().representations(g, visible)
        topology = self.topology(g, visible)
        base = self.trunk_embedding(self, g, visible, topology)
        value = self.trunk_embedding(self.k0, g, visible, topology)
        # Keep the auxiliary branch; remove only its multiplication into the output.
        return base, value, torch.ones_like(value)


class AblationSSL(VisibleGraphSSL):
    def begin_epoch(self, epoch, g, seed):
        if self.teacher is not None:
            super().begin_epoch(epoch, g, seed)

    def forward(self, g, masks):
        if self.teacher is not None:
            return super().forward(g, masks)
        if len(masks) != 2 or any(
            m.dtype != torch.bool
            or m.shape != g["observed"].shape
            or (m & ~g["observed"]).any()
            for m in masks
        ):
            raise ValueError("Two masks restricted to observed targets are required")
        losses = [
            reconstruction_losses(
                self.encoder.encode_graph(g, g["observed"] & ~m)[1],
                g["x"],
                m,
                g["node_type"],
                g["feature_modality"],
            )
            for m in masks
        ]
        terms = {
            key: torch.stack([r[key] for r in losses]).mean()
            for key in ("region", "subregion")
        }
        return sum(terms.values()), terms


def make_encoder(feature_modality, cfg):
    """Construct GWFM, PDFM, or one of the three reported component ablations."""
    if cfg.get("ablation") in ("no_weighting", "no_prototype"):
        return AblationEncoder(feature_modality, cfg)
    if cfg.get("ablation") == "no_modality" or (
        cfg.get("visibility_strategy") == "visible_relations"
        and "modality_arch" not in cfg
    ):
        return MixedInputEncoder(feature_modality, cfg)
    return encoder_from_config(feature_modality, cfg)
