"""Shared GraphSAGE backbone, PDFM-style heads, and prototype objectives."""

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import SAGEConv


class PDFMEncoder(nn.Module):
    """Shared parameter layout and per-modality heads; subclasses implement message passing.

    Keep layer names and initialization order compatible with the final saved weights.
    """

    def __init__(
        self, feature_modality, hidden=64, block_dim=16, sage_backbone="max1_distance"
    ):
        super().__init__()
        if sage_backbone != "max1_distance":
            raise ValueError("The final release uses one GraphSAGE layer")
        self.sage_backbone = sage_backbone
        self.register_buffer("feature_modality", feature_modality.clone())
        self.widths = torch.bincount(feature_modality).tolist()
        self.block_dim = block_dim
        self.embedding_dim = block_dim * len(self.widths)
        # A zero standardized value is observed only when its explicit mask says so.
        self.input = nn.Linear(len(feature_modality) + len(self.widths), hidden)
        self.sage = SAGEConv(hidden, hidden, aggr="max", project=True)
        self.embedding = nn.Linear(hidden, self.embedding_dim)
        self.decoders = nn.ModuleList(
            nn.Linear(block_dim, width) for width in self.widths
        )

    def decode(self, z):
        return torch.cat(
            [
                head(part)
                for head, part in zip(self.decoders, z.split(self.block_dim, dim=1))
            ],
            dim=1,
        )


def reconstruction_losses(inference, target, mask, node_type, feature_modality):
    """Return equal-modality losses separately for both physical scales."""
    if (
        inference.ndim != 2
        or target.shape != inference.shape
        or mask.dtype != torch.bool
        or mask.ndim != 2
        or len(mask) != len(inference)
        or node_type.shape != (len(inference),)
        or feature_modality.shape != (inference.shape[1],)
        or feature_modality.dtype != torch.long
        or (feature_modality < 0).any()
        or (feature_modality >= mask.shape[1]).any()
        or ((node_type != 0) & (node_type != 1)).any()
    ):
        raise ValueError(
            "Expected aligned inference/target, boolean supervision mask, and scale/modality indices"
        )
    supervised = mask[:, feature_modality]
    if (
        not torch.isfinite(target[supervised]).all()
        or not torch.isfinite(inference[supervised]).all()
    ):
        raise ValueError("Supervised targets and inferences must be finite")
    result = {}
    for level, name in ((0, "region"), (1, "subregion")):
        terms = []
        for m in range(mask.shape[1]):
            rows = (node_type == level) & mask[:, m]
            if rows.any():
                cols = feature_modality == m
                terms.append(
                    F.huber_loss(inference[rows][:, cols], target[rows][:, cols])
                )
        # An empty slice keeps zero gradients without reading unobserved NaN/Inf values.
        result[name] = (
            torch.stack(terms).mean() if terms else inference.reshape(-1)[:0].sum()
        )
    return result


def masked_pdfm_loss(encoder, graph, mask):
    """Hide one SSL target view; supervise only its originally observed entries."""
    if (
        mask.dtype != torch.bool
        or mask.shape != graph["observed"].shape
        or (mask & ~graph["observed"]).any()
    ):
        raise ValueError("SSL targets must be a boolean subset of natural observations")
    _, inference = encoder.encode_graph(graph, graph["observed"] & ~mask)
    return sum(
        reconstruction_losses(
            inference, graph["x"], mask, graph["node_type"], graph["feature_modality"]
        ).values()
    )


def encoder_from_config(feature_modality, cfg):
    """Only the final GWFM and task-adapted PDFM baseline are supported."""
    if cfg.get("modality_arch") == "specific_shared":
        from .modality_structure import ModalityStructuredEncoder

        return ModalityStructuredEncoder(
            feature_modality,
            "specific_shared",
            cfg["hidden"],
            cfg["block_dim"],
            cfg["spatial_neighbors"],
            cfg["distance_bandwidths"],
        )
    if cfg.get("pdfm_neighborhood") == "scale":
        from .pdfm_neighborhood import ScalePDFMEncoder

        return ScalePDFMEncoder(
            feature_modality,
            cfg["spatial_neighbors"],
            cfg["hidden"],
            cfg["block_dim"],
            aggregation=cfg["pdfm_aggregation"],
        )
    raise ValueError("This release only contains GWFM specific_shared and PDFM scale")
