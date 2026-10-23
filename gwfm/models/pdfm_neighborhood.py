"""Shared GraphSAGE with the saved experiment's distance-weighted mean at both scales."""

import torch
from torch.nn import functional as F
from torch_geometric.nn import SAGEConv

from .model import PDFMEncoder
from .visible_graph import modality_neighbors


class PDFMNeighborhoodEncoder(PDFMEncoder):
    def __init__(
        self,
        feature_modality,
        neighborhood,
        aggregation,
        neighbors=8,
        hidden=64,
        block_dim=16,
    ):
        if neighborhood != "visible" or aggregation != "mean":
            raise ValueError("Expected visible neighbors and mean aggregation")
        if not isinstance(neighbors, int) or neighbors < 1:
            raise ValueError("Positive neighbor count required")
        super().__init__(feature_modality, hidden, block_dim)
        self.neighborhood = neighborhood
        self.aggregation = aggregation
        self.neighbors = neighbors
        with torch.random.fork_rng():
            layer = SAGEConv(hidden, hidden, aggr="mean", project=True)
        layer.load_state_dict(self.sage.state_dict())
        self.sage = layer

    def topology(self, graph, visible):
        if (
            visible.dtype != torch.bool
            or visible.shape != graph["observed"].shape
            or (visible & ~graph["observed"]).any()
            or len(graph["split"].unique()) != 1
        ):
            raise ValueError(
                "One isolated partition and a boolean subset of observations required"
            )
        spatial = modality_neighbors(
            graph["pos"].numpy() * 200.0,
            graph["node_type"].numpy(),
            visible.numpy(),
            self.neighbors,
        )
        child = (graph["node_type"] == 1).nonzero().flatten()
        parent = graph["parent"][child]
        if len(child) and (
            (parent < 0).any() or (graph["node_type"][parent] != 0).any()
        ):
            raise ValueError("Invalid parent containment")
        up = torch.stack([child, parent])
        hierarchy = torch.cat([up, up.flip(0)], 1)
        result = []
        for edges, _ in spatial:
            squared = (
                ((graph["pos"][edges[0]] - graph["pos"][edges[1]]) ** 2)
                .sum(1)
                .clamp_min(1e-20)
            )
            nearest = squared.new_full((len(visible),), float("inf"))
            nearest.scatter_reduce_(
                0, edges[1], squared, reduce="amin", include_self=True
            )
            weight = nearest[edges[1]] / squared
            result.append(
                (
                    torch.cat([edges, hierarchy], 1),
                    torch.cat([weight, weight.new_ones(hierarchy.shape[1])]),
                )
            )
        return result

    def base_embedding(
        self, x, observed, edge_index, edge_weight=None, pos=None, node_type=None
    ):
        x = torch.where(observed[:, self.feature_modality], x, 0.0)
        h = F.gelu(self.input(torch.cat([x, observed.to(x.dtype)], 1)))
        weight = h.new_ones(edge_index.shape[1]) if edge_weight is None else edge_weight
        if (
            weight.shape != (edge_index.shape[1],)
            or not torch.isfinite(weight).all()
            or (weight < 0).any()
        ):
            raise ValueError("One finite nonnegative weight per edge required")
        source, target = edge_index
        message = F.relu(self.sage.lin(h))[source] * weight[:, None]
        mass = h.new_zeros(len(h)).index_add(0, target, weight)
        mean = (
            torch.zeros_like(h).index_add(0, target, message)
            / mass.clamp_min(1e-12)[:, None]
        )
        return self.embedding(F.relu(self.sage.lin_l(mean) + self.sage.lin_r(h)))

    def encode_graph(self, graph, observed):
        graphs = self.topology(graph, observed)
        # Each modality routes through the same parameters; there are no extra encoders or heads.
        blocks = [
            self.base_embedding(graph["x"], observed, edges, weight)[
                :, m * self.block_dim : (m + 1) * self.block_dim
            ]
            for m, (edges, weight) in enumerate(graphs)
        ]
        z = torch.cat(blocks, 1)
        return z, self.decode(z)


class ScalePDFMEncoder(PDFMNeighborhoodEncoder):
    """Keep scale-specific routing; both scales use normalized weighted mean."""

    def __init__(
        self,
        feature_modality,
        neighbors=8,
        hidden=64,
        block_dim=16,
        aggregation="mean",
    ):
        if aggregation != "mean":
            raise ValueError(f"Unsupported PDFM aggregation: {aggregation}")
        super().__init__(
            feature_modality, "visible", "mean", neighbors, hidden, block_dim
        )
        self.scale_aggregation = aggregation

    def topology(self, graph, visible):
        graphs = super().topology(graph, visible)
        old_edges = graph["edge_index"]
        to_region = graph["node_type"][old_edges[1]] == 0
        result = []
        for edges, weight in graphs:
            to_child = graph["node_type"][edges[1]] == 1
            result.append(
                (
                    torch.cat([old_edges[:, to_region], edges[:, to_child]], 1),
                    torch.cat([graph["edge_weight"][to_region], weight[to_child]]),
                )
            )
        return result
