"""Visible-neighbor messages and separate parent–child relations in both GWFM branches."""

from copy import deepcopy

import numpy as np
from scipy.spatial import cKDTree
import torch
from torch import nn
from torch_geometric.nn import SAGEConv

from .branch_beta import BranchBetaEncoder, BranchBetaSSL
from .model import reconstruction_losses
from .prototype_beta import scale_mean


def modality_neighbors(pos, kind, visible, neighbors):
    """Build same-scale edges from each node to at most k visible donors, excluding itself."""
    pos, kind, visible = np.asarray(pos), np.asarray(kind), np.asarray(visible)
    if (
        pos.shape != (len(kind), 2)
        or visible.shape != (len(kind), 3)
        or visible.dtype != bool
        or not np.isfinite(pos).all()
        or not np.isin(kind, [0, 1]).all()
        or not isinstance(neighbors, int)
        or neighbors < 1
    ):
        raise ValueError(
            "Finite coordinates, two scales, three boolean modalities and positive k required"
        )
    graphs = []
    for m in range(3):
        sources, destinations, differences = [], [], []
        for level in (0, 1):
            query = np.flatnonzero(kind == level)
            donors = query[visible[query, m]]
            if not len(donors) or not len(query):
                continue
            k = min(neighbors + 1, len(donors))
            d, idx = cKDTree(pos[donors]).query(pos[query], k=list(range(1, k + 1)))
            source = donors[idx]
            destination = np.broadcast_to(query[:, None], source.shape)
            eligible = source != destination
            keep = eligible & (eligible.cumsum(1) <= neighbors)
            nearest = np.where(eligible, d, np.inf).min(1)
            delta = np.maximum(
                d * d - np.where(np.isfinite(nearest), nearest, 0.0)[:, None] ** 2, 0.0
            )
            sources.append(source[keep])
            destinations.append(destination[keep])
            differences.append(delta[keep])
        if sources:
            edges = torch.from_numpy(
                np.stack([np.concatenate(sources), np.concatenate(destinations)])
            )
            squared_difference = torch.from_numpy(
                np.concatenate(differences).astype(np.float32)
            )
        else:
            edges = torch.empty((2, 0), dtype=torch.long)
            squared_difference = torch.empty(0)
        graphs.append((edges, squared_difference))
    return graphs


class VisibleGraphEncoder(BranchBetaEncoder):
    def __init__(
        self,
        feature_modality,
        visibility_strategy,
        hidden=64,
        block_dim=16,
        neighbors=8,
        bandwidths=(14.402076958647303, 4.3206784710364285),
        post_sage_gate=False,
        spatial_sum=False,
    ):
        if (
            visibility_strategy != "visible_relations"
            or post_sage_gate
            or not spatial_sum
        ):
            raise ValueError("Unknown visible-graph strategy")
        if not isinstance(post_sage_gate, bool) or not isinstance(spatial_sum, bool):
            raise ValueError("Gate and sum switches must be boolean")
        super().__init__(feature_modality, "dual_self_proto", hidden, block_dim)
        self.visibility_strategy = visibility_strategy
        self.neighbors = neighbors
        self.spatial_sum = spatial_sum
        self.separate = True
        self.register_buffer(
            "spatial_bandwidths", torch.tensor(bandwidths, dtype=torch.float32)
        )
        if (
            self.spatial_bandwidths.shape != (2,)
            or not torch.isfinite(self.spatial_bandwidths).all()
            or (self.spatial_bandwidths <= 0).any()
        ):
            raise ValueError("Two positive training-fitted spatial bandwidths required")
        with torch.random.fork_rng():
            self.sage = SAGEConv(hidden, hidden, aggr=["mean", "max"], project=True)
            if self.separate:
                previous = self.sage.lin_l
                projection = nn.Linear(6 * hidden, hidden)
                with torch.no_grad():
                    projection.weight.copy_(previous.weight.repeat(1, 3) / 3)
                    projection.bias.copy_(previous.bias)
                self.sage.lin_l = projection
            if spatial_sum:
                previous = self.sage.lin_l
                relations = 3 if self.separate else 1
                projection = nn.Linear(3 * relations * hidden, hidden)
                with torch.no_grad():
                    projection.weight.zero_()
                    for relation in range(relations):
                        projection.weight[
                            :, relation * 3 * hidden : (relation * 3 + 2) * hidden
                        ].copy_(
                            previous.weight[
                                :, relation * 2 * hidden : (relation * 2 + 2) * hidden
                            ]
                        )
                    projection.bias.copy_(previous.bias)
                self.sage.lin_l = projection
        self.k0.sage = deepcopy(self.sage)
        self.log_bandwidth = nn.Parameter(torch.zeros(2, 3))
        self.k0.log_bandwidth = nn.Parameter(torch.zeros(2, 3))

    def topology(self, g, visible):
        if (
            visible.dtype != torch.bool
            or visible.shape != g["observed"].shape
            or len(g["split"].unique()) != 1
            or (visible & ~g["observed"]).any()
        ):
            raise ValueError(
                "One isolated partition and a subset of currently observed modalities required"
            )
        spatial = modality_neighbors(
            g["pos"].numpy() * 200.0,
            g["node_type"].numpy(),
            visible.numpy(),
            self.neighbors,
        )
        child = (g["node_type"] == 1).nonzero().flatten()
        up = torch.stack([child, g["parent"][child]])
        if len(child) and ((up[1] < 0).any() or (g["node_type"][up[1]] != 0).any()):
            raise ValueError("Invalid parent containment")
        return spatial, up, up.flip(0)

    def spatial_weights(self, trunk, g, m, edges, delta):
        """Learn separate distance bandwidths for each branch, scale, and modality."""
        level = g["node_type"][edges[1]]
        bandwidth = self.spatial_bandwidths[level] * trunk.log_bandwidth[
            level, m
        ].exp().clamp(0.25, 4.0)
        return torch.exp(-(delta / bandwidth.square()).clamp(max=30.0))

    @staticmethod
    def pool(projected, edges, weights, maximum, include_sum=False):
        """Concatenate weighted mean, maximum, and optional sum messages at each destination."""
        source, target = edges
        messages = projected[source] * weights[:, None]
        total = projected.new_zeros(len(projected)).index_add(0, target, weights)
        summed = torch.zeros_like(projected).index_add(0, target, messages)
        mean = summed / total.clamp_min(1e-12)[:, None]
        parts = [mean, maximum(messages, index=target, dim_size=len(projected))]
        return torch.cat(parts + [summed] if include_sum else parts, 1)

    def beta_embedding(self, g, visible):
        return self.trunk_embedding(self, g, visible, self.topology(g, visible))

    def representations(self, g, visible):
        """Return geographic context, the final embedding z = beta * h_k, and beta in (0, 2)."""
        topology = self.topology(g, visible)
        base = self.trunk_embedding(self, g, visible, topology)
        value = self.trunk_embedding(self.k0, g, visible, topology)
        beta = 2 * self.beta_map(base).sigmoid()
        return base, beta * value, beta


class VisibleGraphSSL(BranchBetaSSL):
    @torch.no_grad()
    def begin_epoch(self, epoch, g, seed):
        if epoch != 1:
            return
        if not (g["split"] == 0).all():
            raise ValueError("Prototype calibration uses training families only")
        enc = self.encoder
        base = enc.beta_embedding(g, g["observed"])
        scores = enc.similarity(base, g["observed"])
        spread = (
            scale_mean(
                scores.var(1, unbiased=False), g["node_type"], g["observed"].any(1)
            )
            .sqrt()
            .clamp_min(1e-4)
        )
        enc.log_scale.copy_(-spread.log())
        enc.initial_scale.copy_(spread.reciprocal())
        state = enc.state_dict()
        self.teacher.load_state_dict(
            {name: state[name] for name in self.teacher.state_dict()}
        )

    def forward(
        self, g, masks, hierarchy=None, generator=None, *, corruption="modality_mask"
    ):
        if len(masks) != 2 or any(
            m.dtype != torch.bool
            or m.shape != g["observed"].shape
            or (m & ~g["observed"]).any()
            for m in masks
        ):
            raise ValueError("Two masks restricted to observed targets required")
        if corruption != "modality_mask":
            raise ValueError("Unknown SSL corruption")
        views = []
        for mask in masks:
            visible = g["observed"] & ~mask
            student = g
            views.append(self.encoder.representations(student, visible))
        targets = masks
        losses = [
            reconstruction_losses(
                self.encoder.decode(z),
                g["x"],
                mask,
                g["node_type"],
                g["feature_modality"],
            )
            for (_, z, _), mask in zip(views, targets)
        ]
        terms = {
            k: torch.stack([r[k] for r in losses]).mean()
            for k in ("region", "subregion")
        }
        with torch.no_grad():
            teacher_base = self.teacher.beta_embedding(g, g["observed"])
        terms["prototype"], terms["variance"] = self.consistency(
            [v[0] for v in views], teacher_base, g["observed"], g["node_type"]
        )
        return (
            terms["region"]
            + terms["subregion"]
            + 0.05 * (terms["prototype"] + terms["variance"]),
            terms,
        )
