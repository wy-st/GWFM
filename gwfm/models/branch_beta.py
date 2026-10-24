"""Independent attribute (E_0, stored as k0) and geographic-weight (beta) branches."""

from copy import deepcopy

import torch
from torch import nn

from .model import PDFMEncoder
from .prototype_beta import PrototypeBetaEncoder, PrototypeBetaSSL


class BranchBetaEncoder(PrototypeBetaEncoder):
    def __init__(
        self,
        feature_modality,
        strategy,
        hidden=64,
        block_dim=16,
        sage_backbone="max1_distance",
    ):
        if strategy != "dual_self_proto":
            raise ValueError("Only the final independent beta/k0 branches are retained")
        super().__init__(
            feature_modality, "proto_self_beta", hidden, block_dim, sage_backbone
        )
        self.branch_strategy = strategy
        self.has_prototype = True
        # Forking RNG preserves the original paired initialization, without sharing parameters.
        with torch.random.fork_rng():
            self.k0 = PDFMEncoder(
                feature_modality, hidden, block_dim, sage_backbone=sage_backbone
            )
        state = self.state_dict()
        self.k0.load_state_dict({k: state[k] for k in self.k0.state_dict()})
        self.k0.decoders = nn.ModuleList()


class BranchBetaSSL(PrototypeBetaSSL):
    def __init__(self, encoder):
        nn.Module.__init__(self)
        self.encoder = encoder
        self.strategy = encoder.strategy
        self.teacher = None
        if encoder.has_prototype:
            self.teacher = deepcopy(encoder).requires_grad_(False)
            self.teacher.k0 = None
            self.teacher.beta_map = None
            self.teacher.decoders = nn.ModuleList()
        self.register_buffer("center", torch.zeros(16))
        self._center_update = None

    @torch.no_grad()
    def update_teacher(self):
        if self.teacher is None:
            return
        source = dict(self.encoder.named_parameters())
        for name, p in self.teacher.named_parameters():
            p.mul_(0.99).add_(source[name], alpha=0.01)
        if self._center_update is not None:
            self.center.mul_(0.9).add_(self._center_update, alpha=0.1)
