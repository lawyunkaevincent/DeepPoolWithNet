"""
Q-network for DRT dispatch.

Candidate generation now produces exactly one CandidateInsertion per
eligible taxi (two-phase insertion) plus a DEFER pseudo-action, so every
taxi contributes a single "lottery ticket" to the argmax. The older
TaxiFairQNetwork, which normalized scores within each taxi group to
counteract O(N^2)-per-taxi candidate explosion, is no longer needed and
has been removed.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class CandidateScorerMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], dropout: float = 0.1):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # mask ignored — present only to share interface with DuelingCandidateScorerMLP
        b, c, f = x.shape
        return self.net(x.reshape(b * c, f)).reshape(b, c)


class DuelingCandidateScorerMLP(nn.Module):
    """
    Dueling-DQN variant of CandidateScorerMLP.

    Mirrors DuelingZoneCNNQNetwork from DeepPoolZone/zone_qnetwork.py,
    adapted for the parametric (per-candidate) action space used in RealDQN.

    Architecture:
        Shared trunk  : input_dim → hidden_dims[:-1]
        Value stream  : trunk_out → hidden_dims[-1] → 1   (V(s))
        Advantage stream: trunk_out → hidden_dims[-1] → 1  (A(s,a))
        Q(s,a) = V(s) + A(s,a) − mean_valid(A)

    The mean advantage is computed only over valid (non-padded) candidates,
    so masked-out padding rows don't corrupt the centering term.
    """

    def __init__(self, input_dim: int, hidden_dims: Sequence[int], dropout: float = 0.1):
        super().__init__()
        if len(hidden_dims) == 0:
            raise ValueError("DuelingCandidateScorerMLP requires at least one hidden dim")

        # Shared trunk: all hidden layers except the last
        trunk_layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims[:-1]:
            trunk_layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        self.shared_trunk = nn.Sequential(*trunk_layers)
        trunk_out = prev

        last_h = hidden_dims[-1]

        # Value stream: trunk_out → last_h → 1
        self.value_stream = nn.Sequential(
            nn.Linear(trunk_out, last_h),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(last_h, 1),
        )

        # Advantage stream: trunk_out → last_h → 1
        self.advantage_stream = nn.Sequential(
            nn.Linear(trunk_out, last_h),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(last_h, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        x    : [B, C, F]
        mask : [B, C]  — 1.0 for valid candidates, 0.0 for padding
        Returns Q-values [B, C].
        """
        b, c, f = x.shape
        shared = self.shared_trunk(x.reshape(b * c, f)).reshape(b, c, -1)  # [B, C, trunk_out]

        value = self.value_stream(shared.reshape(b * c, -1)).reshape(b, c)      # [B, C]
        advantage = self.advantage_stream(shared.reshape(b * c, -1)).reshape(b, c)  # [B, C]

        # Mean advantage over valid candidates only (avoid polluting with -1e9 padding)
        if mask is not None:
            valid = mask > 0.0                                               # [B, C] bool
            adv_sum = (advantage * valid.float()).sum(dim=1, keepdim=True)
            valid_count = valid.float().sum(dim=1, keepdim=True).clamp(min=1.0)
            adv_mean = adv_sum / valid_count
        else:
            adv_mean = advantage.mean(dim=1, keepdim=True)

        return value + (advantage - adv_mean)  # [B, C]


class ParametricQNetwork(nn.Module):
    """Q(s, a) scorer for variable-size candidate sets.

    Input shape: [batch, num_candidates, feature_dim + 1]
    The last channel is a binary valid-mask channel.

    When ``use_dueling=True`` (default) the scorer is
    ``DuelingCandidateScorerMLP`` — separate V(s) and A(s,a) streams with
    mean-advantage centering over valid candidates only.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        dropout: float = 0.1,
        use_dueling: bool = True,
    ):
        super().__init__()
        if use_dueling:
            self.scorer = DuelingCandidateScorerMLP(input_dim, hidden_dims, dropout)
        else:
            self.scorer = CandidateScorerMLP(input_dim, hidden_dims, dropout)

    def forward(self, x_with_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x_with_mask[..., :-1]
        mask = x_with_mask[..., -1]
        q_values = self.scorer(x, mask=mask)
        q_values = q_values.masked_fill(mask <= 0.0, -1e9)
        return q_values, mask


class SpatialEncodedQNetwork(nn.Module):
    """Q(s, a) scorer with separate tabular and spatial encoders.

    Splits the per-row feature vector into:
        [tabular (tab_dim) | spatial (spat_dim) | mask (1)]
    Then encodes each modality independently before merging into the dueling head.

    Spatial features are global state (identical across all candidates at one
    decision point), so the spatial branch only runs ONCE per decision —
    extracted from candidate index 0, then broadcast to every candidate before
    concat. This is both faster and a stronger inductive bias than letting one
    big Linear(189→256) re-discover the modality boundary from scratch.

    Architecture:
        tabular(tab_dim) ─► Linear(tab_dim → tab_emb) → ReLU → Dropout
                                                           │
                                                           ▼ per-candidate [B,C,tab_emb]
        spatial(spat_dim) ─► Linear(spat_dim → spat_emb) → ReLU → Dropout
                          ─► Linear(spat_emb → spat_emb) → ReLU
                                                           │
                                                           ▼ per-decision [B,1,spat_emb]
                                          (broadcast to C candidates)
                                                           │
                            concat → [B, C, tab_emb + spat_emb] → DuelingCandidateScorerMLP
    """

    def __init__(
        self,
        tab_dim: int,
        spat_dim: int,
        hidden_dims: Sequence[int],
        tab_emb: int = 64,
        spat_emb: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        if spat_dim <= 0:
            raise ValueError(
                "SpatialEncodedQNetwork requires spat_dim > 0. "
                "Use ParametricQNetwork when spatial features are disabled."
            )
        self.tab_dim = int(tab_dim)
        self.spat_dim = int(spat_dim)
        self.tab_emb = int(tab_emb)
        self.spat_emb = int(spat_emb)

        self.tabular_encoder = nn.Sequential(
            nn.Linear(tab_dim, tab_emb),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.spatial_encoder = nn.Sequential(
            nn.Linear(spat_dim, spat_emb),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(spat_emb, spat_emb),
            nn.ReLU(),
        )
        self.scorer = DuelingCandidateScorerMLP(
            input_dim=tab_emb + spat_emb,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )

    def forward(self, x_with_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Layout per row: [tab(tab_dim) | spat(spat_dim) | mask(1)]
        tab = x_with_mask[..., : self.tab_dim]                                    # [B, C, tab_dim]
        spat_per_cand = x_with_mask[..., self.tab_dim : self.tab_dim + self.spat_dim]  # [B, C, spat_dim]
        mask = x_with_mask[..., -1]                                               # [B, C]

        b, c, _ = tab.shape

        # Tabular: per-candidate
        tab_emb = self.tabular_encoder(tab.reshape(b * c, self.tab_dim)).reshape(b, c, -1)

        # Spatial: per-decision (use candidate 0 — values are identical across C)
        spat = spat_per_cand[:, 0, :]                                             # [B, spat_dim]
        spat_emb = self.spatial_encoder(spat)                                     # [B, spat_emb]
        spat_emb = spat_emb.unsqueeze(1).expand(b, c, spat_emb.shape[-1])         # [B, C, spat_emb]

        merged = torch.cat([tab_emb, spat_emb], dim=-1)                           # [B, C, tab_emb + spat_emb]

        q_values = self.scorer(merged, mask=mask)
        q_values = q_values.masked_fill(mask <= 0.0, -1e9)
        return q_values, mask
