"""
zone_qnetwork.py
----------------
CNN Q-network for the zone-based repositioning agent.

Architecture (inspired by DeepPool's Q-network):
    Input  : (batch, 2, ROWS, COLS)   — demand + vehicle density grids
    Conv1  : 16 filters, 3×3, padding=1  → (batch, 16, ROWS, COLS)
    ReLU
    Conv2  : 32 filters, 3×3, padding=1  → (batch, 32, ROWS, COLS)
    ReLU
    Conv3  : 64 filters, 3×3, padding=1  → (batch, 64, ROWS, COLS)
    ReLU
    Flatten → (batch, 64 * ROWS * COLS)
    Linear(64*R*C → 256)  + ReLU
    Linear(256 → n_zones)
    Output : Q-value for each zone (ROWS × COLS = n_zones actions)

Why a CNN?
----------
The spatial structure matters: a taxi in zone (3,4) should weight nearby zones
more heavily than distant ones. A CNN naturally captures this locality — the
same convolutional filters apply everywhere on the map, learning "if there is
high demand to my north-east and few vehicles there, move there" regardless of
which zone the taxi is currently in.

Usage
-----
    net = ZoneCNNQNetwork(grid_rows=7, grid_cols=10)
    # state shape: (batch, 2, 7, 10)
    q_values = net(state_tensor)   # → (batch, 70)
    action = q_values.argmax(dim=-1)  # → zone_id per batch element
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class ZoneCNNQNetwork(nn.Module):
    """
    Convolutional Q-network that maps a (2, ROWS, COLS) spatial state
    to Q-values over all ROWS×COLS zone actions.

    Parameters
    ----------
    grid_rows : int
        Number of rows in the zone grid.
    grid_cols : int
        Number of columns in the zone grid.
    conv_channels : sequence of ints
        Number of output channels for each convolutional layer.
        Default: (16, 32, 64)
    fc_hidden : int
        Width of the fully-connected hidden layer after flattening.
    """

    def __init__(
        self,
        grid_rows: int = 7,
        grid_cols: int = 10,
        conv_channels: Sequence[int] = (16, 32, 64),
        fc_hidden: int = 256,
    ) -> None:
        super().__init__()

        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.n_zones = grid_rows * grid_cols

        # --- Convolutional backbone ---
        conv_layers: list[nn.Module] = []
        in_ch = 2  # demand + vehicle channels
        for out_ch in conv_channels:
            conv_layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ])
            in_ch = out_ch
        self.conv_backbone = nn.Sequential(*conv_layers)

        # --- Fully connected head ---
        flat_dim = in_ch * grid_rows * grid_cols
        self.fc_head = nn.Sequential(
            nn.Linear(flat_dim, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(fc_hidden, self.n_zones),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (batch, 2, ROWS, COLS)
            Stacked demand and vehicle density grids.

        Returns
        -------
        torch.Tensor, shape (batch, n_zones)
            Q-value for each zone action.
        """
        features = self.conv_backbone(x)          # (B, 64, R, C)
        flat = features.flatten(start_dim=1)       # (B, 64*R*C)
        q_values = self.fc_head(flat)              # (B, n_zones)
        return q_values


# ---------------------------------------------------------------------------
# Dueling architecture variant (optional upgrade)
# ---------------------------------------------------------------------------

class DuelingZoneCNNQNetwork(nn.Module):
    """
    Dueling-DQN variant of ZoneCNNQNetwork.

    Separates state value V(s) from advantage A(s,a):
        Q(s,a) = V(s) + A(s,a) - mean_a(A(s,a))

    This helps the network learn that some states are simply bad regardless
    of which zone the taxi moves to (e.g., when all zones have zero demand).
    """

    def __init__(
        self,
        grid_rows: int = 7,
        grid_cols: int = 10,
        conv_channels: Sequence[int] = (16, 32, 64),
        fc_hidden: int = 256,
    ) -> None:
        super().__init__()

        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.n_zones = grid_rows * grid_cols

        # Shared convolutional backbone
        conv_layers: list[nn.Module] = []
        in_ch = 2
        for out_ch in conv_channels:
            conv_layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ])
            in_ch = out_ch
        self.conv_backbone = nn.Sequential(*conv_layers)

        flat_dim = in_ch * grid_rows * grid_cols

        # Value stream: V(s) → scalar
        self.value_stream = nn.Sequential(
            nn.Linear(flat_dim, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(fc_hidden, 1),
        )

        # Advantage stream: A(s,a) → n_zones values
        self.advantage_stream = nn.Sequential(
            nn.Linear(flat_dim, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(fc_hidden, self.n_zones),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv_backbone(x)
        flat = features.flatten(start_dim=1)

        value = self.value_stream(flat)           # (B, 1)
        advantage = self.advantage_stream(flat)   # (B, n_zones)

        # Q = V + (A - mean(A))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values
