"""
zone_state.py
-------------
Builds the spatial state representation used by the zone-based DQN.

State = two 2-D grid layers stacked as a (2, ROWS, COLS) numpy array:

    Layer 0 — demand_grid[row, col]  : number of PENDING/DEFERRED requests
                                       whose pickup edge falls in that zone.
    Layer 1 — vehicle_grid[row, col] : number of IDLE taxis (no onboard
                                       passengers, no future stops) currently
                                       in that zone.

Both layers are normalised to [0, 1] by dividing by the respective fleet or
request count so that scale differences do not dominate learning.

Usage
-----
    tracker = ZoneStateTracker(zone_map, grid_rows, grid_cols)
    state = tracker.build_state(taxi_plans, requests)  # → (2, R, C) np.ndarray
"""

from __future__ import annotations

import sys
import os
from typing import Dict, Optional

import numpy as np

# Allow importing from RealDQN/DQNetwork without installing as a package
_REAL_DQN = os.path.join(os.path.dirname(__file__), "..", "RealDQN", "DQNetwork")
if _REAL_DQN not in sys.path:
    sys.path.insert(0, _REAL_DQN)

from DRTDataclass import Request, RequestStatus, TaxiPlan, TaxiStatus
from zone_map import ZoneMap


class ZoneStateTracker:
    """
    Converts the dispatcher's in-memory state into the spatial grid tensors
    that the CNN Q-network consumes.

    Parameters
    ----------
    zone_map : ZoneMap
        Pre-built mapping from edge IDs to zone grid cells.
    normalise : bool
        If True (default), each grid is divided by its maximum count so
        values lie in [0, 1].  Avoids dominating the network when there
        are many requests.
    """

    N_CHANNELS = 2  # demand + vehicle

    def __init__(self, zone_map: ZoneMap, normalise: bool = True) -> None:
        self.zone_map = zone_map
        self.rows = zone_map.grid_rows
        self.cols = zone_map.grid_cols
        self.normalise = normalise

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def build_state(
        self,
        taxi_plans: Dict[str, TaxiPlan],
        requests: Dict[str, Request],
    ) -> np.ndarray:
        """
        Build the (2, ROWS, COLS) state array.

        Parameters
        ----------
        taxi_plans : dict
            {taxi_id: TaxiPlan} as maintained by the dispatcher.
        requests : dict
            {person_id: Request} as maintained by the dispatcher.

        Returns
        -------
        np.ndarray, shape (2, ROWS, COLS), dtype float32
        """
        demand_grid = self._compute_demand_grid(requests)
        vehicle_grid = self._compute_vehicle_grid(taxi_plans)

        if self.normalise:
            demand_max = demand_grid.max()
            vehicle_max = vehicle_grid.max()
            if demand_max > 0:
                demand_grid = demand_grid / demand_max
            if vehicle_max > 0:
                vehicle_grid = vehicle_grid / vehicle_max

        state = np.stack([demand_grid, vehicle_grid], axis=0).astype(np.float32)
        return state  # shape: (2, ROWS, COLS)

    def demand_at_zone(
        self,
        zone_id: int,
        requests: Dict[str, Request],
    ) -> int:
        """Count pending/deferred requests in a given zone (unnormalised)."""
        count = 0
        for req in requests.values():
            if req.status not in (RequestStatus.PENDING, RequestStatus.DEFERRED):
                continue
            z = self.zone_map.edge_to_zone(req.from_edge)
            if z == zone_id:
                count += 1
        return count

    def vehicles_at_zone(
        self,
        zone_id: int,
        taxi_plans: Dict[str, TaxiPlan],
    ) -> int:
        """Count idle (no stops, no onboard) taxis in a given zone."""
        count = 0
        for plan in taxi_plans.values():
            if plan.onboard_count > 0 or plan.stops:
                continue
            z = self.zone_map.edge_to_zone(plan.current_edge)
            if z == zone_id:
                count += 1
        return count

    def taxi_zone(self, taxi_id: str, taxi_plans: Dict[str, TaxiPlan]) -> int:
        """Return the zone_id of a specific taxi, or -1 if unknown."""
        plan = taxi_plans.get(taxi_id)
        if plan is None:
            return -1
        return self.zone_map.edge_to_zone(plan.current_edge)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_demand_grid(self, requests: Dict[str, Request]) -> np.ndarray:
        """
        Build raw (ROWS, COLS) demand count grid.

        A request contributes to the zone containing its pickup edge (from_edge).
        Only PENDING and DEFERRED requests are counted.
        """
        grid = np.zeros((self.rows, self.cols), dtype=np.float32)

        for req in requests.values():
            if req.status not in (RequestStatus.PENDING, RequestStatus.DEFERRED):
                continue
            zone_id = self.zone_map.edge_to_zone(req.from_edge)
            if zone_id < 0:
                continue
            col, row = self.zone_map.zone_to_cell(zone_id)
            grid[row, col] += 1.0

        return grid

    def _compute_vehicle_grid(self, taxi_plans: Dict[str, TaxiPlan]) -> np.ndarray:
        """
        Build raw (ROWS, COLS) idle vehicle count grid.

        A taxi is counted as idle in a zone if:
          - onboard_count == 0  (no passengers inside)
          - stops is empty      (no planned pickups/dropoffs)

        Taxis with remaining_capacity > 0 but still serving stops are NOT
        counted as idle — they will finish their current plan before
        repositioning.
        """
        grid = np.zeros((self.rows, self.cols), dtype=np.float32)

        for plan in taxi_plans.values():
            if plan.onboard_count > 0 or plan.stops:
                continue  # not idle
            zone_id = self.zone_map.edge_to_zone(plan.current_edge)
            if zone_id < 0:
                continue
            col, row = self.zone_map.zone_to_cell(zone_id)
            grid[row, col] += 1.0

        return grid
