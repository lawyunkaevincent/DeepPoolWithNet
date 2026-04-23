from __future__ import annotations

import argparse
import json
import os
import random
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple


# ─── Day-schedule time windows ────────────────────────────────────────────────
# Each entry: (start_step, end_step, demand_key, label)
# Steps are simulation time units; 1440 total assumes 1 step = 1 minute (SUMO step-length=60).
TIME_WINDOWS: List[Tuple[int, int, str, str]] = [
    (0,    360,  "very_low",     "12am–6am   (very low)"),
    (360,  540,  "morning_peak", "6am–9am    (morning peak)"),
    (540,  960,  "medium",       "9am–4pm    (medium)"),
    (960,  1200, "evening_peak", "4pm–8pm    (evening peak)"),
    (1200, 1440, "low_medium",   "8pm–12am   (low to medium)"),
]

# Default relative demand multipliers (higher = more requests in that window).
DEFAULT_DEMAND: Dict[str, float] = {
    "very_low":     0.1,
    "morning_peak": 1.0,
    "medium":       0.5,
    "evening_peak": 1.0,
    "low_medium":   0.3,
}


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class EdgeStats:
    edge_id: str
    unreachable_count: int = 0
    reachable_to: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "EdgeStats":
        return cls(
            edge_id=data["edge_id"],
            unreachable_count=data.get("unreachable_count", 0),
            reachable_to=list(data.get("reachable_to", [])),
        )

    def reachable_count(self) -> int:
        return len(self.reachable_to)


class ConnectivityReport:
    def __init__(self, results: Dict[str, EdgeStats], total_candidates: int):
        self.results = results
        self.total_candidates = total_candidates
        self.edge_ids: Set[str] = set(results.keys())

    @classmethod
    def load_json(cls, path: str | Path) -> "ConnectivityReport":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        results = {
            edge_id: EdgeStats.from_dict(stats)
            for edge_id, stats in data["results"].items()
        }
        total_candidates = int(data.get("total_candidates", len(results)))
        return cls(results, total_candidates)

    def has_edge(self, edge_id: str) -> bool:
        return edge_id in self.results

    def stats(self, edge_id: str) -> Optional[EdgeStats]:
        return self.results.get(edge_id)

    def reachable_from(self, edge_id: str) -> List[str]:
        stats = self.results.get(edge_id)
        if stats is None:
            return []
        return stats.reachable_to

    def eligible_reachable_from(self, edge_id: str, min_reachable: int = 1) -> List[str]:
        return [
            candidate
            for candidate in self.reachable_from(edge_id)
            if self.has_edge(candidate) and self.results[candidate].reachable_count() >= min_reachable
        ]

    def top_edges_by_reachability(self, min_reachable: int = 1) -> List[str]:
        return [
            stats.edge_id
            for stats in sorted(
                (
                    s for s in self.results.values()
                    if s.reachable_count() >= min_reachable
                ),
                key=lambda s: (s.reachable_count(), -s.unreachable_count, s.edge_id),
                reverse=True,
            )
        ]


@dataclass
class TaxiAnchor:
    trip_id: str
    trip_from: Optional[str]
    trip_to: Optional[str]
    stop_edge: Optional[str]

    def ordered_edges(self, mode: str) -> List[str]:
        options: List[Optional[str]]
        if mode == "stop_first":
            options = [self.stop_edge, self.trip_to, self.trip_from]
        elif mode == "trip_to_first":
            options = [self.trip_to, self.stop_edge, self.trip_from]
        elif mode == "trip_from_first":
            options = [self.trip_from, self.trip_to, self.stop_edge]
        else:
            raise ValueError(f"Unknown anchor mode: {mode}")
        seen: Set[str] = set()
        out: List[str] = []
        for edge in options:
            if edge and edge not in seen:
                out.append(edge)
                seen.add(edge)
        return out


@dataclass
class RequestRide:
    person_id: str
    depart: float
    from_edge: str
    to_edge: str


# ─── Generator ────────────────────────────────────────────────────────────────

class RequestChainGenerator:
    def __init__(self, report: ConnectivityReport, rng: random.Random):
        self.report = report
        self.rng = rng

    # ── Static helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _lane_to_edge(lane_id: Optional[str]) -> Optional[str]:
        if not lane_id:
            return None
        if "_" not in lane_id:
            return lane_id
        return lane_id.rsplit("_", 1)[0]

    @staticmethod
    def read_taxi_anchor(taxi_file: str | Path) -> TaxiAnchor:
        tree = ET.parse(taxi_file)
        root = tree.getroot()

        for trip in root.findall("trip"):
            trip_id = trip.get("id", "taxi_0")
            trip_from = trip.get("from")
            trip_to = trip.get("to")
            stop = trip.find("stop")
            stop_edge = None
            if stop is not None:
                stop_edge = RequestChainGenerator._lane_to_edge(stop.get("lane"))
            return TaxiAnchor(
                trip_id=trip_id,
                trip_from=trip_from,
                trip_to=trip_to,
                stop_edge=stop_edge,
            )

        raise ValueError("No <trip> element found in taxi file.")

    @staticmethod
    def parse_net_coords(net_file: str | Path) -> Dict[str, Tuple[float, float]]:
        """Extract each edge's midpoint coordinate from a SUMO net.xml file.

        Junction-internal edges (IDs starting with ':') are skipped.
        Returns a dict mapping edge_id -> (x, y).
        """
        tree = ET.parse(net_file)
        root = tree.getroot()
        coords: Dict[str, Tuple[float, float]] = {}
        for edge_elem in root.findall("edge"):
            edge_id = edge_elem.get("id", "")
            if not edge_id or edge_id.startswith(":"):
                continue
            for lane in edge_elem.findall("lane"):
                shape_str = lane.get("shape", "")
                if not shape_str:
                    continue
                pts: List[Tuple[float, float]] = []
                for token in shape_str.split():
                    xy = token.split(",")
                    if len(xy) >= 2:
                        try:
                            pts.append((float(xy[0]), float(xy[1])))
                        except ValueError:
                            pass
                if pts:
                    coords[edge_id] = pts[len(pts) // 2]
                    break
        return coords

    # ── Internal helpers ────────────────────────────────────────────────────

    def _filter_existing_edges(self, edge_ids: Sequence[str]) -> List[str]:
        return [edge_id for edge_id in edge_ids if self.report.has_edge(edge_id)]

    def _choose(self, candidates: Sequence[str], label: str) -> str:
        if not candidates:
            raise ValueError(f"No candidates available for {label}.")
        return self.rng.choice(list(candidates))

    def _intersection(self, a: Sequence[str], b: Sequence[str]) -> List[str]:
        bset = set(b)
        return [x for x in a if x in bset]

    def _rank_by_reachability(self, edge_ids: Sequence[str]) -> List[str]:
        filtered = self._filter_existing_edges(edge_ids)
        return sorted(
            filtered,
            key=lambda edge_id: (
                self.report.results[edge_id].reachable_count(),
                -self.report.results[edge_id].unreachable_count,
                edge_id,
            ),
            reverse=True,
        )

    def _pick_anchor_edge(
        self,
        taxi_anchor: TaxiAnchor,
        anchor_mode: str,
        min_reachable_pickup: int,
    ) -> str:
        tried: List[str] = []
        for edge in taxi_anchor.ordered_edges(anchor_mode):
            tried.append(edge)
            if not self.report.has_edge(edge):
                continue
            if self.report.eligible_reachable_from(edge, min_reachable=min_reachable_pickup):
                return edge
        raise ValueError(
            "No usable taxi anchor edge found in the connectivity report. "
            f"Tried: {tried}. Either lower --min-reachable-pickup or regenerate the report with taxi edges included."
        )

    def _eligible_targets(self, edge_id: str, min_reachable_dropoff: int) -> List[str]:
        return self.report.eligible_reachable_from(edge_id, min_reachable=min_reachable_dropoff)

    def _sample_pair(
        self,
        eligible_from: Sequence[str],
        min_reachable_dropoff: int,
        trip_time_fn: Optional[Callable[[str, str], Optional[float]]],
        min_trip_time: float,
        max_retries: int,
        to_constraint: Optional[Set[str]] = None,
    ) -> Optional[Tuple[str, str]]:
        """Sample a (from, to) pair where from->to is reachable and (when
        trip_time_fn is supplied) the SUMO-routed direct trip time is
        >= min_trip_time seconds.

        If to_constraint is provided, to_edge must additionally belong to that set
        (used for cycle closure: the last ride's drop-off must reach the first
        ride's pickup).

        Returns None if no valid pair is found within max_retries attempts.
        """
        if not eligible_from:
            return None

        dtt_required = min_trip_time > 0 and trip_time_fn is not None
        from_list = list(eligible_from)

        for _ in range(max_retries):
            from_e = self.rng.choice(from_list)
            to_candidates = [
                t for t in self._eligible_targets(from_e, min_reachable_dropoff)
                if t != from_e
            ]
            if to_constraint is not None:
                to_candidates = [t for t in to_candidates if t in to_constraint]
            if not to_candidates:
                continue
            to_e = self.rng.choice(to_candidates)

            if not dtt_required:
                return from_e, to_e

            estimated = trip_time_fn(from_e, to_e)
            if estimated is None:
                # Edges unknown to the net or no routable path — fall through
                # and resample.
                continue
            if estimated >= min_trip_time:
                return from_e, to_e
        return None

    def _sample_depart_gap(
        self,
        depart_steps: Sequence[float],
        max_random_deviation_pct: float = 0.0,
    ) -> float:
        if not depart_steps:
            raise ValueError("depart_steps must contain at least one value.")

        base_step = float(self._choose([str(step) for step in depart_steps], "depart-step"))
        if base_step < 0:
            raise ValueError("depart-step values must be non-negative.")
        if max_random_deviation_pct < 0:
            raise ValueError("max_random_deviation_pct must be non-negative.")

        if max_random_deviation_pct == 0:
            return base_step

        deviation = base_step * (max_random_deviation_pct / 100.0)
        low = max(0.0, base_step - deviation)
        high = base_step + deviation
        return self.rng.uniform(low, high)

    # ── Stop selection (day mode) ────────────────────────────────────────────

    def select_stops(
        self,
        num_stops: int,
        min_reachable: int = 1,
        coords: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> List[str]:
        """Select num_stops edges that are all mutually reachable and spread across the map.

        "Mutually reachable" means every selected stop can reach and be reached
        from every other selected stop according to the connectivity report.

        Geographic spread strategy:
        - If coords (from --net) are provided: greedy farthest-first selection
          so stops maximise minimum distance to any already-chosen stop.
        - Without coords: spread evenly through the connectivity-ranked candidate
          list as a rough spatial proxy.

        Args:
            num_stops:     Number of stop edges to select (>= 2).
            min_reachable: Minimum reachable_count a candidate edge must have.
            coords:        Optional edge_id -> (x, y) map from parse_net_coords().

        Returns:
            List of selected edge IDs (sorted alphabetically for determinism).

        Raises:
            ValueError if a valid set of num_stops stops cannot be found.
        """
        if num_stops < 2:
            raise ValueError("num_stops must be at least 2.")

        candidates = [
            e for e, s in self.report.results.items()
            if s.reachable_count() >= min_reachable
        ]
        if len(candidates) < num_stops:
            raise ValueError(
                f"Only {len(candidates)} eligible edges (reachable_count >= {min_reachable}), "
                f"but {num_stops} stops requested. Lower --num-stops or --min-reachable-pickup."
            )

        # Build mutual reachability sets
        reach: Dict[str, Set[str]] = {
            e: set(self.report.results[e].reachable_to) for e in candidates
        }

        def mutually_reachable_with_all(edge: str, selected: List[str]) -> bool:
            return all(edge in reach[s] and s in reach[edge] for s in selected)

        # Pre-filter: keep only edges that have >= (num_stops - 1) mutual connections
        mutual_count: Dict[str, int] = {
            e: sum(1 for f in candidates if f != e and f in reach[e] and e in reach[f])
            for e in candidates
        }
        eligible = [e for e in candidates if mutual_count[e] >= num_stops - 1]

        if len(eligible) < num_stops:
            raise ValueError(
                f"Cannot find {num_stops} mutually reachable stops: "
                f"only {len(eligible)} edges have >= {num_stops - 1} mutual connections. "
                f"Lower --num-stops or --min-reachable-pickup."
            )

        # ── Greedy selection ────────────────────────────────────────────────
        if coords:
            eligible_geo = [e for e in eligible if e in coords]
            if len(eligible_geo) < num_stops:
                print(
                    f"[warn] Only {len(eligible_geo)} of {len(eligible)} eligible edges have "
                    f"coordinates in the net file (need {num_stops}). "
                    "Falling back to all eligible edges without geographic spread.",
                    file=sys.stderr,
                )
                eligible_geo = eligible

            # Seed: leftmost edge (smallest x)
            first = min(eligible_geo, key=lambda e: coords.get(e, (0.0, 0.0))[0])
            selected: List[str] = [first]

            while len(selected) < num_stops:
                valid = [
                    e for e in eligible_geo
                    if e not in selected and mutually_reachable_with_all(e, selected)
                ]
                if not valid:
                    raise ValueError(
                        f"Could not find {num_stops} mutually reachable, spread stops. "
                        f"Selected so far ({len(selected)}): {selected}. "
                        "Try reducing --num-stops."
                    )

                def _min_dist_to_selected(e: str) -> float:
                    ex, ey = coords.get(e, (0.0, 0.0))
                    return min(
                        ((ex - coords.get(s, (0.0, 0.0))[0]) ** 2
                         + (ey - coords.get(s, (0.0, 0.0))[1]) ** 2) ** 0.5
                        for s in selected
                    )

                selected.append(max(valid, key=_min_dist_to_selected))

        else:
            # No coordinates: sort by (mutual_count desc, edge_id) and spread evenly
            sorted_eligible = sorted(eligible, key=lambda e: (-mutual_count[e], e))
            rank_of: Dict[str, int] = {e: i for i, e in enumerate(sorted_eligible)}
            n = len(sorted_eligible)

            first = sorted_eligible[0]
            selected = [first]
            selected_set: Set[str] = {first}

            for i in range(1, num_stops):
                target_rank = int(i * n / num_stops)
                valid = [
                    e for e in sorted_eligible
                    if e not in selected_set and mutually_reachable_with_all(e, selected)
                ]
                if not valid:
                    raise ValueError(
                        f"Could not find {num_stops} mutually reachable stops. "
                        f"Found {len(selected)} so far: {selected}. "
                        "Try reducing --num-stops or supply --net for better geographic spread."
                    )
                best = min(valid, key=lambda e: abs(rank_of[e] - target_rank))
                selected.append(best)
                selected_set.add(best)

        return sorted(selected)

    # ── Day schedule generation ──────────────────────────────────────────────

    def generate_day_schedule(
        self,
        stops: List[str],
        num_requests: int,
        demand_weights: Optional[Dict[str, float]] = None,
        day_steps: int = 1440,
    ) -> List[RequestRide]:
        """Generate num_requests rides spread across a full simulated day.

        Requests are distributed across five named time windows (see TIME_WINDOWS).
        Within each window the depart times are uniformly sampled. Origin and
        destination are drawn uniformly at random from stops (origin != destination).
        The returned list is sorted by depart time with sequential person IDs.

        Args:
            stops:          Pool of stop edge IDs (must be >= 2, all mutually reachable).
            num_requests:   Total rides to generate across the whole day.
            demand_weights: Override any DEFAULT_DEMAND values.
                            Keys: very_low, morning_peak, medium, evening_peak, low_medium.
            day_steps:      Total simulation steps in one day.
                            Default 1440 (1 step = 1 minute, SUMO step-length=60).
                            Use 86400 for 1-second steps.

        Returns:
            List[RequestRide] sorted by depart time.
        """
        if len(stops) < 2:
            raise ValueError("Need at least 2 stops to generate rides.")
        if num_requests < 1:
            raise ValueError("num_requests must be at least 1.")

        weights = {**DEFAULT_DEMAND, **(demand_weights or {})}

        # Scale the canonical 1440-step windows to the requested day_steps
        scale = day_steps / 1440.0
        windows: List[Tuple[float, float, str, str]] = []
        for s, e, key, label in TIME_WINDOWS:
            windows.append((s * scale, e * scale, key, label))
        # Clamp last window end to exact day_steps to avoid floating-point drift
        last = windows[-1]
        windows[-1] = (last[0], float(day_steps), last[2], last[3])

        # Effective weight = window_duration × demand_multiplier
        window_weights = [(end - start) * weights[key] for start, end, key, _ in windows]
        total_weight = sum(window_weights)

        # Distribute num_requests proportionally; give remainder to last window
        allocated: List[int] = []
        remaining = num_requests
        for i, ww in enumerate(window_weights):
            if i == len(window_weights) - 1:
                allocated.append(remaining)
            else:
                n = round(num_requests * ww / total_weight)
                allocated.append(n)
                remaining -= n

        # Generate rides
        rides: List[RequestRide] = []
        for (start, end, key, _), n_window in zip(windows, allocated):
            for _ in range(n_window):
                depart = self.rng.uniform(start, end)
                from_edge = self.rng.choice(stops)
                to_choices = [s for s in stops if s != from_edge]
                to_edge = self.rng.choice(to_choices)
                rides.append(RequestRide(
                    person_id="",  # assigned after sort
                    depart=depart,
                    from_edge=from_edge,
                    to_edge=to_edge,
                ))

        rides.sort(key=lambda r: r.depart)
        for i, ride in enumerate(rides):
            ride.person_id = str(i)

        return rides

    # ── Day schedule generation (unbounded — full eligible edge set) ────────

    def generate_day_schedule_unbounded(
        self,
        num_requests: int,
        min_reachable_pickup: int = 1,
        min_reachable_dropoff: int = 1,
        trip_time_fn: Optional[Callable[[str, str], Optional[float]]] = None,
        min_trip_time: float = 0.0,
        min_trip_time_fallback: float = 0.0,
        demand_weights: Optional[Dict[str, float]] = None,
        day_steps: int = 1440,
        max_retries_per_request: int = 200,
        chain_requests: bool = True,
        close_cycle: bool = True,
    ) -> Tuple[List[RequestRide], List[str]]:
        """Day schedule over the FULL pool of eligible edges (no fixed stop set).

        Each request is sampled per-depart-time, chronologically, so that:
          * from_0 is drawn uniformly from every edge meeting
            min_reachable_pickup;
          * for i >= 1, from_i is drawn from
            eligible_reachable_from(to_{i-1})  — loose chain;
          * if close_cycle, the last ride's to_edge is restricted to edges that
            can reach from_0 (loose cycle closure).
        A minimum direct trip time is enforced via Euclidean distance /
        avg_speed when --net coords are supplied.

        Args:
            num_requests:          Total rides to generate across the whole day.
            min_reachable_pickup:  Pickup edge must have reachable_count >= this.
            min_reachable_dropoff: Dropoff edge must have reachable_count >= this.
            trip_time_fn:          Callable (from_id, to_id) -> seconds, typically
                                   built from sumolib's getShortestPath with a
                                   travel-time cost. Required when
                                   min_trip_time > 0.
            min_trip_time:         Minimum direct trip time in seconds. 0 disables
                                   the filter.
            demand_weights:        Override any DEFAULT_DEMAND values.
            day_steps:             Total simulation steps in one day.
            max_retries_per_request: Retry budget per sampled pair.
            chain_requests:        If True, from_i must be reachable from
                                   to_{i-1}. If False, sample pairs independently.
            close_cycle:           If True (and chain_requests), force the last
                                   ride's to_edge to be able to reach from_0.
                                   Falls back to unrestricted sampling with a
                                   warning if no such edge can be found.

        Returns:
            (rides, eligible_edges) — rides sorted by depart time, plus the
            eligible edge pool that was used (for summary printing).
        """
        if num_requests < 1:
            raise ValueError("num_requests must be at least 1.")

        eligible_edges = [
            e for e, s in self.report.results.items()
            if s.reachable_count() >= min_reachable_pickup
        ]
        if len(eligible_edges) < 2:
            raise ValueError(
                f"Only {len(eligible_edges)} eligible edges with reachable_count "
                f">= {min_reachable_pickup}. Need at least 2."
            )
        if min_trip_time > 0 and trip_time_fn is None:
            print(
                "[warn] --min-trip-time set but no trip_time_fn was supplied "
                "(did you forget --net?); pairs will be accepted without a "
                "dtt check.",
                file=sys.stderr,
            )

        weights = {**DEFAULT_DEMAND, **(demand_weights or {})}

        # Allocate per-window counts then materialise all depart times up front,
        # sorted ascending — chain semantics need chronological sampling.
        scale = day_steps / 1440.0
        windows: List[Tuple[float, float, str, str]] = [
            (s * scale, e * scale, key, label) for s, e, key, label in TIME_WINDOWS
        ]
        last = windows[-1]
        windows[-1] = (last[0], float(day_steps), last[2], last[3])

        window_weights = [(end - start) * weights[key] for start, end, key, _ in windows]
        total_weight = sum(window_weights)

        allocated: List[int] = []
        remaining = num_requests
        for i, ww in enumerate(window_weights):
            if i == len(window_weights) - 1:
                allocated.append(remaining)
            else:
                n = round(num_requests * ww / total_weight)
                allocated.append(n)
                remaining -= n

        depart_times: List[float] = []
        for (start, end, _, _), n_window in zip(windows, allocated):
            for _ in range(n_window):
                depart_times.append(self.rng.uniform(start, end))
        depart_times.sort()

        rides: List[RequestRide] = []
        first_from: Optional[str] = None
        cycle_targets: Optional[Set[str]] = None
        dropped = 0
        chain_fallbacks = 0
        cycle_fallback = False
        dtt_fallback_count = 0
        fallback_enabled = (
            min_trip_time > 0
            and 0 < min_trip_time_fallback < min_trip_time
            and trip_time_fn is not None
        )

        for i, depart in enumerate(depart_times):
            is_last = (i == len(depart_times) - 1)

            # ── Determine the candidate pool for from_i ─────────────────────
            if chain_requests and rides:
                prev_to = rides[-1].to_edge
                from_pool: Sequence[str] = self.report.eligible_reachable_from(
                    prev_to, min_reachable=min_reachable_pickup,
                )
                if not from_pool:
                    # Primary filter (min_reachable_pickup) left no candidates.
                    # Relax it, but ONLY to edges actually reachable from prev_to
                    # so the chain stays physically routable by SUMO.
                    relaxed = [
                        e for e in self.report.reachable_from(prev_to)
                        if self.report.has_edge(e)
                    ]
                    if not relaxed:
                        # prev_to has zero outgoing reachability — drop the slot
                        # rather than generate an unroutable chain link.
                        dropped += 1
                        continue
                    from_pool = relaxed
                    chain_fallbacks += 1
            else:
                from_pool = eligible_edges

            # ── Cycle-closure constraint on the last ride's to_edge ─────────
            to_constraint: Optional[Set[str]] = None
            if chain_requests and close_cycle and is_last and first_from is not None:
                if cycle_targets is None:
                    cycle_targets = {
                        e for e, stats in self.report.results.items()
                        if first_from in stats.reachable_to
                    }
                to_constraint = cycle_targets if cycle_targets else None

            # ── Tier 1: primary threshold, with cycle constraint if applicable ──
            pair = self._sample_pair(
                eligible_from=from_pool,
                min_reachable_dropoff=min_reachable_dropoff,
                trip_time_fn=trip_time_fn,
                min_trip_time=min_trip_time,
                max_retries=max_retries_per_request,
                to_constraint=to_constraint,
            )
            used_fallback = False
            used_cycle_fallback = False

            # ── Tier 2: same threshold but drop the cycle constraint ──
            if pair is None and to_constraint is not None:
                used_cycle_fallback = True
                pair = self._sample_pair(
                    eligible_from=from_pool,
                    min_reachable_dropoff=min_reachable_dropoff,
                    trip_time_fn=trip_time_fn,
                    min_trip_time=min_trip_time,
                    max_retries=max_retries_per_request,
                    to_constraint=None,
                )

            # ── Tier 3: fallback dtt threshold (still attempt cycle first) ──
            if pair is None and fallback_enabled:
                pair = self._sample_pair(
                    eligible_from=from_pool,
                    min_reachable_dropoff=min_reachable_dropoff,
                    trip_time_fn=trip_time_fn,
                    min_trip_time=min_trip_time_fallback,
                    max_retries=max_retries_per_request,
                    to_constraint=to_constraint,
                )
                if pair is not None:
                    used_fallback = True
                    # Tier 3a kept the cycle constraint, so Tier 2's speculative
                    # flag is incorrect — reset it.
                    used_cycle_fallback = False
                elif to_constraint is not None:
                    pair = self._sample_pair(
                        eligible_from=from_pool,
                        min_reachable_dropoff=min_reachable_dropoff,
                        trip_time_fn=trip_time_fn,
                        min_trip_time=min_trip_time_fallback,
                        max_retries=max_retries_per_request,
                        to_constraint=None,
                    )
                    if pair is not None:
                        used_fallback = True
                        used_cycle_fallback = True

            if pair is None:
                dropped += 1
                continue

            if used_cycle_fallback:
                cycle_fallback = True
            if used_fallback:
                dtt_fallback_count += 1

            from_e, to_e = pair
            if first_from is None:
                first_from = from_e
            rides.append(RequestRide(
                person_id="",
                depart=depart,
                from_edge=from_e,
                to_edge=to_e,
            ))
            dtt_str = ""
            if trip_time_fn is not None and min_trip_time > 0:
                est = trip_time_fn(from_e, to_e)
                if est is not None:
                    dtt_str = f" dtt={est:.1f}s"
            tag = " [fallback]" if used_fallback else ""
            print(
                f"[gen]{tag} {len(rides)}/{len(depart_times)}  "
                f"depart={depart:.1f}s  {from_e} -> {to_e}{dtt_str}",
                flush=True,
            )

        # depart_times was pre-sorted, so rides are already chronological.
        for i, ride in enumerate(rides):
            ride.person_id = str(i)

        if chain_fallbacks:
            print(
                f"[warn] chain broken at {chain_fallbacks} request(s): "
                "previous dropoff had no eligible next pickup; sampled from the "
                "full pool at those points.",
                file=sys.stderr,
            )
        if cycle_fallback:
            print(
                "[warn] Could not close the cycle within retry budget; "
                "last ride's drop-off was sampled without the cycle constraint.",
                file=sys.stderr,
            )
        if dtt_fallback_count:
            print(
                f"[warn] {dtt_fallback_count} request(s) accepted at the fallback "
                f"dtt threshold ({min_trip_time_fallback:.0f}s) because the "
                f"primary threshold ({min_trip_time:.0f}s) exhausted its retries.",
                file=sys.stderr,
            )
        if dropped:
            print(
                f"[warn] {dropped} request(s) dropped (retries exhausted). "
                "Consider lowering --min-trip-time, setting "
                "--min-trip-time-fallback, or lowering --min-reachable-pickup.",
                file=sys.stderr,
            )

        return rides, eligible_edges

    # ── Chain generation (original mode) ────────────────────────────────────

    def generate_chain(
        self,
        num_requests: int,
        taxi_anchor: TaxiAnchor,
        anchor_mode: str = "stop_first",
        first_pool_top_k: int = 5,
        depart_start: float = 0.0,
        depart_steps: Sequence[float] = (100.0,),
        max_random_deviation_pct: float = 0.0,
        close_cycle: bool = True,
        unique_person_ids: bool = True,
        min_reachable_pickup: int = 1,
        min_reachable_dropoff: int = 1,
    ) -> tuple[str, List[RequestRide]]:
        if num_requests < 1:
            raise ValueError("num_requests must be at least 1.")
        if min_reachable_pickup < 0 or min_reachable_dropoff < 0:
            raise ValueError("Minimum reachable thresholds must be non-negative.")
        if not depart_steps:
            raise ValueError("depart_steps must contain at least one value.")
        if any(step < 0 for step in depart_steps):
            raise ValueError("All depart_steps must be non-negative.")
        if max_random_deviation_pct < 0:
            raise ValueError("max_random_deviation_pct must be non-negative.")

        anchor_edge = self._pick_anchor_edge(
            taxi_anchor=taxi_anchor,
            anchor_mode=anchor_mode,
            min_reachable_pickup=min_reachable_pickup,
        )

        rides: List[RequestRide] = []
        anchor_reachable = self.report.eligible_reachable_from(
            anchor_edge,
            min_reachable=min_reachable_pickup,
        )
        if not anchor_reachable:
            raise ValueError(
                f"Taxi anchor edge '{anchor_edge}' has no eligible pickup candidates. "
                "Try lowering --min-reachable-pickup."
            )

        top_edges = self.report.top_edges_by_reachability(min_reachable=min_reachable_pickup)
        top_edges = top_edges[: max(first_pool_top_k, 1)]
        first_from_candidates = self._intersection(anchor_reachable, top_edges)
        if not first_from_candidates:
            first_from_candidates = self._rank_by_reachability(anchor_reachable)[: max(first_pool_top_k, 1)]
        if not first_from_candidates:
            raise ValueError("No valid candidates for the first request pickup edge.")

        first_from = self._choose(first_from_candidates, "first request from-edge")
        first_to_candidates = self._eligible_targets(first_from, min_reachable_dropoff)
        if not first_to_candidates:
            raise ValueError(
                f"Chosen first pickup edge '{first_from}' has no reachable dropoff candidates that satisfy "
                f"min reachable count {min_reachable_dropoff}."
            )
        first_to = self._choose(self._rank_by_reachability(first_to_candidates), "first request to-edge")

        current_depart = depart_start
        rides.append(
            RequestRide(
                person_id="0" if unique_person_ids else "req",
                depart=current_depart,
                from_edge=first_from,
                to_edge=first_to,
            )
        )

        edges_that_can_reach_first_from = {
            edge_id
            for edge_id, stats in self.report.results.items()
            if first_from in stats.reachable_to
        }

        for idx in range(1, num_requests):
            prev_to = rides[-1].to_edge
            from_candidates = self.report.eligible_reachable_from(
                prev_to,
                min_reachable=min_reachable_pickup,
            )
            if not from_candidates:
                raise ValueError(
                    f"Previous dropoff edge '{prev_to}' has no eligible candidates for next pickup. "
                    "Try lowering --min-reachable-pickup."
                )

            ranked_from_candidates = self._rank_by_reachability(from_candidates)
            current_from = self._choose(ranked_from_candidates, f"request {idx} from-edge")
            to_candidates = self._eligible_targets(current_from, min_reachable_dropoff)
            if not to_candidates:
                raise ValueError(
                    f"Chosen pickup edge '{current_from}' has no reachable dropoff candidates satisfying "
                    f"min reachable count {min_reachable_dropoff}."
                )

            if close_cycle and idx == num_requests - 1:
                cycle_candidates = self._intersection(to_candidates, list(edges_that_can_reach_first_from))
                if cycle_candidates:
                    current_to = self._choose(
                        self._rank_by_reachability(cycle_candidates),
                        "last request to-edge (cycle closing)",
                    )
                else:
                    current_to = self._choose(
                        self._rank_by_reachability(to_candidates),
                        "last request to-edge (fallback)",
                    )
            else:
                current_to = self._choose(
                    self._rank_by_reachability(to_candidates),
                    f"request {idx} to-edge",
                )

            current_depart += self._sample_depart_gap(
                depart_steps=depart_steps,
                max_random_deviation_pct=max_random_deviation_pct,
            )

            rides.append(
                RequestRide(
                    person_id=str(idx) if unique_person_ids else "req",
                    depart=current_depart,
                    from_edge=current_from,
                    to_edge=current_to,
                )
            )

        return anchor_edge, rides

    # ── Output ───────────────────────────────────────────────────────────────

    @staticmethod
    def write_requests_file(rides: Sequence[RequestRide], output_file: str | Path) -> None:
        root = ET.Element(
            "routes",
            {
                "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
                "xsi:noNamespaceSchemaLocation": "http://sumo.dlr.de/xsd/routes_file.xsd",
            },
        )

        for ride in rides:
            person = ET.SubElement(
                root,
                "person",
                {
                    "id": ride.person_id,
                    "depart": str(round(ride.depart)),
                },
            )
            ET.SubElement(
                person,
                "ride",
                {
                    "from": ride.from_edge,
                    "to": ride.to_edge,
                    "lines": "taxi",
                },
            )

        tree = ET.ElementTree(root)
        ET.indent(tree, space="    ")
        tree.write(output_file, encoding="utf-8", xml_declaration=True)


def _load_sumolib():
    """Import sumolib, falling back to $SUMO_HOME/tools if necessary."""
    try:
        import sumolib  # type: ignore
        return sumolib
    except ImportError:
        sumo_home = os.environ.get("SUMO_HOME")
        if not sumo_home:
            raise ImportError(
                "Could not import sumolib. Install SUMO and set the SUMO_HOME "
                "environment variable, or `pip install sumolib`."
            )
        tools = os.path.join(sumo_home, "tools")
        if tools not in sys.path:
            sys.path.append(tools)
        import sumolib  # type: ignore
        return sumolib


def build_trip_time_estimator(
    net_file: str | Path,
) -> Callable[[str, str], Optional[float]]:
    """Return an estimator that uses sumolib's shortest-path router to compute
    the free-flow travel time between two edges (seconds).

    sumolib's getShortestPath minimises distance. We then sum
    edge_length / max(edge_speed, 0.1) along the returned path to obtain a
    free-flow travel time in seconds. This is an upper bound on the fastest
    path's travel time but is close to it for typical urban networks.

    Results are memoised so repeated queries are cheap.
    """
    sumolib = _load_sumolib()
    print(f"[sumolib] Loading net for routing: {net_file}")
    net = sumolib.net.readNet(str(net_file))

    cache: Dict[Tuple[str, str], Optional[float]] = {}

    def estimator(from_id: str, to_id: str) -> Optional[float]:
        key = (from_id, to_id)
        if key in cache:
            return cache[key]
        try:
            from_edge = net.getEdge(from_id)
            to_edge = net.getEdge(to_id)
        except KeyError:
            cache[key] = None
            return None
        result = net.getShortestPath(from_edge, to_edge)
        path = result[0] if result else None
        if not path:
            cache[key] = None
            return None
        travel_time = sum(
            e.getLength() / max(e.getSpeed(), 0.1) for e in path
        )
        cache[key] = float(travel_time)
        return float(travel_time)

    return estimator


def write_stops_file(stops: List[str], path: str | Path) -> None:
    """Save the selected stop edge IDs to a JSON file.

    The file contains a single object with a ``stops`` list so it is easy
    to load and iterate when building a taxi.rou.xml:

    .. code-block:: python

        import json
        data = json.load(open("stops.json"))
        for edge in data["stops"]:
            ...

    Args:
        stops: Ordered list of selected stop edge IDs.
        path:  Destination file path (will be created or overwritten).
    """
    payload = {"num_stops": len(stops), "stops": stops}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate SUMO taxi request files from a connectivity report.\n\n"
            "MODES\n"
            "  day   (default) Distribute requests across a full day using time-varying\n"
            "                  demand. Stops are selected to be fully connected and\n"
            "                  spread across the map.\n"
            "  chain           Legacy chained-ride mode anchored to a taxi route file.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Shared ──────────────────────────────────────────────────────────────
    parser.add_argument("--report", required=True, help="Path to connectivity_report.json")
    parser.add_argument("--output", required=True, help="Path to output requests .rou.xml")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--mode",
        choices=["day", "chain"],
        default="day",
        help="Generation mode: 'day' (full-day schedule) or 'chain' (legacy). Default: day",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=100,
        help="Total number of person requests to generate (default: 100)",
    )
    parser.add_argument(
        "--min-reachable-pickup",
        type=int,
        default=1,
        help="Only use pickup edges whose reachable_count >= this value (default: 1)",
    )
    parser.add_argument(
        "--min-reachable-dropoff",
        type=int,
        default=1,
        help="Only use dropoff edges whose reachable_count >= this value (default: 1)",
    )

    # ── Day mode ────────────────────────────────────────────────────────────
    day_group = parser.add_argument_group(
        "day mode",
        "Options for --mode day (full-day schedule with stop pool)",
    )
    day_group.add_argument(
        "--num-stops",
        type=int,
        default=5,
        help=(
            "Number of stop edges in the ride pool. "
            "All stops will be mutually reachable and spread across the map. "
            "Set to 0 to disable the pool and sample origin/destination per "
            "request from every edge meeting --min-reachable-pickup (each request "
            "is still guaranteed to be from->to reachable). (default: 5)"
        ),
    )
    day_group.add_argument(
        "--min-trip-time",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help=(
            "Minimum free-flow direct trip time (in seconds) required between a "
            "request's origin and destination. Computed by SUMO's router "
            "(sumolib.net.getShortestPath with edge_length / edge_speed as the "
            "cost). Requires --net. 0 disables the filter. (default: 0.0)"
        ),
    )
    day_group.add_argument(
        "--min-trip-time-fallback",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help=(
            "Fallback dtt threshold used when the primary --min-trip-time "
            "exhausts its retry budget for a given depart slot. Must be less "
            "than --min-trip-time. Requests accepted at this level are tagged "
            "[fallback] in the progress log. 0 disables the fallback "
            "(requests that can't meet --min-trip-time are dropped). "
            "(default: 0.0)"
        ),
    )
    day_group.add_argument(
        "--no-chain-requests",
        action="store_true",
        help=(
            "Unbounded day mode (--num-stops 0) only. Disables loose-chain "
            "sampling; each request's from/to pair is drawn independently "
            "(from->to reachability is still enforced per request)."
        ),
    )
    day_group.add_argument(
        "--no-close-cycle-day",
        action="store_true",
        help=(
            "Unbounded day mode (--num-stops 0) only. Disables the cycle-closing "
            "constraint on the last request's drop-off edge."
        ),
    )
    day_group.add_argument(
        "--net",
        metavar="NET_XML",
        default=None,
        help=(
            "Path to SUMO net.xml. "
            "Used to read edge coordinates for even geographic distribution of stops. "
            "Strongly recommended; without it stops are spread by connectivity rank only."
        ),
    )
    day_group.add_argument(
        "--day-steps",
        type=int,
        default=1440,
        help=(
            "Total simulation steps in one day (default: 1440). "
            "Use 1440 with SUMO step-length=60 (1 step = 1 minute). "
            "Use 86400 with step-length=1 (1 step = 1 second)."
        ),
    )
    day_group.add_argument(
        "--save-stops",
        metavar="STOPS_JSON",
        default=None,
        help=(
            "Optional path to save the selected stop edges as a JSON file "
            "(e.g. stops.json). The file lists the edge IDs so you can use "
            "them later to build a taxi.rou.xml."
        ),
    )

    # ── Demand multipliers (day mode) ────────────────────────────────────────
    demand_group = parser.add_argument_group(
        "demand profile",
        (
            "Relative demand multipliers for each time window (day mode only). "
            "Higher values generate proportionally more requests in that window."
        ),
    )
    demand_group.add_argument(
        "--demand-very-low",
        type=float,
        default=DEFAULT_DEMAND["very_low"],
        metavar="MULT",
        help=f"12am–6am demand multiplier  (default: {DEFAULT_DEMAND['very_low']})",
    )
    demand_group.add_argument(
        "--demand-morning-peak",
        type=float,
        default=DEFAULT_DEMAND["morning_peak"],
        metavar="MULT",
        help=f"6am–9am  demand multiplier  (default: {DEFAULT_DEMAND['morning_peak']})",
    )
    demand_group.add_argument(
        "--demand-medium",
        type=float,
        default=DEFAULT_DEMAND["medium"],
        metavar="MULT",
        help=f"9am–4pm  demand multiplier  (default: {DEFAULT_DEMAND['medium']})",
    )
    demand_group.add_argument(
        "--demand-evening-peak",
        type=float,
        default=DEFAULT_DEMAND["evening_peak"],
        metavar="MULT",
        help=f"4pm–8pm  demand multiplier  (default: {DEFAULT_DEMAND['evening_peak']})",
    )
    demand_group.add_argument(
        "--demand-low-medium",
        type=float,
        default=DEFAULT_DEMAND["low_medium"],
        metavar="MULT",
        help=f"8pm–12am demand multiplier  (default: {DEFAULT_DEMAND['low_medium']})",
    )

    # ── Chain mode ──────────────────────────────────────────────────────────
    chain_group = parser.add_argument_group(
        "chain mode",
        "Options for --mode chain (legacy chained-ride generation)",
    )
    chain_group.add_argument(
        "--taxi",
        default=None,
        help="Path to taxi.rou.xml (required for chain mode)",
    )
    chain_group.add_argument(
        "--depart-start",
        type=float,
        default=0.0,
        help="First request depart time (default: 0.0)",
    )
    chain_group.add_argument(
        "--depart-step",
        type=float,
        nargs="+",
        default=[100.0],
        help="One or more depart gaps between requests (default: 100). Example: --depart-step 50 100 200",
    )
    chain_group.add_argument(
        "--max-random-deviation-pct",
        type=float,
        default=0.0,
        help="Max random deviation %% applied to the chosen depart-step (default: 0.0)",
    )
    chain_group.add_argument(
        "--first-top-k",
        type=int,
        default=5,
        help="Prefer first pickup from the top-k most reachable edges near the taxi anchor (default: 5)",
    )
    chain_group.add_argument(
        "--anchor-mode",
        choices=["stop_first", "trip_to_first", "trip_from_first"],
        default="stop_first",
        help="How to pick the taxi anchor edge from taxi.rou.xml (default: stop_first)",
    )
    chain_group.add_argument(
        "--no-close-cycle",
        action="store_true",
        help="Disable the last-request cycle-closing preference",
    )

    return parser.parse_args()


def _print_day_summary(
    stops: List[str],
    rides: List[RequestRide],
    demand_weights: Dict[str, float],
    day_steps: int,
    output: str,
) -> None:
    scale = day_steps / 1440.0
    print(f"Mode            : day")
    print(f"Stops selected  : {len(stops)}")
    for i, s in enumerate(stops):
        print(f"  stop[{i}]       : {s}")
    print(f"Total requests  : {len(rides)}")
    print(f"Day steps       : {day_steps}")
    print(f"Output          : {output}")
    print()
    print("Demand profile and request distribution:")
    total_weight = sum(
        (int(e * scale) - int(s * scale)) * demand_weights[key]
        for s, e, key, _ in TIME_WINDOWS
    )
    for s, e, key, label in TIME_WINDOWS:
        ws, we = int(s * scale), int(e * scale)
        ww = (we - ws) * demand_weights[key]
        n = sum(1 for r in rides if ws <= r.depart < we) + (
            1 if rides and rides[-1].depart == we else 0
        )
        # Simpler: just count
        n = sum(1 for r in rides if ws <= r.depart < (we if e < 1440 else day_steps + 1))
        pct = ww / total_weight * 100 if total_weight > 0 else 0
        print(f"  {label}  mult={demand_weights[key]:.2f}  target={pct:.1f}%  generated={n}")


def _print_day_summary_unbounded(
    eligible_edges: List[str],
    rides: List[RequestRide],
    demand_weights: Dict[str, float],
    day_steps: int,
    output: str,
    min_trip_time: float,
    min_trip_time_fallback: float,
    min_reachable_pickup: int,
    min_reachable_dropoff: int,
    chain_requests: bool,
    close_cycle: bool,
) -> None:
    scale = day_steps / 1440.0
    print(f"Mode            : day (unbounded)")
    print(f"Eligible edges  : {len(eligible_edges)}  "
          f"(reachable_count >= {min_reachable_pickup})")
    print(f"Dropoff thresh. : reachable_count >= {min_reachable_dropoff}")
    print(f"Chain requests  : {'yes (loose: from_i reachable from to_(i-1))' if chain_requests else 'no'}")
    print(f"Cycle closure   : {'yes (to_last reaches from_0)' if (chain_requests and close_cycle) else 'no'}")
    if min_trip_time > 0:
        print(f"Min trip time   : {min_trip_time:.1f}s  (via sumolib shortest-path, "
              f"edge_length / edge_speed)")
        if 0 < min_trip_time_fallback < min_trip_time:
            print(f"  fallback      : {min_trip_time_fallback:.1f}s "
                  f"(used when primary exhausts retries)")
    else:
        print(f"Min trip time   : disabled (0s)")
    print(f"Total requests  : {len(rides)}")
    print(f"Day steps       : {day_steps}")
    print(f"Output          : {output}")
    print()
    print("Demand profile and request distribution:")
    total_weight = sum(
        (int(e * scale) - int(s * scale)) * demand_weights[key]
        for s, e, key, _ in TIME_WINDOWS
    )
    for s, e, key, label in TIME_WINDOWS:
        ws, we = int(s * scale), int(e * scale)
        ww = (we - ws) * demand_weights[key]
        n = sum(1 for r in rides if ws <= r.depart < (we if e < 1440 else day_steps + 1))
        pct = ww / total_weight * 100 if total_weight > 0 else 0
        print(f"  {label}  mult={demand_weights[key]:.2f}  target={pct:.1f}%  generated={n}")


def main() -> None:
    args = parse_args()
    report = ConnectivityReport.load_json(args.report)
    generator = RequestChainGenerator(report, random.Random(args.seed))

    if args.mode == "day":
        # ── Day mode ────────────────────────────────────────────────────────
        coords: Optional[Dict[str, Tuple[float, float]]] = None
        if args.net:
            coords = RequestChainGenerator.parse_net_coords(args.net)
            print(f"Loaded {len(coords)} edge coordinates from {args.net}")
        else:
            print(
                "[info] --net not provided. Stops will be spread by connectivity rank only. "
                "Supply --net <net.xml> for true geographic distribution.",
                file=sys.stderr,
            )

        demand_weights: Dict[str, float] = {
            "very_low":     args.demand_very_low,
            "morning_peak": args.demand_morning_peak,
            "medium":       args.demand_medium,
            "evening_peak": args.demand_evening_peak,
            "low_medium":   args.demand_low_medium,
        }

        if args.num_stops == 0:
            # ── Unbounded: sample per-request from the full eligible edge set ──
            trip_time_fn: Optional[Callable[[str, str], Optional[float]]] = None
            if args.min_trip_time > 0:
                if not args.net:
                    print(
                        "Error: --min-trip-time requires --net <net.xml> so SUMO's "
                        "router can compute travel times.",
                        file=sys.stderr,
                    )
                    sys.exit(1)
                trip_time_fn = build_trip_time_estimator(args.net)

            rides, eligible_edges = generator.generate_day_schedule_unbounded(
                num_requests=args.num_requests,
                min_reachable_pickup=args.min_reachable_pickup,
                min_reachable_dropoff=args.min_reachable_dropoff,
                trip_time_fn=trip_time_fn,
                min_trip_time=args.min_trip_time,
                min_trip_time_fallback=args.min_trip_time_fallback,
                demand_weights=demand_weights,
                day_steps=args.day_steps,
                chain_requests=not args.no_chain_requests,
                close_cycle=not args.no_close_cycle_day,
            )

            if args.save_stops:
                write_stops_file(eligible_edges, args.save_stops)
                print(f"Eligible edges saved : {args.save_stops}")

            generator.write_requests_file(rides, args.output)
            _print_day_summary_unbounded(
                eligible_edges=eligible_edges,
                rides=rides,
                demand_weights=demand_weights,
                day_steps=args.day_steps,
                output=args.output,
                min_trip_time=args.min_trip_time,
                min_trip_time_fallback=args.min_trip_time_fallback,
                min_reachable_pickup=args.min_reachable_pickup,
                min_reachable_dropoff=args.min_reachable_dropoff,
                chain_requests=not args.no_chain_requests,
                close_cycle=not args.no_close_cycle_day,
            )
        else:
            stops = generator.select_stops(
                num_stops=args.num_stops,
                min_reachable=args.min_reachable_pickup,
                coords=coords,
            )

            if args.save_stops:
                write_stops_file(stops, args.save_stops)
                print(f"Stops saved     : {args.save_stops}")

            rides = generator.generate_day_schedule(
                stops=stops,
                num_requests=args.num_requests,
                demand_weights=demand_weights,
                day_steps=args.day_steps,
            )

            generator.write_requests_file(rides, args.output)
            _print_day_summary(stops, rides, demand_weights, args.day_steps, args.output)

    else:
        # ── Chain mode ──────────────────────────────────────────────────────
        if args.taxi is None:
            print("Error: --taxi is required for chain mode.", file=sys.stderr)
            sys.exit(1)

        taxi_anchor = RequestChainGenerator.read_taxi_anchor(args.taxi)
        anchor_edge, rides = generator.generate_chain(
            num_requests=args.num_requests,
            taxi_anchor=taxi_anchor,
            anchor_mode=args.anchor_mode,
            first_pool_top_k=args.first_top_k,
            depart_start=args.depart_start,
            depart_steps=args.depart_step,
            max_random_deviation_pct=args.max_random_deviation_pct,
            close_cycle=not args.no_close_cycle,
            min_reachable_pickup=args.min_reachable_pickup,
            min_reachable_dropoff=args.min_reachable_dropoff,
        )
        generator.write_requests_file(rides, args.output)

        print(f"Mode            : chain")
        print(f"Taxi anchor edge: {anchor_edge}")
        print(f"Total requests  : {len(rides)}")
        print(f"First request   : {rides[0].from_edge} -> {rides[0].to_edge}")
        print(f"Last request    : {rides[-1].from_edge} -> {rides[-1].to_edge}")
        print(f"Output          : {args.output}")
        print(
            f"Thresholds used : pickup >= {args.min_reachable_pickup}, "
            f"dropoff >= {args.min_reachable_dropoff}"
        )


if __name__ == "__main__":
    main()
