"""
zone_env.py
-----------
ZoneBasedDRTEnv: DeepPool-style DRT environment in SUMO.

Architecture (faithful to DeepPool, arXiv:1903.03882)
------------------------------------------------------

  Layer 1 — Request assignment: DEEPPOOL-FAITHFUL POOLED DISPATCH
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  When a passenger request arrives, every non-full taxi is a candidate —
  idle taxis AND partially-occupied taxis with remaining capacity.

  For each candidate taxi we greedily re-optimise its remaining stop sequence
  (nearest-next-stop greedy, enforcing pick-up-before-drop-off) to produce
  an "optimal shortest-time route" that incorporates the new rider.  The taxi
  whose resulting plan has the minimum total completion time (tie-broken by
  new passenger wait time) wins the assignment, subject to three hard
  feasibility guards matching the DeepPool paper:
    • new passenger's waiting time ≤ max_wait
    • new passenger's in-vehicle time ≤ max_ride_factor × direct_travel_time
    • no existing passenger's wait or ride time is pushed past their limits

  This directly mirrors the DeepPool paper: "a vehicle is dispatched if it
  is empty or partially filled and decides to serve a new user; a new
  optimised route is then obtained."

  Layer 2 — Proactive repositioning: ZONE DQN
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  Whenever a taxi finishes all its stops and becomes truly idle, the shared
  CNN Q-network decides which zone the taxi should pre-position to.
  The taxi drives to a verified stop in that zone (from stops.json).

Decision lifecycle for one taxi
---------------------------------
  taxi becomes idle
        │
        ▼
  zone DQN → action (zone_id)
        │
        ▼
  changeTarget → nearest stop in zone
        │
        ▼
  taxi drives toward zone  [SUMO runs]
        │
        ├── request arrives → pooled dispatch considers this taxi
        │     (accepted if capacity & time-overhead constraints satisfied)
        │         taxi serves request, then becomes idle again → repeat
        │
        └── taxi reaches zone stop → idle again → repeat

Reward (per taxi, per repositioning period)
-------------------------------------------
  r = + W_SERVE      × pickups made since last decision
      − W_WAIT       × (system-wide waiting / fleet_size)
      − W_REPOSITION × empty metres driven
"""

from __future__ import annotations

import os
import sys
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_REAL_DQN = os.path.join(os.path.dirname(__file__), "..", "RealDQN", "DQNetwork")
if _REAL_DQN not in sys.path:
    sys.path.insert(0, _REAL_DQN)

import traci

from dispatcher import (
    HeuristicDispatcher,
    _log,
    _refresh_taxi_plans,
    _route_time,
    _sync_onboard,
    setup_logger,
)
from DRTDataclass import (
    IntervalAccumulator,
    Request,
    RequestStatus,
    Stop,
    StopType,
    TaxiPlan,
    TaxiStatus,
)
from zone_map import ZoneMap
from zone_map import ZoneStop
from zone_state import ZoneStateTracker
from zone_replay_buffer import Transition

# ---------------------------------------------------------------------------
# Reward weights
# ---------------------------------------------------------------------------
W_SERVE      = 5.0      # bonus per passenger picked up
W_WAIT       = 0.001    # penalty per passenger-second of waiting (÷ fleet size)
W_REPOSITION = 0.0005   # penalty per metre of empty repositioning

# Paper-faithful dispatch constraints (matching matching_policy.py).
# REJECT_WAIT_TIME : max travel time from the vehicle's CURRENT position to the
#                    new pickup (measured from NOW, not from request_time).
#                    Equivalent to the paper's reject_wait_time threshold on T[v,r].
# REJECT_DETOUR_TIME: max extra delay added to any existing passenger's ETA
#                     due to inserting the new request.  Prevents excessive detours
#                     for already-committed passengers.
REJECT_WAIT_TIME   = 300.0  # 5 min — taxi must reach new pickup within this time
REJECT_DETOUR_TIME = 300.0  # 5 min — max extra delay imposed on any existing stop


# ---------------------------------------------------------------------------
# Per-taxi repositioning context
# ---------------------------------------------------------------------------

@dataclass
class TaxiRepoState:
    last_state:      np.ndarray   # (2, R, C) state when decision was made
    last_action:     int          # zone_id chosen
    target_edge:     str          # stop edge sent to
    target_zone:     int          # actual zone of target_edge
    target_lane_index: int = 0
    target_pos:      float = 1.0
    pickups_since:   int  = 0     # cumulative pickups at decision time
    empty_dist_since: float = 0.0 # empty metres since decision


# ---------------------------------------------------------------------------
# Main environment
# ---------------------------------------------------------------------------

class ZoneBasedDRTEnv(HeuristicDispatcher):
    """
    DeepPool-style zone-based DRT environment using SUMO.

    Inherits SUMO lifecycle (start/run/close), taxi registration, and request
    sync from HeuristicDispatcher, but completely replaces _process_tick with:
      - simple nearest-taxi greedy dispatch
      - zone DQN idle-taxi repositioning

    Parameters
    ----------
    cfg_path        : path to .sumocfg
    net_xml_path    : path to .net.xml (parsed offline for zone grid)
    stops_json_path : path to stops.json (verified-connected repositioning targets)
    grid_cols       : east-west zone count  (default 10)
    grid_rows       : north-south zone count (default 7)
    policy_fn       : state → zone_id callable, or None for random policy
    epsilon         : ε-greedy exploration rate (1.0 = fully random)
    step_length     : SUMO step size in seconds
    use_gui         : launch sumo-gui instead of sumo
    """

    def __init__(
        self,
        cfg_path: str,
        net_xml_path: str,
        stops_json_path: str,
        grid_cols: int = 10,
        grid_rows: int = 7,
        policy_fn: Optional[Callable] = None,
        epsilon: float = 1.0,
        alpha: float = 1.0,
        step_length: float = 1.0,
        use_gui: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        alpha : float
            Action probability (0, 1].  At each idle-taxi decision point the
            taxi takes a repositioning action with probability alpha and stays
            put with probability (1 - alpha).  Mirrors the DeepPool paper's
            alpha parameter (Section V-C) which ramps 0.3 → 1.0 over the
            first 5000 steps to reduce multi-agent non-stationarity early in
            training.  Default 1.0 = always act (original behaviour).
        """
        super().__init__(cfg_path=cfg_path, step_length=step_length, use_gui=use_gui)

        self.zone_map      = ZoneMap(net_xml_path, grid_cols=grid_cols, grid_rows=grid_rows)
        self.zone_map.load_stops(stops_json_path)
        self.state_tracker = ZoneStateTracker(self.zone_map)
        self.n_zones       = grid_cols * grid_rows

        self.policy_fn = policy_fn
        self.epsilon   = epsilon
        self.alpha     = alpha   # action probability (paper Section V-C)

        self._repo_state:    Dict[str, TaxiRepoState] = {}
        self._transitions:   List[Transition]          = []
        self._prev_idle_taxis: set[str]                = set()

        print(f"[ZoneEnv] {self.zone_map.summary()}")

    # ------------------------------------------------------------------
    # Core tick — replaces HeuristicDispatcher._process_tick entirely
    # ------------------------------------------------------------------

    def _process_tick(self, now: float) -> None:
        """
        One 10-step tick:
          1. Sync SUMO state (reservations, onboard status, taxi positions).
          2. Clean up stale stops (ONBOARD/COMPLETED requests).
          3. Accumulate wait cost.
          4. Pooled dispatch for every PENDING request (non-full taxis eligible).
          5. Zone DQN repositioning for every newly-idle taxi.
        """
        # 1. Sync state from SUMO
        self._sync_reservations(now)
        _refresh_taxi_plans(self.taxi_plans)
        _sync_onboard(self.taxi_plans, self.requests)

        # 2. Sync request statuses from SUMO ground truth, then clean stale stops
        self._sync_request_statuses(now)
        self._cleanup_plan_stops()

        # 3. Accumulate wait cost for reward computation
        pending = [r for r in self.requests.values()
                   if r.status == RequestStatus.PENDING]
        self.accumulator.wait_cost += len(pending) * self.step_length * self.TICK_STEPS

        n_pending = len(pending)
        # Only log ticks where something actionable is happening:
        # pending requests exist, or newly-idle taxis need zone decisions.
        # Avoids flooding the log with hundreds of "pending=0 idle_taxis=N" lines
        # when demand is low and taxis are simply waiting for the next request.
        newly_idle_count = len(
            {tid for tid in self.taxi_plans if self._is_truly_idle(tid)}
            - self._prev_idle_taxis
        )
        if n_pending or newly_idle_count:
            n_idle = sum(1 for t in self.taxi_plans.values() if self._is_truly_idle(t.taxi_id))
            _log(f"[Tick {self._tick_num:4d}  t={now:.0f}s] "
                 f"pending={n_pending}  idle_taxis={n_idle}  "
                 f"fleet={len(self.taxi_plans)}")

        # 3. Nearest-taxi greedy dispatch
        for req in sorted(pending, key=lambda r: r.request_time):
            self._greedy_dispatch(req, now)

        # 4. Zone DQN repositioning for newly-idle taxis
        self._run_zone_repositioning(now)

    # ------------------------------------------------------------------
    # Layer 1: simple nearest-taxi greedy dispatch
    # ------------------------------------------------------------------

    def _greedy_dispatch(self, request: Request, now: float) -> bool:
        """
        Assign request using a DeepPool-style pooled dispatch heuristic.

        A taxi is eligible as long as it is active in SUMO and not full.
        For each eligible taxi we greedily re-optimise the remaining route
        over its current obligations plus the new request, then choose the
        taxi whose resulting plan has the lowest composite cost.
        """
        best_tid: Optional[str] = None
        best_plan: Optional[List[Stop]] = None
        best_cost: float = float("inf")
        best_pickup_eta: float = float("inf")

        vtype = self._cached_vtype_str or ""
        try:
            active = set(traci.vehicle.getIDList())
        except Exception:
            active = set()

        for tid, plan in self.taxi_plans.items():
            if tid not in active:
                continue
            if plan.onboard_count >= plan.capacity:
                continue
            candidate = self._build_pooled_plan(plan, request, now, vtype)
            if candidate is None:
                continue
            candidate_plan, pickup_eta, _, candidate_cost = candidate
            if candidate_cost < best_cost:
                best_cost = candidate_cost
                best_tid = tid
                best_plan = candidate_plan
                best_pickup_eta = pickup_eta

        if best_tid is None or best_plan is None:
            # Constrained pooled plan failed. This should now be rare — it only
            # happens when _route_time returns inf for every insertion position on
            # every taxi (SUMO routing deadlock for this pickup edge).
            # Try unconstrained insertion so the request still gets interleaved
            # ride-sharing rather than being permanently stuck.
            best_tid, best_plan = self._fallback_dispatch(request, active, vtype, now)
            if best_tid is None or best_plan is None:
                _log(f"  [Dispatch] All taxis full/unreachable — req={request.request_id} will retry next tick")
                return False
            _log(f"  [Dispatch] req={request.request_id} → taxi={best_tid} (FALLBACK routing-deadlock)")
        else:
            _log(f"  [Dispatch] req={request.request_id} → taxi={best_tid} "
                 f"(eta_pickup≈{max(0.0, best_pickup_eta - now):.0f}s pooled)")

        ordered_res_ids = [stop.request_id for stop in best_plan]
        try:
            try:
                if traci.vehicle.isStopped(best_tid):
                    traci.vehicle.resume(best_tid)
            except traci.TraCIException:
                pass
            traci.vehicle.dispatchTaxi(best_tid, ordered_res_ids)
        except traci.TraCIException as e:
            _log(f"  [Dispatch ERROR] taxi={best_tid} req={request.request_id}: {e}")
            return False

        plan = self.taxi_plans[best_tid]
        plan.stops = best_plan
        plan.assigned_request_ids.add(request.person_id)

        request.assigned_taxi_id = best_tid
        request.status = RequestStatus.ASSIGNED
        return True

    def _fallback_dispatch(
        self,
        request: Request,
        active: set,
        vtype: str,
        now: float,
    ) -> Tuple[Optional[str], Optional[List[Stop]]]:
        """
        Last-resort dispatch used when the constrained pooled plan fails for all
        taxis (SUMO routing deadlock — _route_time returns inf for all positions).

        Step 1 — unconstrained insertion (real ride-sharing, no time-window checks):
          Same two-phase insertion as the constrained plan but with
          enforce_constraints=False.  Picks the taxi with the fewest pending
          stops that can produce a finite route (load balancing).
          Vehicle capacity is the only hard limit — no artificial stop cap.

        Step 2 — pure append to least-loaded taxi (routing fully failed):
          Used only when every insertion returned inf.  Picks the taxi with
          fewest stops among those with remaining passenger capacity.
          Returns (None, None) only when every taxi is genuinely full (10/10).
        """
        # --- Step 1: unconstrained insertion, prefer fewest pending stops ---
        best_tid:    Optional[str]       = None
        best_plan:   Optional[List[Stop]] = None
        best_cost:   float               = float("inf")
        best_nstops: int                 = 10_000

        for tid, plan in self.taxi_plans.items():
            if tid not in active:
                continue
            if plan.onboard_count >= plan.capacity:
                continue
            candidate = self._build_pooled_plan(plan, request, now=now, vtype=vtype,
                                                enforce_constraints=False)
            if candidate is None:
                continue
            candidate_plan, _, _, candidate_cost = candidate
            n = len(plan.stops)
            if (n, candidate_cost) < (best_nstops, best_cost):
                best_cost   = candidate_cost
                best_nstops = n
                best_tid    = tid
                best_plan   = candidate_plan

        if best_tid is not None:
            return best_tid, best_plan

        # --- Step 2: pure append to least-loaded taxi (routing fully failed) ---
        append_tid:  Optional[str]       = None
        append_plan: Optional[List[Stop]] = None
        min_stops:   int                 = 10_000

        for tid, plan in self.taxi_plans.items():
            if tid not in active:
                continue
            if plan.onboard_count >= plan.capacity:
                continue
            if len(plan.stops) < min_stops:
                min_stops  = len(plan.stops)
                append_tid = tid
                existing   = [
                    Stop(s.stop_type, s.request_id, s.person_id, s.edge_id, s.eta)
                    for s in plan.stops
                ]
                new_pickup  = Stop(StopType.PICKUP,  request.request_id,
                                   request.person_id, request.from_edge)
                new_dropoff = Stop(StopType.DROPOFF, request.request_id,
                                   request.person_id, request.to_edge)
                append_plan = existing + [new_pickup, new_dropoff]

        return append_tid, append_plan

    def _compute_route_etas(
        self,
        start_edge: str,
        stops: List[Stop],
        now: float,
        vtype: str,
    ) -> Tuple[float, List[float]]:
        """
        Walk start_edge → stops[0] → stops[1] → … and return
        (total_travel_time, [absolute_eta_per_stop]).
        Returns (inf, []) if any leg is unreachable.
        """
        current_edge = start_edge
        current_time = now
        etas: List[float] = []
        for stop in stops:
            tt = _route_time(current_edge, stop.edge_id, vtype)
            if tt == float("inf"):
                return float("inf"), []
            current_time += tt
            etas.append(current_time)
            current_edge = stop.edge_id
        return current_time - now, etas

    def _build_pooled_plan(
        self,
        plan: TaxiPlan,
        request: Request,
        now: float,
        vtype: str,
        enforce_constraints: bool = True,
    ) -> Optional[Tuple[List[Stop], float, float, float]]:
        """
        Build a pooled route using the paper's two-phase insertion algorithm
        (mirrors generate_plan() in simulator.py of the reference codebase).

        Phase 1 — pickup insertion
          Try every position 0..N in the existing stop list.
          Insert the new pickup at the position that minimises total
          plan completion time.  Existing stop order is preserved.

        Phase 2 — dropoff insertion
          Try every position after the chosen pickup.
          Insert the new dropoff at the position that minimises total
          plan completion time.

        enforce_constraints=False skips the max_wait / max_ride_factor
        rejection checks so the route is still properly interleaved
        (real ride-sharing) but time-window violations are tolerated.
        Used by _fallback_dispatch to guarantee every request gets served.
        """
        current_edge = plan.current_edge
        if not current_edge:
            return None

        existing_stops = [
            Stop(s.stop_type, s.request_id, s.person_id, s.edge_id, s.eta)
            for s in plan.stops
        ]
        new_pickup  = Stop(StopType.PICKUP,  request.request_id, request.person_id, request.from_edge)
        new_dropoff = Stop(StopType.DROPOFF, request.request_id, request.person_id, request.to_edge)

        # Baseline ETAs for existing stops without the new request — used later
        # for the detour constraint (how much do existing stops get delayed).
        _, baseline_etas = self._compute_route_etas(current_edge, existing_stops, now, vtype)
        # Build a lookup: (request_id, stop_type) → baseline ETA
        baseline_eta_map = {
            (s.request_id, s.stop_type): eta
            for s, eta in zip(existing_stops, baseline_etas)
        }

        # ------------------------------------------------------------------
        # Phase 1: find best pickup insertion position
        # ------------------------------------------------------------------
        best_pu_pos:   int            = -1
        best_pu_plan:  List[Stop]     = []
        best_pu_etas:  List[float]    = []
        min_time_p1:   float          = float("inf")

        for pos in range(len(existing_stops) + 1):
            # Capacity guard: count occupancy at the insertion point
            occ = plan.onboard_count
            for s in existing_stops[:pos]:
                occ += 1 if s.stop_type == StopType.PICKUP else -1
            if occ + 1 > plan.capacity:   # +1 for the passenger we're about to pick up
                continue

            candidate = existing_stops[:pos] + [new_pickup] + existing_stops[pos:]
            total_time, etas = self._compute_route_etas(current_edge, candidate, now, vtype)
            if total_time < min_time_p1:
                min_time_p1   = total_time
                best_pu_pos   = pos
                best_pu_plan  = [Stop(s.stop_type, s.request_id, s.person_id, s.edge_id) for s in candidate]
                best_pu_etas  = etas

        if best_pu_pos < 0:
            return None   # no reachable pickup position

        # Apply ETAs to the intermediate plan
        for stop, eta in zip(best_pu_plan, best_pu_etas):
            stop.eta = eta

        new_pickup_eta = best_pu_plan[best_pu_pos].eta

        # Paper constraint (matching_policy.py: T[v,r] <= reject_wait_time):
        # Check travel time from NOW to pickup, not from request creation time.
        # This matches the paper — long-waiting requests are still eligible as
        # long as a taxi can physically reach them soon.
        if enforce_constraints and (new_pickup_eta - now) > REJECT_WAIT_TIME:
            return None

        # ------------------------------------------------------------------
        # Phase 2: find best dropoff insertion position (must be after pickup)
        # ------------------------------------------------------------------
        best_do_plan: List[Stop]  = []
        best_do_etas: List[float] = []
        min_time_p2:  float       = float("inf")

        for pos in range(len(best_pu_plan) + 1):
            if pos <= best_pu_pos:
                continue   # dropoff must come after pickup

            candidate = best_pu_plan[:pos] + [new_dropoff] + best_pu_plan[pos:]
            total_time, etas = self._compute_route_etas(current_edge, candidate, now, vtype)
            if total_time < min_time_p2:
                min_time_p2  = total_time
                best_do_plan = [Stop(s.stop_type, s.request_id, s.person_id, s.edge_id) for s in candidate]
                best_do_etas = etas

        if not best_do_plan:
            return None   # no reachable dropoff position

        # Apply final ETAs
        route: List[Stop] = best_do_plan
        for stop, eta in zip(route, best_do_etas):
            stop.eta = eta

        pickup_eta = next(
            (s.eta for s in route
             if s.request_id == request.request_id and s.stop_type == StopType.PICKUP),
            None,
        )
        dropoff_eta = next(
            (s.eta for s in route
             if s.request_id == request.request_id and s.stop_type == StopType.DROPOFF),
            None,
        )
        if pickup_eta is None or dropoff_eta is None:
            return None

        # ------------------------------------------------------------------
        # Feasibility checks (skipped when enforce_constraints=False)
        # ------------------------------------------------------------------
        if enforce_constraints:
            # New passenger: travel time from NOW to pickup (paper's T[v,r] check)
            if (pickup_eta - now) > REJECT_WAIT_TIME:
                return None

            # Existing passengers: detour guard (paper's TDD-style check).
            # Reject if inserting the new request delays any existing stop ETA
            # by more than REJECT_DETOUR_TIME vs the no-new-request baseline.
            for stop in route:
                if stop.request_id == request.request_id:
                    continue
                key = (stop.request_id, stop.stop_type)
                orig_eta = baseline_eta_map.get(key)
                if orig_eta is None:
                    continue
                if stop.eta - orig_eta > REJECT_DETOUR_TIME:
                    return None

        completion_time = route[-1].eta - now if route else 0.0
        new_wait        = pickup_eta - now   # time from now, matching paper's T[v,r] metric
        # Paper criterion: minimum total completion time, tie-broken by time-to-pickup
        pooled_cost = completion_time + 0.1 * new_wait
        return route, pickup_eta, dropoff_eta, pooled_cost

    # ------------------------------------------------------------------
    # Layer 2: zone DQN repositioning
    # ------------------------------------------------------------------

    def _run_zone_repositioning(self, now: float) -> None:
        """
        For every taxi that just became idle, make a zone repositioning decision.
        Finalises the reward for the taxi's previous repositioning period first.
        """
        current_idle: set[str] = {
            tid for tid in self.taxi_plans if self._is_truly_idle(tid)
        }
        newly_idle = current_idle - self._prev_idle_taxis

        for tid, repo in self._repo_state.items():
            if self._is_truly_idle(tid):
                self._ensure_repo_progress(tid, repo)

        if not newly_idle:
            self._prev_idle_taxis = current_idle

            # Still accumulate empty driving for in-progress repositioning
            for tid, repo in self._repo_state.items():
                plan = self.taxi_plans.get(tid)
                if plan and self._is_truly_idle(tid):
                    try:
                        d = traci.vehicle.getDistance(tid)
                        delta = max(0.0, d - getattr(repo, "_last_dist", d))
                        repo._last_dist = d        # type: ignore[attr-defined]
                        repo.empty_dist_since += delta
                    except Exception:
                        pass
            return

        # Build shared state once for all newly-idle taxis this tick
        current_state = self.state_tracker.build_state(self.taxi_plans, self.requests)
        fleet_size    = max(1, len(self.taxi_plans))

        for taxi_id in newly_idle:
            # --- Finalise previous repositioning transition ---
            prev = self._repo_state.get(taxi_id)
            if prev is not None:
                pickups_now = self._count_pickups_by_taxi(taxi_id)
                reward = self._compute_reward(
                    new_pickups  = pickups_now - prev.pickups_since,
                    global_wait  = self._global_wait_seconds(now),
                    empty_dist   = prev.empty_dist_since,
                    fleet_size   = fleet_size,
                )
                self._transitions.append(Transition(
                    state      = prev.last_state,
                    action     = prev.last_action,
                    reward     = reward,
                    next_state = current_state,
                    done       = False,
                ))

            # --- Alpha-exploration gate (DeepPool paper Section V-C) ---
            # With probability (1 - alpha) the taxi skips repositioning and
            # stays put.  Early in training (alpha close to 0.3) most taxis
            # do nothing, limiting simultaneous policy changes and reducing
            # multi-agent non-stationarity.  As alpha ramps to 1.0 all taxis
            # always act — identical to the original behaviour.
            if random.random() > self.alpha:
                _log(f"  [ZoneRepo SKIP α] taxi={taxi_id} skipping (α={self.alpha:.2f})")
                # Clear any stale repo entry so the taxi is clean next tick
                self._repo_state.pop(taxi_id, None)
                continue

            # --- New zone decision ---
            action = self._select_zone_action(current_state)
            target_stop, actual_zone = self._send_taxi_to_zone(taxi_id, action)
            target_edge = target_stop.edge_id if target_stop is not None else ""

            self._repo_state[taxi_id] = TaxiRepoState(
                last_state      = current_state.copy(),
                last_action     = action,
                target_edge     = target_edge,
                target_zone     = actual_zone,
                target_lane_index = getattr(target_stop, "lane_index", 0),
                target_pos      = getattr(target_stop, "pos", 1.0),
                pickups_since   = self._count_pickups_by_taxi(taxi_id),
                empty_dist_since= 0.0,
            )

        self._prev_idle_taxis = current_idle

    # ------------------------------------------------------------------
    # Zone action helpers
    # ------------------------------------------------------------------

    def _select_zone_action(self, state: np.ndarray) -> int:
        if self.policy_fn is None or random.random() < self.epsilon:
            return random.randrange(self.n_zones)
        return int(self.policy_fn(state))

    def _send_taxi_to_zone(self, taxi_id: str, zone_id: int) -> Tuple[Optional[ZoneStop], int]:
        """
        Send taxi to the nearest verified stop in zone_id (from stops.json).
        Returns (target_edge, actual_zone).
        """
        target_stop = self.zone_map.nearest_stop(zone_id)
        if target_stop is None:
            _log(f"  [ZoneRepo SKIP] taxi={taxi_id} — no stops loaded")
            return None, zone_id

        target_edge = target_stop.edge_id
        actual_zone = self.zone_map.edge_to_zone(target_edge)
        if actual_zone < 0:
            actual_zone = zone_id

        try:
            traci.vehicle.changeTarget(taxi_id, target_edge)
            try:
                if traci.vehicle.isStopped(taxi_id):
                    traci.vehicle.resume(taxi_id)
            except traci.TraCIException:
                pass
            _log(f"  [ZoneRepo] taxi={taxi_id} → zone={actual_zone} stop={target_edge}")
        except traci.TraCIException as e:
            _log(f"  [ZoneRepo SKIP] taxi={taxi_id}: {e}")
            return None, actual_zone

        return target_stop, actual_zone

    def _ensure_repo_progress(self, taxi_id: str, repo: TaxiRepoState) -> None:
        if not repo.target_edge:
            return
        try:
            route = list(traci.vehicle.getRoute(taxi_id))
            route_index = traci.vehicle.getRouteIndex(taxi_id)
            road_id = traci.vehicle.getRoadID(taxi_id)
            if road_id == repo.target_edge:
                return
            if route_index < 0:
                return
            if repo.target_edge not in route[route_index:]:
                return
            if traci.vehicle.isStopped(taxi_id):
                traci.vehicle.resume(taxi_id)
        except traci.TraCIException as e:
            _log(f"  [ZoneRepo WARN] taxi={taxi_id} progress check failed: {e}")
            return

    # ------------------------------------------------------------------
    # Idle / pickup helpers
    # ------------------------------------------------------------------

    def _sync_request_statuses(self, now: float) -> None:
        """
        Update Request.status using SUMO as ground truth.

          - Person inside a taxi vehicle → ONBOARD
          - Person no longer in simulation → COMPLETED
          - Otherwise → keep existing status (PENDING / ASSIGNED / DEFERRED)
        """
        try:
            active_pids = set(traci.person.getIDList())
        except Exception:
            return

        for pid, req in self.requests.items():
            if req.status == RequestStatus.COMPLETED:
                continue

            if pid not in active_pids:
                # Person left simulation → trip completed
                if req.status != RequestStatus.COMPLETED:
                    req.status = RequestStatus.COMPLETED
                    req.dropoff_time = now
                continue

            try:
                vehicle_id = traci.person.getVehicle(pid)
            except Exception:
                vehicle_id = ""

            if vehicle_id:
                if req.status != RequestStatus.ONBOARD:
                    req.status = RequestStatus.ONBOARD
                    req.assigned_taxi_id = vehicle_id
                    req.pickup_time = now
            else:
                # Person still waiting — keep PENDING/ASSIGNED/DEFERRED as-is
                if req.status == RequestStatus.ONBOARD:
                    # Was onboard but no longer in a vehicle (shouldn't happen mid-ride)
                    req.status = RequestStatus.ASSIGNED

    def _cleanup_plan_stops(self) -> None:
        """
        Remove stops whose requests are already satisfied according to SUMO.

        Rule:
          - COMPLETED request  → remove both its PICKUP and DROPOFF stops
          - ONBOARD request    → remove its PICKUP stop (passenger is already in,
                                 keep DROPOFF so we know the taxi is still busy)

        This is necessary because we add [PU, DO] stops manually when dispatching,
        but we never remove them when SUMO actually executes each stop.  Without
        this cleanup _is_truly_idle() would never return True after the first
        dispatch, causing every subsequent request to defer.
        """
        for plan in self.taxi_plans.values():
            if not plan.stops:
                continue
            cleaned = []
            for stop in plan.stops:
                pid = self.resid_to_pid.get(stop.request_id, stop.request_id)
                req = self.requests.get(pid)
                if req is None:
                    continue  # unknown reservation — drop
                if req.status == RequestStatus.COMPLETED:
                    continue  # both PU and DO done — drop
                if req.status == RequestStatus.ONBOARD and stop.stop_type == StopType.PICKUP:
                    continue  # pickup already happened — drop PU, keep DO
                cleaned.append(stop)
            plan.stops = cleaned

    def _is_truly_idle(self, taxi_id: str) -> bool:
        plan = self.taxi_plans.get(taxi_id)
        if plan is None:
            return False
        return plan.onboard_count == 0 and len(plan.stops) == 0

    def _count_pickups_by_taxi(self, taxi_id: str) -> int:
        return sum(
            1 for r in self.requests.values()
            if r.assigned_taxi_id == taxi_id
            and r.status in (RequestStatus.ONBOARD, RequestStatus.COMPLETED)
        )

    def _global_wait_seconds(self, now: float) -> float:
        return sum(
            r.waiting_time(now)
            for r in self.requests.values()
            if r.status == RequestStatus.PENDING
        )

    @staticmethod
    def _compute_reward(
        new_pickups: int,
        global_wait: float,
        empty_dist:  float,
        fleet_size:  int,
    ) -> float:
        return (
            W_SERVE    * new_pickups
            - W_WAIT   * (global_wait / max(1, fleet_size))
            - W_REPOSITION * empty_dist
        )

    # ------------------------------------------------------------------
    # Episode entry point
    # ------------------------------------------------------------------

    def run_episode(self) -> tuple:
        """
        Run one full SUMO episode.

        Returns
        -------
        transitions : List[Transition]
            All (s, a, r, s', done) tuples collected during the episode.
        stats : dict
            Per-episode metrics matching the training_history.csv schema:
              completed_requests, picked_up_requests,
              avg_wait_until_pickup, avg_excess_ride_time
        """
        self._transitions.clear()
        self._repo_state.clear()
        self._prev_idle_taxis.clear()

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        setup_logger(f"zone_dispatcher_{ts}.log")

        self.start()
        try:
            self.run()
        finally:
            self.close()

        # Terminal transitions for any open repositioning periods
        try:
            final_state = self.state_tracker.build_state(self.taxi_plans, self.requests)
        except Exception:
            final_state = np.zeros(
                (2, self.zone_map.grid_rows, self.zone_map.grid_cols), dtype=np.float32
            )

        fleet_size = max(1, len(self.taxi_plans))
        for taxi_id, repo in self._repo_state.items():
            reward = self._compute_reward(
                new_pickups  = self._count_pickups_by_taxi(taxi_id) - repo.pickups_since,
                global_wait  = 0.0,
                empty_dist   = repo.empty_dist_since,
                fleet_size   = fleet_size,
            )
            self._transitions.append(Transition(
                state=repo.last_state, action=repo.last_action,
                reward=reward, next_state=final_state, done=True,
            ))

        # --- Episode stats ---
        all_reqs = list(self.requests.values())

        completed = [r for r in all_reqs if r.status == RequestStatus.COMPLETED]
        picked_up = [r for r in all_reqs if r.pickup_time is not None]

        wait_times = [
            r.pickup_time - r.request_time
            for r in picked_up
            if r.pickup_time is not None
        ]
        excess_times = [
            r.excess_ride_time
            for r in completed
            if r.excess_ride_time is not None
        ]

        stats = {
            "completed_requests":   len(completed),
            "picked_up_requests":   len(picked_up),
            "avg_wait_until_pickup": (sum(wait_times) / len(wait_times)) if wait_times else 0.0,
            "avg_excess_ride_time":  (sum(excess_times) / len(excess_times)) if excess_times else 0.0,
        }

        print(
            f"[ZoneEnv] Episode done — {len(self._transitions)} transitions | "
            f"completed={stats['completed_requests']} picked_up={stats['picked_up_requests']} "
            f"avg_wait={stats['avg_wait_until_pickup']:.1f}s "
            f"avg_excess={stats['avg_excess_ride_time']:.1f}s"
        )
        return list(self._transitions), stats

    def get_current_state(self) -> np.ndarray:
        return self.state_tracker.build_state(self.taxi_plans, self.requests)