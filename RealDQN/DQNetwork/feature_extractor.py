from __future__ import annotations

import math

from DRTDataclass import CandidateInsertion, GlobalStateSummary, Request, TaxiPlan, TaxiStatus


def _bool(x: bool) -> int:
    return 1 if x else 0


def encode_global_state(summary: GlobalStateSummary) -> dict[str, float]:
    t = (float(summary.sim_time) % 86400) / 86400.0
    return {
        "sim_time": float(summary.sim_time),
        "pending_req_count": float(summary.pending_req_count),
        "onboard_count": float(summary.onboard_count),
        "idle_taxi_count": float(summary.idle_taxi_count),
        "active_taxi_count": float(summary.active_taxi_count),
        "avg_wait_time": float(summary.avg_wait_time),
        "max_wait_time": float(summary.max_wait_time),
        "avg_occupancy": float(summary.avg_occupancy),
        "fleet_utilization": float(summary.fleet_utilization),
        "recent_demand_rate": float(summary.recent_demand_rate),
        # Medium impact: system pressure (high value → dangerous to defer)
        "pending_to_idle_ratio": float(summary.pending_req_count) / max(1.0, float(summary.idle_taxi_count)),
        # Lower impact: time-of-day cyclical encoding (continuous across midnight)
        "time_of_day_sin": math.sin(2.0 * math.pi * t),
        "time_of_day_cos": math.cos(2.0 * math.pi * t),
    }


def encode_request_features(request: Request, now: float) -> dict[str, float]:
    waited = max(0.0, now - request.request_time)
    max_wait = float(getattr(request, "max_wait", 0.0) or 0.0)
    direct_tt = float(getattr(request, "direct_travel_time", 0.0) or 0.0)
    safe_max_wait = max(1.0, max_wait)
    return {
        "req_waited_so_far": waited,
        "req_direct_travel_time": direct_tt,
        "req_max_wait": max_wait,
        "req_wait_slack": max_wait - waited,
        "req_has_assigned_taxi": _bool(request.assigned_taxi_id is not None),
        # High impact: normalized urgency — 0.0 = just arrived, 1.0 = at deadline
        "req_wait_urgency": waited / safe_max_wait,
        # High impact: inverse of urgency, makes the boundary explicit to the NN
        "req_slack_ratio": max(0.0, max_wait - waited) / safe_max_wait,
    }


def encode_candidate_features(
    candidate: CandidateInsertion,
    request: Request,
    taxi_plans: dict[str, TaxiPlan],
    now: float,
) -> dict[str, float]:
    if candidate.is_defer:
        return {
            "cand_is_defer": 1.0,
            "cand_is_feasible": 1.0,
            "cand_pickup_index": -1.0,
            "cand_dropoff_index": -1.0,
            "cand_added_route_time": 0.0,
            "cand_pickup_eta_new": 0.0,
            "cand_dropoff_eta_new": 0.0,
            "cand_future_wait": 0.0,
            "cand_total_wait": 0.0,
            "cand_new_ride_time": 0.0,
            "cand_max_existing_delay": 0.0,
            "cand_avg_existing_delay": 0.0,
            "cand_max_pickup_delay": 0.0,
            "cand_new_wait_violation": 0.0,
            "cand_new_ride_violation": 0.0,
            "cand_exist_wait_viol_sum": 0.0,
            "cand_exist_wait_viol_max": 0.0,
            "cand_exist_ride_viol_sum": 0.0,
            "cand_exist_ride_viol_max": 0.0,
            "cand_result_stop_count": 0.0,
            "cand_taxi_remaining_route_time": 0.0,
            "cand_taxi_onboard_count": 0.0,
            "cand_taxi_remaining_capacity": 0.0,
            "cand_taxi_is_idle": 0.0,
            "cand_taxi_is_active": 0.0,
            # new features — neutral values for defer
            "cand_ride_detour_ratio": 0.0,
            "cand_taxi_load_ratio": 0.0,
            "cand_insertion_position_ratio": 0.0,
            "cand_added_route_fraction": 0.0,
        }

    plan = taxi_plans.get(candidate.taxi_id)
    future_wait = max(0.0, candidate.pickup_eta_new - now)
    total_wait = max(0.0, candidate.pickup_eta_new - request.request_time)
    new_ride_time = max(0.0, candidate.dropoff_eta_new - candidate.pickup_eta_new)

    taxi_is_idle = bool(plan and plan.status == TaxiStatus.IDLE and len(plan.stops) == 0)
    taxi_is_active = bool(plan and plan.status != TaxiStatus.IDLE)
    remaining_route_time = float(getattr(plan, "remaining_route_time", 0.0) if plan else 0.0)
    onboard_count = float(getattr(plan, "onboard_count", 0.0) if plan else 0.0)
    remaining_capacity = float(getattr(plan, "remaining_capacity", 0.0) if plan else 0.0)
    capacity = float(getattr(plan, "capacity", 1) if plan else 1) or 1.0

    direct_tt = float(getattr(request, "direct_travel_time", 0.0) or 0.0)
    result_stop_count = float(len(candidate.resulting_stops))

    return {
        "cand_is_defer": 0.0,
        "cand_is_feasible": _bool(candidate.is_feasible),
        "cand_pickup_index": float(candidate.pickup_index),
        "cand_dropoff_index": float(candidate.dropoff_index),
        "cand_added_route_time": float(candidate.added_route_time),
        "cand_pickup_eta_new": float(candidate.pickup_eta_new),
        "cand_dropoff_eta_new": float(candidate.dropoff_eta_new),
        "cand_future_wait": future_wait,
        "cand_total_wait": total_wait,
        "cand_new_ride_time": new_ride_time,
        "cand_max_existing_delay": float(candidate.max_existing_delay),
        "cand_avg_existing_delay": float(candidate.avg_existing_delay),
        "cand_max_pickup_delay": float(candidate.max_pickup_delay),
        "cand_new_wait_violation": float(candidate.new_wait_violation),
        "cand_new_ride_violation": float(candidate.new_ride_violation),
        "cand_exist_wait_viol_sum": float(candidate.existing_wait_violation_sum),
        "cand_exist_wait_viol_max": float(candidate.existing_wait_violation_max),
        "cand_exist_ride_viol_sum": float(candidate.existing_ride_violation_sum),
        "cand_exist_ride_viol_max": float(candidate.existing_ride_violation_max),
        "cand_result_stop_count": result_stop_count,
        "cand_taxi_remaining_route_time": remaining_route_time,
        "cand_taxi_onboard_count": onboard_count,
        "cand_taxi_remaining_capacity": remaining_capacity,
        "cand_taxi_is_idle": _bool(taxi_is_idle),
        "cand_taxi_is_active": _bool(taxi_is_active),
        # High impact: how much detour vs solo trip — proximity to max_ride_factor=2
        "cand_ride_detour_ratio": new_ride_time / max(1.0, direct_tt),
        # High impact: taxi fullness as a fraction
        "cand_taxi_load_ratio": onboard_count / capacity,
        # Medium impact: where in the queue is this insertion (front=0, back=1)
        "cand_insertion_position_ratio": float(candidate.pickup_index) / max(1.0, result_stop_count),
        # Medium impact: how much extra time we add relative to what's already committed
        "cand_added_route_fraction": float(candidate.added_route_time) / max(1.0, remaining_route_time),
    }


def flatten_decision_features(
    summary: GlobalStateSummary,
    request: Request,
    candidate: CandidateInsertion,
    taxi_plans: dict[str, TaxiPlan],
    now: float,
) -> dict[str, float]:
    row: dict[str, float] = {}
    row.update(encode_global_state(summary))
    row.update(encode_request_features(request, now))
    row.update(encode_candidate_features(candidate, request, taxi_plans, now))
    # Spatial demand/supply grid from ZoneStateTracker (shape 2×R×C, values in [0,1]).
    # Appended AFTER the scaler-normalised features so the existing scaler is unchanged.
    # Named "spatial_0" … "spatial_139" (for a 2×7×10 grid = 140 features).
    if summary.spatial_grid is not None:
        flat = summary.spatial_grid.flatten()
        for i, v in enumerate(flat):
            row[f"spatial_{i}"] = float(v)
    return row
