"""
Reward shaping for DQN training.

PRIMARY objective: minimise avg_wait_until_pickup and avg_excess_ride_time.
All other metrics are secondary.

v3 — Per-passenger reward (replaces v2):
    The reward is now directly based on the predicted per-passenger wait
    and excess ride time for the chosen action.  This aligns the reward
    signal with the evaluation metrics so "higher reward" always means
    "lower avg_wait and avg_excess_ride".

    Signal budget (typical magnitude at 300s excess wait, 200s excess ride):
        wait_penalty       ≈ 1.5   (dominant — per-passenger predicted wait)
        ride_penalty       ≈ 1.0   (strong — per-passenger excess ride)
        system_pressure    ≈ 0.3-1 (prevents greedy "private taxi" strategy)
        overload           ≈ 0-1   (guard rail for taxi stop count)
        defer/existing     ≈ 0-0.3 (minor)
        completion         ≈ 0.3   (small positive offset)
"""
from __future__ import annotations

import math


def compute_shaped_reward_v2(
    accumulator,
    elapsed_time: float,
    chosen_is_defer: bool,
    chosen_candidate=None,
    request=None,
    requests_dict=None,  # kept for API compatibility
) -> float:
    """
    Per-passenger reward focused on minimising avg_wait_until_pickup and
    avg_excess_ride_time as the PRIMARY objectives.

    Every term is derived from the action the agent just took (per-passenger),
    not from system-wide accumulator rates.  This ensures that "higher reward"
    directly corresponds to "lower wait + lower excess ride".
    """

    # ────────────────────────────────────────────────────────
    # 1. PER-PASSENGER WAIT PENALTY — PRIMARY dominant signal
    #    Directly penalises the predicted wait for the passenger
    #    affected by this decision.  Normalised by 300s so the
    #    penalty ≈ 1.0 when the passenger waits 300s.
    # ────────────────────────────────────────────────────────
    wait_penalty = 0.0
    excess_ride_penalty = 0.0

    if chosen_candidate is not None and not getattr(chosen_candidate, 'is_defer', True):
        if request is not None:
            predicted_wait = max(0.0, chosen_candidate.pickup_eta_new - request.request_time)
            # Reduced from 1.1 → 0.8: wait is already ~50% better than
            # benchmark (~180s vs 365s), so free reward budget for
            # excess ride improvement.
            wait_penalty = 1.2 * (predicted_wait / 300.0)

        # ────────────────────────────────────────────────────
        # 2. PER-PASSENGER EXCESS RIDE PENALTY — PRIMARY signal
        #    Penalises only the detour beyond the direct travel time.
        #    Increased from 1.0 → 2.0: excess ride is 20% worse than
        #    benchmark so this now has 3x the per-second weight of wait,
        #    pushing the agent to prefer low-detour insertions.
        # ────────────────────────────────────────────────────
        if request is not None and request.direct_travel_time > 0:
            predicted_ride = chosen_candidate.dropoff_eta_new - chosen_candidate.pickup_eta_new
            excess_ride = max(0.0, predicted_ride - request.direct_travel_time)
            excess_ride_penalty = 2.5 * (excess_ride / 200.0)

    # ────────────────────────────────────────────────────────
    # 3. EXISTING PASSENGER DELAY (critical for excess ride)
    #    Penalises delay imposed on passengers already in the
    #    taxi.  This is the key lever: without a strong weight
    #    here the agent inserts into busy taxis (low wait for
    #    the new pax) but inflates ride time for everyone else.
    #    Also penalise the SUM of existing ride violations —
    #    this captures aggregate harm, not just worst-case.
    # ────────────────────────────────────────────────────────
    existing_delay_penalty = 0.0
    if chosen_candidate is not None and not getattr(chosen_candidate, 'is_defer', True):
        existing_delay_penalty = (
            1 * (chosen_candidate.max_existing_delay / 300.0)
            + 0.5 * (getattr(chosen_candidate, 'existing_ride_violation_sum', 0.0) / 200.0)
        )

    # ────────────────────────────────────────────────────────
    # 4. TAXI OVERLOAD PENALTY (guard rail)
    #    Superlinear penalty when taxi has >3 stops — prevents
    #    piling that causes the high excess ride times.
    # ────────────────────────────────────────────────────────
    overload_penalty = 0.0
    if chosen_candidate is not None and not getattr(chosen_candidate, 'is_defer', True):
        n_stops = len(getattr(chosen_candidate, 'resulting_stops', []))
        if n_stops > 4:
            overload_penalty = 0.1 * (n_stops - 4) ** 1.5

    # ────────────────────────────────────────────────────────
    # 5. SYSTEM PRESSURE PENALTY (prevents greedy "private taxi")
    #    Without this, the agent learns to always pick the nearest
    #    idle taxi for a direct ride (zero detour), ignoring the
    #    fact that 20 other passengers are still waiting.
    #    This small term penalises decisions made while many
    #    passengers are still pending — it makes the agent prefer
    #    ride-sharing (accept some detour) over hogging an idle taxi.
    #    Scaled so it's noticeable but never dominates the primary
    #    per-passenger wait/ride signals.
    # ────────────────────────────────────────────────────────
    t = max(elapsed_time, 1.0)
    pending_rate = accumulator.wait_cost / t  # pending-passengers × seconds
    system_pressure_penalty = 0.3 * math.sqrt(max(0.0, pending_rate))

    # ────────────────────────────────────────────────────────
    # 6. DEFER PENALTY (minor)
    # ────────────────────────────────────────────────────────
    defer_penalty = 0.3 if chosen_is_defer else 0.0

    # ────────────────────────────────────────────────────────
    # 7. COMPLETION BONUS (small positive offset)
    #    Keeps reward from being purely negative.  Small so it
    #    cannot outweigh the wait/ride penalties.
    # ────────────────────────────────────────────────────────
    quality_bonus = 0.3 * accumulator.completed_dropoffs

    reward = (
        quality_bonus
        - wait_penalty
        - excess_ride_penalty
        - existing_delay_penalty
        - overload_penalty
        - system_pressure_penalty
        - defer_penalty
    )
    return reward


def compute_shaped_reward_v3(
    accumulator,
    elapsed_time: float,
    chosen_is_defer: bool,
    chosen_candidate=None,
    request=None,
    requests_dict=None,
    *,
    wait_target: float = 170.0,
    ride_time_target: float = 500.0,
) -> float:
    """
    v3 — Target-driven reward shaping.

    Provide two targets via CLI:
        wait_target      : desired avg_wait_until_pickup (seconds)
        ride_time_target : desired total ride time (seconds)
                           The per-request excess target is computed as
                           ride_time_target - direct_travel_time.

    Penalty design:
        - At target: penalty ≈ 1.0 (moderate push to stay below)
        - Above target: penalty grows quadratically (strong push to improve)
        - Below target: penalty is linear and gentle (don't over-optimise)

    Both metrics use the SAME penalty structure, so whichever one is
    further above target automatically gets the stronger gradient —
    no manual weight balancing needed.
    """
    wait_penalty = 0.0
    excess_ride_penalty = 0.0

    is_real_action = (
        chosen_candidate is not None
        and not getattr(chosen_candidate, 'is_defer', True)
    )

    if is_real_action and request is not None:
        predicted_wait = max(
            0.0, chosen_candidate.pickup_eta_new - request.request_time
        )

        # Normalise by target: ratio = 1.0 means exactly at target
        wait_ratio = predicted_wait / max(1.0, wait_target)
        if wait_ratio > 1.0:
            # Above target: quadratic — urgently needs improvement
            wait_penalty = 1.0 + 2.0 * (wait_ratio - 1.0) ** 2
        else:
            # Below target: gentle linear — don't waste budget
            wait_penalty = 0.5 * wait_ratio

        if request.direct_travel_time > 0:
            predicted_ride = (
                chosen_candidate.dropoff_eta_new
                - chosen_candidate.pickup_eta_new
            )
            # Per-request excess target: how much detour is acceptable
            # for THIS request given its direct travel time.
            excess_target = max(1.0, ride_time_target - request.direct_travel_time)
            excess_ride = max(
                0.0, predicted_ride - request.direct_travel_time
            )
            excess_ratio = excess_ride / excess_target
            if excess_ratio > 1.0:
                # Above target: quadratic
                excess_ride_penalty = 1.0 + 2.0 * (excess_ratio - 1.0) ** 2
            else:
                # Below target: gentle linear
                excess_ride_penalty = 0.5 * excess_ratio

    # ── Existing passenger delay ──
    # Inserting a new passenger can push existing passengers' ride times
    # above the target.  Penalise using the same quadratic-above /
    # linear-below shape so the agent treats existing passengers' detour
    # with the same urgency as the new passenger's.
    # Use ride_time_target as the delay threshold (conservative: we don't
    # know each existing pax's dtt here, so use the full target).
    existing_delay_penalty = 0.0
    if is_real_action:
        max_delay = chosen_candidate.max_existing_delay
        delay_ratio = max_delay / max(1.0, ride_time_target)
        if delay_ratio > 1.0:
            existing_delay_penalty = 1.0 + 2.0 * (delay_ratio - 1.0) ** 2
        else:
            existing_delay_penalty = 0.5 * delay_ratio
        # Also penalise aggregate ride violations across all existing pax
        existing_delay_penalty += 0.5 * (
            getattr(chosen_candidate, 'existing_ride_violation_sum', 0.0)
            / max(1.0, ride_time_target)
        )

    # ── Overload penalty ──
    overload_penalty = 0.0
    if is_real_action:
        n_stops = len(getattr(chosen_candidate, 'resulting_stops', []))
        if n_stops > 3:
            overload_penalty = 0.1 * (n_stops - 3) ** 1.5

    # ── System pressure ──
    t = max(elapsed_time, 1.0)
    pending_rate = accumulator.wait_cost / t
    system_pressure_penalty = 0.3 * math.sqrt(max(0.0, pending_rate))

    # ── Defer penalty ──
    defer_penalty = 0.3 if chosen_is_defer else 0.0

    # ── Completion bonus ──
    quality_bonus = 0.3 * accumulator.completed_dropoffs

    reward = (
        quality_bonus
        - wait_penalty
        - excess_ride_penalty
        - existing_delay_penalty
        - overload_penalty
        - system_pressure_penalty
        - defer_penalty
    )
    return reward


def compute_shaped_reward(
    accumulator,
    elapsed_time: float,
    chosen_is_defer: bool,
) -> float:
    """
    ORIGINAL reward function — kept for backward compatibility.
    Use compute_shaped_reward_v3 for new training runs.
    """
    t = max(elapsed_time, 1.0)

    raw_wait = accumulator.wait_cost / t
    wait_penalty = 0.3 * min(raw_wait, 10.0)

    raw_ride = accumulator.ride_cost / t
    ride_penalty = 0.2 * min(raw_ride, 10.0)

    raw_empty = accumulator.empty_dist_cost / t
    empty_penalty = 0.0005 * min(raw_empty, 100.0)

    defer_penalty = 0.3 if chosen_is_defer else 0.0

    completion_bonus = 2.0 * accumulator.completed_dropoffs

    return completion_bonus - wait_penalty - ride_penalty - empty_penalty - defer_penalty
