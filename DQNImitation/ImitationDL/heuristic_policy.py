from __future__ import annotations

from abc import ABC, abstractmethod

from DRTDataclass import CandidateInsertion, Request
from drt_policy_types import CandidateEvaluation, DecisionPoint, PolicyOutput


class BasePolicy(ABC):
    name = "base_policy"

    @abstractmethod
    def select_action(self, decision_point: DecisionPoint, taxi_plans, now: float) -> PolicyOutput:
        raise NotImplementedError


def score_candidate_v2(candidate: CandidateInsertion, request: Request) -> float:
    """
    Heuristic score aligned with DQN reward v2 (compute_shaped_reward_v2).

    Only per-candidate terms are kept — system_pressure and completion_bonus
    are constant across candidates at a single decision point and drop out
    of the argmax. Returned score = -(total penalty), higher is better.
    """
    if candidate.is_defer:
        return -1e9

    # 1. Per-passenger wait penalty (dominant)
    predicted_wait = max(0.0, candidate.pickup_eta_new - request.request_time)
    wait_penalty = 1.2 * (predicted_wait / 300.0)

    # 2. Per-passenger excess ride penalty
    excess_ride_penalty = 0.0
    if request.direct_travel_time > 0:
        predicted_ride = candidate.dropoff_eta_new - candidate.pickup_eta_new
        excess_ride = max(0.0, predicted_ride - request.direct_travel_time)
        excess_ride_penalty = 2.5 * (excess_ride / 200.0)

    # 3. Existing-passenger delay (worst-case + aggregate violations)
    existing_delay_penalty = (
        1.0 * (candidate.max_existing_delay / 300.0)
        + 0.5 * (getattr(candidate, "existing_ride_violation_sum", 0.0) / 200.0)
    )

    # 4. Overload guard rail (taxi with >4 stops)
    overload_penalty = 0.0
    n_stops = len(getattr(candidate, "resulting_stops", []))
    if n_stops > 4:
        overload_penalty = 0.1 * (n_stops - 4) ** 1.5

    total_penalty = (
        wait_penalty
        + excess_ride_penalty
        + existing_delay_penalty
        + overload_penalty
    )
    return -total_penalty


class HeuristicPolicy(BasePolicy):
    name = "heuristic"

    def __init__(self, print_top_k: bool = True):
        self.print_top_k = print_top_k

    def select_action(self, decision_point: DecisionPoint, taxi_plans, now: float) -> PolicyOutput:
        evaluations: list[CandidateEvaluation] = []
        for cand in decision_point.candidate_actions:
            score = score_candidate_v2(cand, decision_point.request)
            evaluations.append(
                CandidateEvaluation(candidate=cand, score=score, policy_name=self.name)
            )

        if not evaluations:
            raise ValueError("HeuristicPolicy received a decision point with no candidates.")

        order = sorted(range(len(evaluations)), key=lambda i: evaluations[i].score, reverse=True)
        for rank, idx in enumerate(order, start=1):
            evaluations[idx].rank = rank

        best_idx = order[0]
        evaluations[best_idx].chosen = True

        return PolicyOutput(
            chosen_action=evaluations[best_idx].candidate,
            evaluations=evaluations,
            policy_name=self.name,
        )
