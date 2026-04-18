from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import trange

from dqn_env import DQNStepEnvironment
from dispatcher import setup_logger
from feature_extractor import flatten_decision_features
from q_network import ParametricQNetwork, TaxiFairQNetwork
from replay_buffer import (
    NStepBuffer, ReplayBuffer, PrioritizedReplayBuffer,
    PrioritizedReplayBatch, Transition,
)
from reward_shaping import compute_shaped_reward_v2, compute_shaped_reward_v3


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class EpisodeStats:
    episode: int
    total_reward: float
    normalised_reward: float   # reward / num_decisions — comparable across episodes
    steps: int
    mean_loss: float
    completed_requests: int
    picked_up_requests: int
    avg_wait_until_pickup: float
    avg_excess_ride_time: float
    epsilon: float
    lr: float
    q_mean: float
    q_max: float
    q_min: float
    target_mean: float


class DQNAgent:
    def __init__(
        self,
        feature_columns: list[str],
        scaler,
        input_dim: int,
        hidden_dims: list[int],
        dropout: float,
        device: torch.device,
        gamma: float,
        lr: float,
        lr_min: float,
        tau: float,
        forbid_defer_when_action_exists: bool = True,
    ):
        self.feature_columns = feature_columns
        self.scaler = scaler
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.forbid_defer_when_action_exists = forbid_defer_when_action_exists

        self.online_net = TaxiFairQNetwork(
            input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout
        ).to(device)
        self.target_net = TaxiFairQNetwork(
            input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout
        ).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=lr)
        self.scheduler: torch.optim.lr_scheduler.CosineAnnealingLR | None = None
        self.lr_min = lr_min

    def init_scheduler(self, total_train_steps: int) -> None:
        """Call once after warm-start, before the first gradient update."""
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(1, total_train_steps), eta_min=self.lr_min
        )

    def current_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def load_warm_start(self, state_dict: dict[str, Any]) -> None:
        missing, unexpected = self.online_net.load_state_dict(state_dict, strict=False)
        self.target_net.load_state_dict(self.online_net.state_dict())
        print(f"Warm start loaded. missing={missing} unexpected={unexpected}")

    def decision_to_matrix(self, decision_point, taxi_plans) -> np.ndarray:
        # Assign each unique taxi a local integer group id (0, 1, 2, ...).
        # Defer action always gets -1. This is used by TaxiFairQNetwork to
        # normalise scores within each taxi group before comparing across taxis.
        seen_taxis: dict[str, int] = {}
        rows: list[list[float]] = []
        group_ids: list[float] = []

        for cand in decision_point.candidate_actions:
            feat = flatten_decision_features(
                decision_point.state_summary,
                decision_point.request,
                cand,
                taxi_plans,
                decision_point.sim_time,
            )
            missing = [c for c in self.feature_columns if c not in feat]
            if missing:
                raise KeyError(f"Missing feature columns at DQN time: {missing[:10]}")
            rows.append([float(feat[c]) for c in self.feature_columns])

            if cand.is_defer:
                group_ids.append(-1.0)
            else:
                tid = cand.taxi_id
                if tid not in seen_taxis:
                    seen_taxis[tid] = len(seen_taxis)
                group_ids.append(float(seen_taxis[tid]))

        x = np.asarray(rows, dtype=np.float32)
        x = self.scaler.transform(x).astype(np.float32)
        group_col = np.array(group_ids, dtype=np.float32).reshape(-1, 1)
        mask = np.ones((x.shape[0], 1), dtype=np.float32)
        # Layout: [features | taxi_group_id | valid_mask]  — matches TaxiFairQNetwork
        return np.concatenate([x, group_col, mask], axis=1)

    def _fair_random_action(
        self,
        valid_indices: list[int],
        candidate_actions: list,
    ) -> int:
        """
        Taxi-fair random exploration.

        Problem with naive random.choice(valid_indices):
          A taxi with N existing stops generates O(N²) insertion candidates,
          while an idle taxi generates exactly 1. Uniform sampling over all
          candidates therefore picks the busiest taxi with probability
          proportional to N² — a vicious circle where long stop lists
          attract more random assignments, making them even longer.

        Fix — two-level sampling:
          1. Pick a taxi uniformly at random from all taxis that have at
             least one valid candidate.
          2. Pick uniformly from that taxi's candidates.

        This gives every eligible taxi exactly the same probability of
        being chosen during exploration, regardless of how many insertion
        positions it currently has.
        """
        # Group valid candidate indices by taxi_id
        taxi_to_indices: dict[str, list[int]] = {}
        for idx in valid_indices:
            cand = candidate_actions[idx]
            key = cand.taxi_id if not cand.is_defer else "__defer__"
            taxi_to_indices.setdefault(key, []).append(idx)

        # Pick a taxi uniformly, then a candidate within it uniformly
        chosen_taxi = random.choice(list(taxi_to_indices.keys()))
        return random.choice(taxi_to_indices[chosen_taxi])

    def select_action(
        self, state_matrix: np.ndarray, decision_point, epsilon: float
    ) -> int:
        with torch.no_grad():
            inp = torch.from_numpy(state_matrix[None, :, :]).to(self.device)
            q_vals, _ = self.online_net(inp)
            scores = q_vals[0].detach().cpu().numpy().astype(float)

        valid_indices = list(range(len(scores)))
        if self.forbid_defer_when_action_exists and any(
            not c.is_defer for c in decision_point.candidate_actions
        ):
            valid_indices = [
                i for i, c in enumerate(decision_point.candidate_actions)
                if not c.is_defer
            ]
            for i, c in enumerate(decision_point.candidate_actions):
                if c.is_defer:
                    scores[i] = -1e9

        if random.random() < epsilon:
            return self._fair_random_action(valid_indices, decision_point.candidate_actions)
        return int(np.argmax(scores))

    def train_step(
        self,
        replay: ReplayBuffer | PrioritizedReplayBuffer,
        batch_size: int,
        n_steps: int = 1,
    ) -> dict[str, float]:
        """Single gradient update. Returns dict with loss and Q-value stats.

        Supports both uniform ReplayBuffer and PrioritizedReplayBuffer.
        When using PER, importance-sampling weights correct the bias from
        non-uniform sampling, and TD errors are fed back to update priorities.

        n_steps: the N used by NStepBuffer.  batch.rewards already contains
                 the N-step discounted return G_t, so the bootstrap target uses
                 γ^N instead of γ:
                     target = G_t + γ^N · (1 − done) · max Q(s_{t+N})
        """
        batch = replay.sample(batch_size, self.device)
        q_values, _ = self.online_net(batch.states)
        q_sa = q_values.gather(1, batch.actions.unsqueeze(1)).squeeze(1)

        gamma_n = self.gamma ** n_steps  # γ^N for N-step bootstrap

        with torch.no_grad():
            # Double DQN: online net selects action, target net evaluates it
            next_q_online, _ = self.online_net(batch.next_states)
            next_actions = torch.argmax(next_q_online, dim=1, keepdim=True)
            next_q_target, _ = self.target_net(batch.next_states)
            next_q = next_q_target.gather(1, next_actions).squeeze(1)
            next_q = next_q * batch.next_state_exists
            targets = batch.rewards + gamma_n * (1.0 - batch.dones) * next_q

        td_errors = q_sa - targets  # element-wise TD error

        if isinstance(batch, PrioritizedReplayBatch):
            # Weight each sample's loss by its IS weight to correct for
            # non-uniform sampling bias
            element_loss = F.smooth_l1_loss(q_sa, targets, reduction="none")
            loss = (batch.is_weights * element_loss).mean()

            # Update priorities in the Sum Tree with new TD errors
            replay.update_priorities(
                batch.leaf_indices,
                td_errors.detach().cpu().numpy(),
            )
        else:
            loss = F.smooth_l1_loss(q_sa, targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 5.0)
        self.optimizer.step()

        # NOTE: scheduler.step() moved to per-episode in main() to avoid
        # exhausting the cosine schedule too quickly.

        self.soft_update()
        return {
            "loss": float(loss.item()),
            "q_mean": float(q_sa.mean().item()),
            "q_max": float(q_sa.max().item()),
            "q_min": float(q_sa.min().item()),
            "target_mean": float(targets.mean().item()),
        }

    def soft_update(self) -> None:
        for target_p, online_p in zip(
            self.target_net.parameters(), self.online_net.parameters()
        ):
            target_p.data.copy_(
                target_p.data * (1.0 - self.tau) + online_p.data * self.tau
            )


# ---------------------------------------------------------------------------
# Reward shaping — moved to reward_shaping.py
# The old compute_shaped_reward is kept here only as a fallback import.
# All new training should use compute_shaped_reward_v2 from reward_shaping.py.
# ---------------------------------------------------------------------------
# (Old function removed — see reward_shaping.py for both v1 and v2)


# ---------------------------------------------------------------------------
# Evaluation / summary helpers
# ---------------------------------------------------------------------------

def summarize_env(env: DQNStepEnvironment) -> dict[str, float]:
    requests = list(env.requests.values())
    completed = [r for r in requests if r.status.name == "COMPLETED"]
    picked_up = [r for r in requests if r.pickup_time is not None]
    dropped = [
        r for r in completed
        if r.dropoff_time is not None and r.pickup_time is not None
    ]
    avg_wait = (
        sum((r.pickup_time - r.request_time) for r in picked_up) / len(picked_up)
        if picked_up else 0.0
    )
    avg_excess = (
        sum((r.excess_ride_time or 0.0) for r in dropped) / len(dropped)
        if dropped else 0.0
    )
    return {
        "completed_requests": float(len(completed)),
        "picked_up_requests": float(len(picked_up)),
        "avg_wait_until_pickup": float(avg_wait),
        "avg_excess_ride_time": float(avg_excess),
    }


def evaluate_policy(
    cfg: str, step_length: float, use_gui: bool, agent: DQNAgent,
    net_xml_path: str | None = None,
    reward_fn=compute_shaped_reward_v2,
    reward_clip: float = 5.0,
) -> dict[str, float]:
    if reward_fn is None:
        reward_fn = compute_shaped_reward_v2
    env = DQNStepEnvironment(
        cfg_path=cfg, step_length=step_length, use_gui=use_gui,
        policy=None, dataset_logger=None, verbose=False,
        net_xml_path=net_xml_path,
    )
    total_reward = 0.0
    steps = 0
    try:
        decision = env.reset_episode()
        while decision is not None:
            state = agent.decision_to_matrix(decision, env.taxi_plans)
            action = agent.select_action(state, decision, epsilon=0.0)
            prev_decision = decision  # save before step overwrites it
            result = env.step_decision(action)
            shaped = reward_fn(
                env.accumulator,
                env.accumulator.elapsed_time,
                bool(result.info.get("chosen_is_defer", False)),
                chosen_candidate=prev_decision.candidate_actions[action],
                request=prev_decision.request,
                requests_dict=env.requests,
            )
            shaped = max(-reward_clip, min(reward_clip, shaped))
            total_reward += shaped
            steps += 1
            decision = None if result.done else result.next_decision
        summary = summarize_env(env)
        summary.update({"eval_total_reward": total_reward, "eval_steps": steps})
        return summary
    finally:
        env.close_episode()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="DQN training with shaped reward and LR schedule."
    )
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--net", default=None,
                        help="Path to .net.xml for spatial demand/supply grid features. "
                             "If omitted, spatial features are disabled.")
    parser.add_argument("--imitation-model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--replay-size", type=int, default=50000)
    parser.add_argument(
        "--warmup-episodes", type=int, default=2,
        help="Run this many episodes with random actions before any training."
             " Fills the replay buffer with diverse experience first."
             " Keep low (2-3) because random actions make episodes run much"
             " longer than normal — passengers wait longer to be served."
    )
    parser.add_argument(
        "--train-every", type=int, default=4,
        help="Perform one gradient update every N decisions."
             " Reduces overfitting to the most recent experience."
    )
    parser.add_argument(
        "--gamma", type=float, default=0.8,
        help="Discount factor. Kept moderate (0.8) to limit Q-value magnitude"
             " — decisions are ~15s apart, so horizon ≈ 5 decisions is enough."
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--lr-min", type=float, default=1e-5,
        help="Minimum LR for cosine annealing schedule."
    )
    parser.add_argument(
        "--tau", type=float, default=0.005,
        help="Soft target update rate. Lower = more stable target Q-values."
    )
    parser.add_argument("--epsilon-start", type=float, default=0.4)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument(
        "--reward-clip", type=float, default=5.0,
        help="Clip shaped rewards to [-clip, +clip] to stabilise Q-targets."
    )
    parser.add_argument(
        "--n-step", type=int, default=5,
        help="N-step return horizon.  Rewards are accumulated over N decisions "
             "before being stored in the replay buffer, giving better credit "
             "assignment for temporally distant outcomes.  Default: 5."
    )
    parser.add_argument(
        "--per-alpha", type=float, default=0.6,
        help="PER: prioritization exponent (0 = uniform, 1 = full priority)."
    )
    parser.add_argument(
        "--per-beta-start", type=float, default=0.4,
        help="PER: initial importance-sampling correction. Annealed to 1.0."
    )
    parser.add_argument(
        "--no-per", action="store_true",
        help="Disable Prioritized Experience Replay (use uniform sampling)."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--step-length", type=float, default=1.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument(
        "--resume-from", type=str, default=None,
        help="Path to a previous DQN output directory to resume from."
             " Loads model weights, optimizer state, and continues episode"
             " numbering from the previous training_history.csv."
             " If omitted, starts fresh from the imitation model."
    )
    parser.add_argument(
        "--wait-target", type=float, default=None,
        help="Target avg_wait_until_pickup (seconds). When set, enables v3"
             " adaptive reward shaping that penalises decisions pushing wait"
             " above this target. Must be used together with --ride-time-target."
    )
    parser.add_argument(
        "--ride-time-target", type=float, default=None,
        help="Target total ride time (seconds). The excess ride time target"
             " is computed per-request as (ride_time_target - direct_travel_time)."
             " Must be used together with --wait-target."
    )
    args = parser.parse_args()

    if (args.wait_target is None) != (args.ride_time_target is None):
        parser.error("--wait-target and --ride-time-target must be used together.")

    # Build reward function — v3 if targets provided, else v2
    if args.wait_target is not None:
        def compute_reward(accumulator, elapsed_time, chosen_is_defer,
                           chosen_candidate=None, request=None, requests_dict=None):
            return compute_shaped_reward_v3(
                accumulator, elapsed_time, chosen_is_defer,
                chosen_candidate=chosen_candidate,
                request=request, requests_dict=requests_dict,
                wait_target=args.wait_target,
                ride_time_target=args.ride_time_target,
            )
        print(f"Reward: v3 adaptive (wait_target={args.wait_target}s, ride_time_target={args.ride_time_target}s)")
    else:
        compute_reward = compute_shaped_reward_v2
        print("Reward: v2 (fixed weights)")

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fix: setup_logger so _log() calls inside dispatcher/env actually appear
    # on the console. Without this, the "drt_dispatcher" logger has no handlers
    # and all output is silently dropped.
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_logger(str(output_dir / f"dqn_train_{ts}.log"))

    imitation_dir = Path(args.imitation_model_dir)
    metadata = json.loads(
        (imitation_dir / "model_metadata.json").read_text(encoding="utf-8")
    )
    scaler = joblib.load(imitation_dir / "feature_scaler.joblib")
    feature_columns = list(metadata["feature_columns"])
    hidden_dims = list(metadata.get("hidden_dims", [256, 128]))
    dropout = float(metadata.get("dropout", 0.1))
    input_dim = int(metadata.get("input_dim", len(feature_columns)))

    # If spatial features are enabled, extend input_dim by the flattened grid size.
    # The first Linear layer of TaxiFairQNetwork will not warm-start (shape mismatch),
    # but all subsequent layers still load correctly via strict=False.
    SPATIAL_DIM = 7 * 10 * 2  # 140  (grid_rows × grid_cols × channels)
    use_spatial = bool(args.net)
    if use_spatial:
        input_dim += SPATIAL_DIM
        print(f"[Train] Spatial features enabled (+{SPATIAL_DIM} dims) → input_dim={input_dim}")

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    agent = DQNAgent(
        feature_columns=feature_columns,
        scaler=scaler,
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        device=device,
        gamma=args.gamma,
        lr=args.lr,
        lr_min=args.lr_min,
        tau=args.tau,
    )

    # ── Resume vs fresh start ──────────────────────────────
    resume_episode = 0  # 0 means start from episode 1
    resumed_history: list[dict] = []

    if args.resume_from is not None:
        resume_dir = Path(args.resume_from)
        resume_model = resume_dir / "dqn_model.pt"
        resume_optim = resume_dir / "dqn_optimizer.pt"
        resume_csv = resume_dir / "training_history.csv"

        if not resume_model.exists():
            raise FileNotFoundError(f"No dqn_model.pt in {resume_dir}")

        # Load model weights
        state = torch.load(resume_model, map_location=device)
        agent.online_net.load_state_dict(state)
        agent.target_net.load_state_dict(state)
        print(f"Resumed model weights from {resume_model}")

        # Load optimizer state if available
        if resume_optim.exists():
            agent.optimizer.load_state_dict(
                torch.load(resume_optim, map_location=device)
            )
            print(f"Resumed optimizer state from {resume_optim}")

        # Load previous history to continue episode numbering
        if resume_csv.exists():
            prev_df = pd.read_csv(resume_csv)
            resumed_history = prev_df.to_dict("records")
            resume_episode = int(prev_df["episode"].max())
            print(f"Resuming from episode {resume_episode} ({len(resumed_history)} previous entries)")

        # Skip warmup when resuming — buffer will fill from the trained policy
        args.warmup_episodes = 0
        print("Warmup skipped (resuming from trained model)")
    else:
        warm_state = torch.load(imitation_dir / "imitation_model.pt", map_location=device)
        agent.load_warm_start(warm_state)

    training_episodes = max(1, args.episodes - args.warmup_episodes)

    print(f"Device: {device}")
    print(f"Warmup: {args.warmup_episodes} episodes (random actions, no training)")
    print(f"Training: {training_episodes} episodes (starting from ep {resume_episode + 1})")
    print(f"Gradient update: every {args.train_every} decisions")
    print(f"Gamma: {args.gamma}  Tau: {args.tau}  Reward clip: {args.reward_clip}")

    use_per = not args.no_per
    if use_per:
        replay = PrioritizedReplayBuffer(
            capacity=args.replay_size,
            alpha=args.per_alpha,
            beta=args.per_beta_start,
        )
        print(f"Replay: Prioritized (alpha={args.per_alpha}, beta_start={args.per_beta_start}→1.0)")
    else:
        replay = ReplayBuffer(capacity=args.replay_size)
        print("Replay: Uniform (PER disabled)")
    history: list[dict] = resumed_history  # prepend old history when resuming
    best_eval_reward = -float("inf")
    global_decision_count = 0
    warmup_decision_counts: list[int] = []  # measured during warmup to calibrate LR schedule

    if args.warmup_episodes == 0:
        # No warmup: scheduler T_max = number of training episodes (stepped once per ep)
        agent.init_scheduler(training_episodes)
        print(f"  [SCHEDULER] no warmup — T_max={training_episodes} episodes", flush=True)

    ep_start = resume_episode + 1
    ep_end = resume_episode + args.episodes
    for episode in trange(ep_start, ep_end + 1, desc="DQN episodes"):
        is_warmup = episode <= (resume_episode + args.warmup_episodes)

        # Epsilon only decays during training episodes
        warmup_offset = resume_episode + args.warmup_episodes
        training_progress = max(
            0.0,
            (episode - warmup_offset) / max(1, training_episodes - 1),
        )
        epsilon = args.epsilon_start + (
            args.epsilon_end - args.epsilon_start
        ) * training_progress
        epsilon = float(np.clip(epsilon, args.epsilon_end, args.epsilon_start))

        # Anneal PER beta from beta_start → 1.0 over training episodes
        if use_per and not is_warmup:
            replay.beta = args.per_beta_start + (1.0 - args.per_beta_start) * training_progress

        env = DQNStepEnvironment(
            cfg_path=args.cfg,
            step_length=args.step_length,
            use_gui=args.gui,
            policy=None,
            dataset_logger=None,
            verbose=False,
            net_xml_path=args.net,
        )
        total_reward = 0.0
        losses: list[float] = []
        q_means: list[float] = []
        q_maxs: list[float] = []
        q_mins: list[float] = []
        target_means: list[float] = []
        steps = 0
        n_step_buf = NStepBuffer(n_steps=args.n_step, gamma=args.gamma)

        try:
            if is_warmup:
                print(f"  [WARMUP {episode}/{args.warmup_episodes}] Starting episode...", flush=True)
            decision = env.reset_episode()
            while decision is not None:
                state = agent.decision_to_matrix(decision, env.taxi_plans)

                # During warmup, use moderate epsilon so the buffer gets
                # mostly good imitation-policy experience with some diversity.
                # 0.8 was far too high — it filled the buffer with terrible
                # experience that the agent then had to unlearn.
                act_epsilon = 0.3 if is_warmup else epsilon
                action_idx = agent.select_action(state, decision, epsilon=act_epsilon)

                # Save decision before step_decision overwrites env.current_decision
                prev_decision = decision

                result = env.step_decision(action_idx)
                global_decision_count += 1

                if is_warmup and steps > 0 and steps % 50 == 0:
                    print(
                        f"  [WARMUP {episode}/{args.warmup_episodes}]"
                        f"  decisions={steps}  buffer={len(replay)}",
                        flush=True,
                    )

                shaped_r = compute_reward(
                    env.accumulator,
                    env.accumulator.elapsed_time,
                    bool(result.info.get("chosen_is_defer", False)),
                    chosen_candidate=prev_decision.candidate_actions[action_idx],
                    request=prev_decision.request,
                    requests_dict=env.requests,
                )
                # Clip reward to stabilise Q-value targets
                shaped_r = float(np.clip(shaped_r, -args.reward_clip, args.reward_clip))

                next_state = (
                    None
                    if result.done or result.next_decision is None
                    else agent.decision_to_matrix(result.next_decision, env.taxi_plans)
                )

                # Push 1-step transition into N-step buffer; it emits completed
                # N-step transitions (with discounted return G_t already baked in)
                # into the replay buffer as they become ready.
                n_step_buf.push(Transition(
                    state=state,
                    action_index=action_idx,
                    reward=shaped_r,
                    next_state=next_state,
                    done=bool(result.done),
                ))
                for nst in n_step_buf.drain_ready():
                    replay.add(nst)

                total_reward += shaped_r
                steps += 1

                # Gradient update: only after warmup, only every N decisions,
                # only when buffer has enough samples
                if (
                    not is_warmup
                    and len(replay) >= args.batch_size
                    and global_decision_count % args.train_every == 0
                ):
                    train_info = agent.train_step(replay, args.batch_size, n_steps=args.n_step)
                    losses.append(train_info["loss"])
                    q_means.append(train_info["q_mean"])
                    q_maxs.append(train_info["q_max"])
                    q_mins.append(train_info["q_min"])
                    target_means.append(train_info["target_mean"])

                decision = None if result.done else result.next_decision

            summary = summarize_env(env)
            normalised_r = total_reward / max(1, steps)

            stats = EpisodeStats(
                episode=episode,
                total_reward=float(total_reward),
                normalised_reward=float(normalised_r),
                steps=steps,
                mean_loss=float(np.mean(losses)) if losses else float("nan"),
                completed_requests=int(summary["completed_requests"]),
                picked_up_requests=int(summary["picked_up_requests"]),
                avg_wait_until_pickup=float(summary["avg_wait_until_pickup"]),
                avg_excess_ride_time=float(summary["avg_excess_ride_time"]),
                epsilon=float(epsilon),
                lr=agent.current_lr(),
                q_mean=float(np.mean(q_means)) if q_means else float("nan"),
                q_max=float(np.mean(q_maxs)) if q_maxs else float("nan"),
                q_min=float(np.mean(q_mins)) if q_mins else float("nan"),
                target_mean=float(np.mean(target_means)) if target_means else float("nan"),
            )
            history.append(stats.__dict__)

            if is_warmup:
                warmup_decision_counts.append(steps)
                print(
                    f"  [WARMUP {episode}/{args.warmup_episodes}] DONE"
                    f"  buffer={len(replay)}"
                    f"  decisions={steps}"
                    f"  avg_wait={summary['avg_wait_until_pickup']:.0f}s"
                    f"  completed={summary['completed_requests']:.0f}",
                    flush=True,
                )
                if episode == (resume_episode + args.warmup_episodes):
                    # Scheduler T_max = number of training episodes (stepped once per ep)
                    agent.init_scheduler(training_episodes)
                    print(
                        f"  [SCHEDULER] LR schedule: {args.lr} -> {args.lr_min}"
                        f" over {training_episodes} episodes (per-episode stepping)",
                        flush=True,
                    )

        finally:
            env.close_episode()

        # Step LR scheduler once per training episode (not per gradient step)
        if not is_warmup and agent.scheduler is not None:
            agent.scheduler.step()

        # Evaluation
        if not is_warmup and (
            episode % args.eval_every == 0 or episode == ep_end
        ):
            eval_summary = evaluate_policy(args.cfg, args.step_length, False, agent, net_xml_path=args.net, reward_fn=compute_reward, reward_clip=args.reward_clip)
            row = history[-1]
            for k, v in eval_summary.items():
                row[f"eval_{k}"] = v
            eval_r = eval_summary["eval_total_reward"]
            print(
                f"\n  [EVAL ep={episode}]  reward={eval_r:.3f}"
                f"  wait={eval_summary.get('avg_wait_until_pickup', 0):.1f}s"
                f"  completed={eval_summary.get('completed_requests', 0):.0f}"
                f"  lr={agent.current_lr():.2e}"
            )
            if eval_r > best_eval_reward:
                best_eval_reward = eval_r
                torch.save(
                    agent.online_net.state_dict(), output_dir / "dqn_model.pt"
                )
                torch.save(
                    agent.optimizer.state_dict(), output_dir / "dqn_optimizer.pt"
                )
                joblib.dump(scaler, output_dir / "feature_scaler.joblib")
                print(f"  → New best model saved (eval reward={best_eval_reward:.3f})")

        pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)

    # Fallback save if no eval checkpoint was ever written
    if not (output_dir / "dqn_model.pt").exists():
        torch.save(agent.online_net.state_dict(), output_dir / "dqn_model.pt")

    # Always save latest optimizer state for resuming
    torch.save(agent.optimizer.state_dict(), output_dir / "dqn_optimizer.pt")
    joblib.dump(scaler, output_dir / "feature_scaler.joblib")
    dqn_metadata = {
        "feature_columns": feature_columns,
        "hidden_dims": hidden_dims,
        "dropout": dropout,
        "input_dim": input_dim,
        "use_taxi_fair": True,  # signals DQNPolicy to load TaxiFairQNetwork
        "warm_start_from": str(imitation_dir),
        "episodes": args.episodes,
        "warmup_episodes": args.warmup_episodes,
        "train_every": args.train_every,
        "gamma": args.gamma,
        "lr": args.lr,
        "lr_min": args.lr_min,
        "tau": args.tau,
        "epsilon_start": args.epsilon_start,
        "epsilon_end": args.epsilon_end,
        "reward_clip": args.reward_clip,
        "per_enabled": use_per,
        "per_alpha": args.per_alpha if use_per else None,
        "per_beta_start": args.per_beta_start if use_per else None,
    }
    (output_dir / "dqn_metadata.json").write_text(
        json.dumps(dqn_metadata, indent=2), encoding="utf-8"
    )
    print(f"\nSaved DQN artifacts to {output_dir}")
    print(f"Best eval reward: {best_eval_reward:.3f}")


if __name__ == "__main__":
    main()
