"""
train_zone_dqn.py
-----------------
Training script for the zone-based repositioning DQN.

What this script does
---------------------
1. Builds a ZoneCNNQNetwork (online) and a copy (target network).
2. Runs SUMO episodes via ZoneBasedDRTEnv.
3. Pushes collected transitions into a ZoneReplayBuffer.
4. Every TRAIN_EVERY episodes, samples a mini-batch and updates the
   online network using the Double DQN Bellman target:
       a* = argmax_a  Q_online(s', a)          (online net selects)
       y  = r + γ · Q_target(s', a*)           (target net evaluates)
   This decouples action selection from evaluation, reducing Q-value
   overestimation (matching DeepPool paper Algorithm 1 lines 15-17).
5. Hard-copies the online network to the target every TARGET_UPDATE_EVERY
   training steps: θ_target ← θ_online  (paper: "Set θ = θ⁻ every N steps").
6. Linearly decays ε (exploration probability) over EPSILON_DECAY_EPISODES.
7. Linearly ramps α (action probability) from ALPHA_START to ALPHA_END over
   ALPHA_DECAY_EP episodes.  At each idle-taxi decision point the taxi acts
   with probability α and stays put with probability (1-α).  This mirrors the
   paper's α parameter that ramps 0.3→1.0 over the first 5000 steps to reduce
   multi-agent non-stationarity early in training.
8. Saves checkpoints and a training log.

Usage
-----
    python train_zone_dqn.py \\
        --cfg  ../SunwaySmallMap/osm.sumocfg \\
        --net  ../SunwaySmallMap/osm.net.xml  \\
        --episodes 200 \\
        --batch-size 64 \\
        --checkpoint-dir checkpoints_zone

Hyperparameters (configurable via CLI)
---------------------------------------
    --episodes              : total training episodes          (default 200)
    --batch-size            : replay mini-batch size           (default 64)
    --buffer-capacity       : replay buffer capacity           (default 20000)
    --gamma                 : discount factor                  (default 0.95)
    --lr                    : learning rate                    (default 1e-3)
    --target-update-every   : hard copy online→target every N  (default 10)
    --epsilon-start         : initial exploration rate         (default 1.0)
    --epsilon-end           : final exploration rate           (default 0.05)
    --epsilon-decay-ep      : episodes over which ε decays     (default 150)
    --alpha-start           : initial action probability       (default 0.3)
    --alpha-end             : final action probability         (default 1.0)
    --alpha-decay-ep        : episodes over which α ramps up   (default 50)
    --train-every           : update network every N episodes  (default 1)
    --grid-cols             : zone grid columns                (default 10)
    --grid-rows             : zone grid rows                   (default 7)
    --use-dueling           : use DuelingZoneCNNQNetwork       (flag)
    --gui                   : launch SUMO-GUI                  (flag)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(__file__)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from zone_qnetwork import ZoneCNNQNetwork, DuelingZoneCNNQNetwork
from zone_replay_buffer import ZoneReplayBuffer
from zone_env import ZoneBasedDRTEnv


# ---------------------------------------------------------------------------
# Double DQN update
# ---------------------------------------------------------------------------

def dqn_update(
    online_net: nn.Module,
    target_net: nn.Module,
    optimizer: optim.Optimizer,
    replay_buffer: ZoneReplayBuffer,
    batch_size: int,
    gamma: float,
    device: torch.device,
) -> float:
    """
    Sample one mini-batch and perform one Double DQN gradient step.

    Double DQN target (vs standard DQN):
        Standard : y = r + γ · max_a Q_target(s', a)
        Double   : a* = argmax_a Q_online(s', a)        ← online picks
                   y  = r + γ · Q_target(s', a*)        ← target evaluates

    Decoupling selection from evaluation reduces overestimation bias,
    matching DeepPool paper Algorithm 1 lines 15-17.

    Returns the loss value (float) for logging.
    """
    states, actions, rewards, next_states, dones = replay_buffer.sample_arrays(batch_size)

    # Move to device
    s  = torch.from_numpy(states).to(device)            # (B, 2, R, C)
    a  = torch.from_numpy(actions).long().to(device)    # (B,)
    r  = torch.from_numpy(rewards).to(device)           # (B,)
    ns = torch.from_numpy(next_states).to(device)       # (B, 2, R, C)
    d  = torch.from_numpy(dones).to(device)             # (B,)

    # Current Q-values: Q_online(s, a)
    q_all  = online_net(s)                                           # (B, n_zones)
    q_vals = q_all.gather(1, a.unsqueeze(1)).squeeze(1)              # (B,)

    # Double DQN target
    with torch.no_grad():
        # Online net selects the best action for next state
        best_actions = online_net(ns).argmax(dim=1, keepdim=True)    # (B, 1)
        # Target net evaluates that action (decoupled from selection)
        q_next = target_net(ns).gather(1, best_actions).squeeze(1)   # (B,)
        y = r + gamma * q_next * (1.0 - d)

    loss = nn.functional.smooth_l1_loss(q_vals, y)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(online_net.parameters(), max_norm=10.0)
    optimizer.step()

    return loss.item()


def hard_update(online_net: nn.Module, target_net: nn.Module) -> None:
    """
    θ_target ← θ_online  (full copy, no blending).

    Matches DeepPool paper Algorithm 1 line 18:
        'Set θ = θ⁻ every N steps.'
    """
    target_net.load_state_dict(online_net.state_dict())


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] device={device}")

    # --- Build networks ---
    NetworkClass = DuelingZoneCNNQNetwork if args.use_dueling else ZoneCNNQNetwork
    online_net = NetworkClass(grid_rows=args.grid_rows, grid_cols=args.grid_cols).to(device)
    target_net = NetworkClass(grid_rows=args.grid_rows, grid_cols=args.grid_cols).to(device)
    target_net.load_state_dict(online_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(online_net.parameters(), lr=args.lr)
    replay_buffer = ZoneReplayBuffer(capacity=args.buffer_capacity)

    # --- Checkpoint directory ---
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # --- Training log ---
    log: list[dict] = []
    best_episode_reward = float("-inf")
    train_step_count = 0   # counts gradient steps; drives hard target update

    # --- CSV log setup (matches training_history.csv schema + alpha column) ---
    csv_path = ckpt_dir / "training_history.csv"
    _CSV_FIELDS = [
        "episode", "total_reward", "normalised_reward", "steps",
        "mean_loss",
        "completed_requests", "picked_up_requests",
        "avg_wait_until_pickup", "avg_excess_ride_time",
        "epsilon", "alpha", "lr",
        "eval_completed_requests", "eval_picked_up_requests",
        "eval_avg_wait_until_pickup", "eval_avg_excess_ride_time",
        "eval_eval_total_reward", "eval_eval_steps",
    ]
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=_CSV_FIELDS)
    csv_writer.writeheader()

    # --- Epsilon schedule ---
    eps_start = args.epsilon_start
    eps_end   = args.epsilon_end
    eps_decay = args.epsilon_decay_ep
    epsilon   = eps_start

    # --- Alpha schedule (paper: ramps 0.3 → 1.0 over first N episodes) ---
    alp_start = args.alpha_start
    alp_end   = args.alpha_end
    alp_decay = args.alpha_decay_ep
    alpha     = alp_start

    # ---------------------------------------------------------------------------
    # Policy function (wraps online net for inference)
    # ---------------------------------------------------------------------------
    def policy_fn(state: np.ndarray) -> int:
        """Map (2, R, C) state to zone_id using the online network."""
        online_net.eval()
        with torch.no_grad():
            s = torch.from_numpy(state).unsqueeze(0).to(device)  # (1, 2, R, C)
            q = online_net(s)                                      # (1, n_zones)
        online_net.train()
        return int(q.argmax(dim=1).item())

    # ---------------------------------------------------------------------------
    # Episode loop
    # ---------------------------------------------------------------------------
    for ep in range(1, args.episodes + 1):
        episode_epsilon = epsilon
        episode_alpha   = alpha

        print(f"\n{'='*60}")
        print(f"[Train] Episode {ep}/{args.episodes}  ε={epsilon:.3f}  α={alpha:.3f}")
        print(f"{'='*60}")

        # Build environment for this episode — pass current alpha so idle taxis
        # skip repositioning with probability (1 - alpha), reducing early-training
        # non-stationarity (mirrors DeepPool paper Section V-C).
        env = ZoneBasedDRTEnv(
            cfg_path=args.cfg,
            net_xml_path=args.net,
            stops_json_path=args.stops,
            grid_cols=args.grid_cols,
            grid_rows=args.grid_rows,
            policy_fn=policy_fn,
            epsilon=episode_epsilon,
            alpha=episode_alpha,
            step_length=1.0,
            use_gui=args.gui,
        )

        # Run simulation — collect transitions + episode stats
        transitions, ep_stats = env.run_episode()

        # Push all transitions into replay buffer
        for t in transitions:
            replay_buffer.push(t.state, t.action, t.reward, t.next_state, t.done)

        episode_reward = sum(t.reward for t in transitions)
        n_trans = len(transitions)

        print(f"[Train] Episode {ep}: {n_trans} transitions, "
              f"total_reward={episode_reward:.2f}, buffer={len(replay_buffer)}")

        # --- Training step (Double DQN update + hard target copy) ---
        loss_val = None
        if ep % args.train_every == 0 and replay_buffer.is_ready(args.batch_size):
            online_net.train()
            loss_val = dqn_update(
                online_net, target_net, optimizer, replay_buffer,
                batch_size=args.batch_size, gamma=args.gamma, device=device,
            )
            train_step_count += 1

            # Hard target update every TARGET_UPDATE_EVERY gradient steps
            # (paper Algorithm 1 line 18: "Set θ = θ⁻ every N steps")
            if train_step_count % args.target_update_every == 0:
                hard_update(online_net, target_net)
                print(f"[Train] Hard target update at step {train_step_count}")

            print(f"[Train] Loss={loss_val:.4f}  train_steps={train_step_count}")

        # --- Epsilon decay (1.0 → eps_end over eps_decay episodes) ---
        if eps_decay > 0:
            progress = min(1.0, (ep - 1) / eps_decay)
            epsilon = eps_start + (eps_end - eps_start) * progress
        else:
            epsilon = eps_end

        # --- Alpha ramp (alp_start → 1.0 over alp_decay episodes) ---
        if alp_decay > 0:
            progress = min(1.0, (ep - 1) / alp_decay)
            alpha = alp_start + (alp_end - alp_start) * progress
        else:
            alpha = alp_end

        # --- Log ---
        current_lr = optimizer.param_groups[0]["lr"]
        row = {
            "episode":               ep,
            "total_reward":          episode_reward,
            "normalised_reward":     episode_reward / max(1, n_trans),
            "steps":                 n_trans,
            "mean_loss":             loss_val,
            "completed_requests":    ep_stats["completed_requests"],
            "picked_up_requests":    ep_stats["picked_up_requests"],
            "avg_wait_until_pickup": ep_stats["avg_wait_until_pickup"],
            "avg_excess_ride_time":  ep_stats["avg_excess_ride_time"],
            "epsilon":               episode_epsilon,
            "alpha":                 episode_alpha,
            "lr":                    current_lr,
            # eval columns left empty (no separate eval loop)
            "eval_completed_requests":    None,
            "eval_picked_up_requests":    None,
            "eval_avg_wait_until_pickup": None,
            "eval_avg_excess_ride_time":  None,
            "eval_eval_total_reward":     None,
            "eval_eval_steps":            None,
        }
        log.append(row)
        csv_writer.writerow(row)
        csv_file.flush()

        # --- Checkpoint (best reward) ---
        if episode_reward > best_episode_reward:
            best_episode_reward = episode_reward
            ckpt_path = ckpt_dir / "best_zone_dqn.pt"
            torch.save({
                "episode":        ep,
                "online_net":     online_net.state_dict(),
                "target_net":     target_net.state_dict(),
                "optimizer":      optimizer.state_dict(),
                "epsilon":        episode_epsilon,
                "alpha":          episode_alpha,
                "train_steps":    train_step_count,
                "episode_reward": episode_reward,
                "grid_rows":      args.grid_rows,
                "grid_cols":      args.grid_cols,
            }, ckpt_path)
            print(f"[Train] ★ New best checkpoint saved → {ckpt_path}")

        # Periodic checkpoint every 10 episodes
        if ep % 10 == 0:
            periodic_path = ckpt_dir / f"zone_dqn_ep{ep:04d}.pt"
            torch.save({
                "episode":     ep,
                "online_net":  online_net.state_dict(),
                "target_net":  target_net.state_dict(),
                "optimizer":   optimizer.state_dict(),
                "epsilon":     episode_epsilon,
                "alpha":       episode_alpha,
                "train_steps": train_step_count,
                "grid_rows":   args.grid_rows,
                "grid_cols":   args.grid_cols,
            }, periodic_path)

        # Save JSON log (full detail backup alongside CSV)
        log_path = ckpt_dir / "training_log.json"
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)

    csv_file.close()
    print(f"\n[Train] Training complete. Best reward={best_episode_reward:.2f}")
    print(f"[Train] Total gradient steps: {train_step_count}")
    print(f"[Train] Checkpoints saved in: {ckpt_dir}")
    print(f"[Train] Training CSV saved → {csv_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train zone-based repositioning DQN for DRT in SUMO"
    )

    # Environment
    p.add_argument("--cfg",       required=True, help="Path to .sumocfg file")
    p.add_argument("--net",       required=True, help="Path to .net.xml file")
    p.add_argument("--stops",     required=True, help="Path to stops.json file")
    p.add_argument("--grid-cols", type=int, default=10, help="Zone grid columns (default 10)")
    p.add_argument("--grid-rows", type=int, default=7,  help="Zone grid rows    (default 7)")
    p.add_argument("--gui",       action="store_true",  help="Launch SUMO-GUI")

    # Training
    p.add_argument("--episodes",            type=int,   default=200,    help="Total training episodes")
    p.add_argument("--batch-size",          type=int,   default=64,     help="Replay mini-batch size")
    p.add_argument("--buffer-capacity",     type=int,   default=20_000, help="Replay buffer capacity")
    p.add_argument("--gamma",               type=float, default=0.95,   help="Discount factor")
    p.add_argument("--lr",                  type=float, default=1e-3,   help="Adam learning rate")
    p.add_argument("--target-update-every", type=int,   default=10,
                   help="Hard-copy online→target every N gradient steps (default 10)")
    p.add_argument("--epsilon-start",       type=float, default=1.0,    help="Initial ε")
    p.add_argument("--epsilon-end",         type=float, default=0.05,   help="Final ε")
    p.add_argument("--epsilon-decay-ep",    type=int,   default=150,    help="Episodes to decay ε over")
    p.add_argument("--alpha-start",         type=float, default=0.3,
                   help="Initial action probability α (paper: 0.3)")
    p.add_argument("--alpha-end",           type=float, default=1.0,
                   help="Final action probability α (paper: 1.0)")
    p.add_argument("--alpha-decay-ep",      type=int,   default=50,
                   help="Episodes over which α ramps from start to end (default 50)")
    p.add_argument("--train-every",         type=int,   default=1,      help="Train every N episodes")
    p.add_argument("--use-dueling",         action="store_true",        help="Use Dueling DQN")
    p.add_argument("--checkpoint-dir",      default="checkpoints_zone", help="Checkpoint directory")

    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())