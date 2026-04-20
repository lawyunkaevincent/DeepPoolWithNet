"""
convert_log_to_csv.py
---------------------
Convert checkpoints_zone/training_log.json  →  checkpoints_zone/training_history.csv

Matches the schema of RealDQN/DQNetwork/artifacts/dqn_model/training_history.csv.

Run from the DeepPoolZone folder:
    python convert_log_to_csv.py
or point at a different checkpoint dir:
    python convert_log_to_csv.py --ckpt-dir my_checkpoints
"""

import argparse
import csv
import json
from pathlib import Path

FIELDS = [
    "episode", "total_reward", "normalised_reward", "steps",
    "mean_loss",
    "completed_requests", "picked_up_requests",
    "avg_wait_until_pickup", "avg_excess_ride_time",
    "p90_wait_time", "p90_excess_ride_time",
    "epsilon", "lr",
    "eval_completed_requests", "eval_picked_up_requests",
    "eval_avg_wait_until_pickup", "eval_avg_excess_ride_time",
    "eval_p90_wait_time", "eval_p90_excess_ride_time",
    "eval_eval_total_reward", "eval_eval_steps",
]

# Map old JSON keys → new CSV column names for backward compatibility
_REMAP = {
    "episode_reward": "total_reward",
    "transitions":    "steps",
    "loss":           "mean_loss",
}


def convert(ckpt_dir: Path) -> None:
    json_path = ckpt_dir / "training_log.json"
    csv_path  = ckpt_dir / "training_history.csv"

    if not json_path.exists():
        print(f"[Error] Not found: {json_path}")
        return

    with open(json_path, "r") as f:
        log = json.load(f)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for entry in log:
            # Remap old key names
            for old_key, new_key in _REMAP.items():
                if old_key in entry and new_key not in entry:
                    entry[new_key] = entry.pop(old_key)

            # Compute derived fields if missing
            if "normalised_reward" not in entry:
                steps = entry.get("steps", 1) or 1
                entry["normalised_reward"] = entry.get("total_reward", 0) / steps

            # Fill missing eval columns with empty string
            for col in FIELDS:
                if col not in entry:
                    entry[col] = ""

            writer.writerow(entry)

    print(f"Wrote {len(log)} episodes → {csv_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", default="checkpoints_zone", help="Checkpoint directory")
    args = p.parse_args()
    convert(Path(args.ckpt_dir))
