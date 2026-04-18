# plot_reward.py
# Plots reward vs episode for TWO logs on one chart:
#   - reward_map1.txt  (Structured Grip Map)
#   - reward_map2.txt  (Sunway Area Network)
# Shows both raw series and moving average (window=10, Pandas rolling).

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ==== Config ====
MA_WINDOW = 10  # moving-average window (episodes)
FILES = [
    ("Structured Grip Map", "reward_map1.txt"),
    ("Sunway Area Network", "reward_map2.txt"),
]

# Color scheme
COLORS = {
    "Structured Grip Map": {"raw": "#9ecae1",  "ma": "#08519c"},  # light blue / dark blue
    "Sunway Area Network": {"raw": "#f4a582",  "ma": "#ca0020"},  # light red  / dark red
}

LINE_RE = re.compile(
    r"Episode\s+(?P<ep>\d+)"
    r"(?:,\s*epsilon=(?P<eps>[-+]?\d*\.?\d+))?"
    r",\s*total reward=(?P<rew>[-+]?\d*\.?\d+)"
)

def load_rewards(path: Path) -> pd.DataFrame:
    episodes, rewards = [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = LINE_RE.search(line)
            if m:
                episodes.append(int(m.group("ep")))
                rewards.append(float(m.group("rew")))
    if not episodes:
        raise ValueError(f"No matching lines found in {path}")
    df = pd.DataFrame({"episode": episodes, "reward": rewards})
    return df.sort_values("episode").reset_index(drop=True)

def main():
    series = []
    for label, fname in FILES:
        p = Path(fname)
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p.resolve()}")
        df = load_rewards(p)

        # Original rolling MA (defined for all episodes)
        # Set min_periods=1 to start MA immediately; change to =MA_WINDOW if you
        # prefer NaNs for the first window-1 episodes.
        df["ma"] = df["reward"].rolling(window=MA_WINDOW, min_periods=1).mean()

        series.append((label, df))

    plt.figure(figsize=(12, 6))

    # Raw (lighter)
    for label, df in series:
        plt.plot(
            df["episode"], df["reward"],
            color=COLORS[label]["raw"], alpha=0.9, linewidth=1.2,
            label=f"{label} (raw)"
        )

    # Moving average (thicker, dark)
    for label, df in series:
        plt.plot(
            df["episode"], df["ma"],
            color=COLORS[label]["ma"], linewidth=2.8,
            label=f"{label} MA (window={MA_WINDOW})"
        )

    plt.title("Total Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
