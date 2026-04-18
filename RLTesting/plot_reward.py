# plot_rewards_simple.py
# Always reads rewards.txt in the current folder and plots:
#   - reward vs. episode
#   - moving average (default window = 5)

import re
import pandas as pd
import matplotlib.pyplot as plt

# === Config ===
MA_WINDOW = 10  # <-- change this if you want a different MA window

LOG_FILE = "reward2.txt"
LINE_RE = re.compile(
    r"Episode\s+(?P<ep>\d+)"
    r"(?:,\s*epsilon=(?P<eps>[-+]?\d*\.?\d+))?"
    r",\s*total reward=(?P<rew>[-+]?\d*\.?\d+)"
)

# --- Parse log ---
episodes = []
rewards = []

with open(LOG_FILE, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        m = LINE_RE.search(line)
        if m:
            episodes.append(int(m.group("ep")))
            rewards.append(float(m.group("rew")))

if not episodes:
    raise ValueError("No matching lines found in rewards.txt")

df = pd.DataFrame({"episode": episodes, "reward": rewards}).sort_values("episode")

# --- Compute moving average ---
ma = df["reward"].rolling(window=MA_WINDOW, min_periods=1).mean()

# --- Plot ---
plt.figure(figsize=(12, 6))
plt.plot(df["episode"], df["reward"], label="Reward", alpha=0.5)
plt.plot(df["episode"], ma, label=f"MA (window={MA_WINDOW})", linewidth=2)

plt.title("Reward vs Episode")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid(True)
plt.legend(loc="best")
plt.tight_layout()
plt.show()
