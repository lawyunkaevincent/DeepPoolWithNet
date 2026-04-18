"""
test_zone_env.py
----------------
Quick sanity-check script.  Runs ONE episode with a RANDOM policy
(no trained network needed) and prints:

  - Zone map summary (how many edges per zone)
  - Sample state grid at the first repositioning decision
  - Total transitions collected
  - Total reward
  - Zone visit frequency (which zones were chosen)

Usage
-----
    cd d:/6Sumo/BenchmarkingDQN/DeepPoolZone
    python test_zone_env.py --cfg ../SunwaySmallMap/osm.sumocfg \
                            --net ../SunwaySmallMap/osm.net.xml

    # To open the SUMO GUI so you can watch taxis moving:
    python test_zone_env.py --cfg ../SunwaySmallMap/osm.sumocfg \
                            --net ../SunwaySmallMap/osm.net.xml \
                            --gui
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

import numpy as np

_HERE = os.path.dirname(__file__)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from zone_map import ZoneMap
from zone_env import ZoneBasedDRTEnv


def main(args: argparse.Namespace) -> None:
    # ------------------------------------------------------------------
    # 1. Inspect the zone map before starting SUMO
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 1: Building zone map (reads net.xml, no SUMO needed)")
    print("=" * 60)
    zm = ZoneMap(args.net, grid_cols=args.grid_cols, grid_rows=args.grid_rows)
    print(zm.summary())
    zm.print_zone_grid()

    # ------------------------------------------------------------------
    # 2. Run one episode with a random policy
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 2: Running one SUMO episode (random repositioning policy)")
    print("=" * 60)

    env = ZoneBasedDRTEnv(
        cfg_path=args.cfg,
        net_xml_path=args.net,
        stops_json_path=args.stops,
        grid_cols=args.grid_cols,
        grid_rows=args.grid_rows,
        policy_fn=None,   # None → random zone selection
        epsilon=1.0,       # 100% random (no trained network yet)
        step_length=1.0,
        use_gui=args.gui,
    )

    transitions = env.run_episode()

    # ------------------------------------------------------------------
    # 3. Report results
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    if not transitions:
        print("  No transitions collected — taxis may never have gone idle.")
        print("  Check that the route file has taxi vehicles with idle time.")
        return

    total_reward = sum(t.reward for t in transitions)
    actions = [t.action for t in transitions]
    zone_counts = Counter(actions)

    print(f"  Transitions collected : {len(transitions)}")
    print(f"  Total reward          : {total_reward:.4f}")
    print(f"  Avg reward/transition : {total_reward / len(transitions):.4f}")
    print(f"  Unique zones visited  : {len(zone_counts)} / {zm.n_zones}")

    # Show the first state grid
    print("\n  First state (demand grid, rows top→bottom):")
    s0 = transitions[0].state  # (2, R, C)
    demand = s0[0]             # (R, C)
    vehicles = s0[1]           # (R, C)
    for row in range(demand.shape[0] - 1, -1, -1):
        row_str = "  ".join(f"{v:.2f}" for v in demand[row])
        print(f"    row {row}: [{row_str}]")

    print("\n  First state (vehicle grid, rows top→bottom):")
    for row in range(vehicles.shape[0] - 1, -1, -1):
        row_str = "  ".join(f"{v:.2f}" for v in vehicles[row])
        print(f"    row {row}: [{row_str}]")

    # Top 10 most-visited zones
    print("\n  Top 10 zones chosen by random policy:")
    for zone_id, count in zone_counts.most_common(10):
        col, row_ = zm.zone_to_cell(zone_id)
        cx, cy = zm.zone_centroid(zone_id)
        n_edges = len(zm.zone_to_edges(zone_id))
        print(f"    zone {zone_id:3d} (col={col}, row={row_}) "
              f"centroid=({cx:.0f}m,{cy:.0f}m) "
              f"edges={n_edges}  chosen={count}x")

    print("\n  Done.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Test zone environment with random policy")
    p.add_argument("--cfg",       required=True,  help="Path to .sumocfg")
    p.add_argument("--net",       required=True,  help="Path to .net.xml")
    p.add_argument("--stops",     required=True,  help="Path to stops.json")
    p.add_argument("--grid-cols", type=int, default=10, help="Zone grid columns (default 10)")
    p.add_argument("--grid-rows", type=int, default=7,  help="Zone grid rows    (default 7)")
    p.add_argument("--gui",       action="store_true",  help="Open SUMO-GUI")
    main(p.parse_args())
