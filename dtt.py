"""
Compute the direct ride time (travel time on the shortest path) for all
2000 trip requests in self_request.rou.xml using SUMO's findRoute.

Output: dtt_results.csv  with columns  person_id, from_edge, to_edge, direct_travel_time
"""

import os
import sys
import xml.etree.ElementTree as ET
import csv

# ── SUMO setup ──────────────────────────────────────────────────────
if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    sys.exit("Please set the SUMO_HOME environment variable.")

import traci

# ── Paths (relative to this script) ────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NET_FILE = os.path.join(SCRIPT_DIR, "SunwaySmallMap", "osm.net.xml")
REQUEST_FILE = os.path.join(SCRIPT_DIR, "SunwaySmallMap", "self_request.rou.xml")
TAXI_FILE = os.path.join(SCRIPT_DIR, "SunwaySmallMap", "taxi_more.rou.xml")
OUTPUT_CSV = os.path.join(SCRIPT_DIR, "dtt_results.csv")

# ── Parse requests from XML ────────────────────────────────────────
def parse_requests(xml_path):
    """Return list of (person_id, from_edge, to_edge) tuples."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    requests = []
    for person in root.findall("person"):
        pid = person.get("id")
        ride = person.find("ride")
        from_edge = ride.get("from")
        to_edge = ride.get("to")
        requests.append((pid, from_edge, to_edge))
    return requests

# ── Main ───────────────────────────────────────────────────────────
def main():
    requests = parse_requests(REQUEST_FILE)
    print(f"Parsed {len(requests)} trip requests.")

    # Start SUMO in headless mode with the network + taxi vType
    # We need the taxi vType loaded so findRoute uses its speed profile
    sumo_cmd = [
        "sumo",
        "-n", NET_FILE,
        "-r", TAXI_FILE,       # loads vType "myTaxi"
        "--no-step-log", "true",
        "--no-warnings", "true",
        "--begin", "0",
        "--end", "1",          # we only need routing, not a full sim
    ]
    traci.start(sumo_cmd)

    # Do one step so the network is fully initialised
    traci.simulationStep()

    vtype = "myTaxi"
    results = []
    failed = 0

    for pid, from_edge, to_edge in requests:
        try:
            route = traci.simulation.findRoute(from_edge, to_edge, vtype, routingMode=1)
            dtt = route.travelTime
        except Exception as e:
            print(f"  [WARN] person {pid}: findRoute({from_edge} -> {to_edge}) failed: {e}")
            dtt = -1.0
            failed += 1
        results.append((pid, from_edge, to_edge, dtt))

    traci.close()

    # ── Write CSV ──────────────────────────────────────────────────
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["person_id", "from_edge", "to_edge", "direct_travel_time"])
        writer.writerows(results)

    # ── Summary stats ──────────────────────────────────────────────
    valid_times = [t for _, _, _, t in results if t > 0]
    print(f"\nResults written to {OUTPUT_CSV}")
    print(f"Total requests : {len(results)}")
    print(f"Successful     : {len(valid_times)}")
    print(f"Failed         : {failed}")
    if valid_times:
        print(f"Min DTT        : {min(valid_times):.2f} s")
        print(f"Max DTT        : {max(valid_times):.2f} s")
        print(f"Mean DTT       : {sum(valid_times) / len(valid_times):.2f} s")
        total = sum(valid_times)
        print(f"Total DTT      : {total:.2f} s  ({total / 3600:.2f} h)")


if __name__ == "__main__":
    main()
