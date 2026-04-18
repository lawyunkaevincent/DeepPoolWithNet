from __future__ import annotations

import argparse
import json
import random
import xml.etree.ElementTree as ET
from pathlib import Path


def load_stops(stops_json: str | Path) -> list[str]:
    with open(stops_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    stops = data.get("stops", [])
    if not stops:
        raise ValueError(f"No stops found in {stops_json}.")
    return stops


def generate_taxi_rou(
    stops: list[str],
    num_taxis: int,
    output_file: str | Path,
    depart: float = 0.0,
    vtype_id: str = "myTaxi",
    color: str = "yellow",
    person_capacity: int = 4,
    seed: int = 42,
) -> list[str]:
    """Generate a taxi.rou.xml where each taxi starts on a stop edge.

    Each taxi trip uses the same edge for both ``from`` and ``to``.
    This gives dispatchTaxi a valid minimal route it can fully override
    to reach any pickup/dropoff in the network (see existing taxi.rou.xml).

    Args:
        stops:           List of eligible stop edge IDs (from stops.json).
        num_taxis:       Number of taxi trips to generate.
        output_file:     Destination .rou.xml path.
        depart:          Depart time for all taxis (default 0.0).
        vtype_id:        Vehicle type ID string (default "myTaxi").
        color:           Taxi colour (default "yellow").
        person_capacity: Max persons per taxi (default 4).
        seed:            Random seed for stop assignment.

    Returns:
        List of edge IDs assigned to each taxi (in order).
    """
    if num_taxis < 1:
        raise ValueError("num_taxis must be at least 1.")

    rng = random.Random(seed)

    # If we need more taxis than stops, allow repeats; otherwise sample without replacement
    if num_taxis <= len(stops):
        assigned = rng.sample(stops, num_taxis)
    else:
        assigned = [rng.choice(stops) for _ in range(num_taxis)]

    root = ET.Element(
        "routes",
        {
            "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "xsi:noNamespaceSchemaLocation": "http://sumo.dlr.de/xsd/routes_file.xsd",
        },
    )

    # vType
    vtype = ET.SubElement(
        root,
        "vType",
        {
            "id": vtype_id,
            "vClass": "taxi",
            "color": color,
            "personCapacity": str(person_capacity),
        },
    )
    ET.SubElement(vtype, "param", {"key": "has.taxi.device", "value": "true"})

    ET.Comment(
        " from=X to=X gives a valid minimal route that dispatchTaxi can fully "
        "override to reach any pickup/dropoff in the network. "
    )

    # One trip per taxi
    depart_str = str(round(depart)) if depart == int(depart) else f"{depart:.2f}"
    for i, edge in enumerate(assigned):
        ET.SubElement(
            root,
            "trip",
            {
                "id": f"t_{i}",
                "type": vtype_id,
                "depart": depart_str,
                "from": edge,
                "to": edge,
            },
        )

    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ")
    tree.write(output_file, encoding="utf-8", xml_declaration=True)
    return assigned


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a SUMO taxi.rou.xml from a stops.json file."
    )
    parser.add_argument(
        "--stops",
        required=True,
        metavar="STOPS_JSON",
        help="Path to stops.json produced by request_chain_generator.py",
    )
    parser.add_argument(
        "--output",
        required=True,
        metavar="TAXI_ROU_XML",
        help="Path to write the output taxi.rou.xml",
    )
    parser.add_argument(
        "--num-taxis",
        type=int,
        default=2,
        help="Number of taxis to generate (default: 2)",
    )
    parser.add_argument(
        "--depart",
        type=float,
        default=0.0,
        help="Depart time for all taxis (default: 0.0)",
    )
    parser.add_argument(
        "--vtype-id",
        default="myTaxi",
        help="Vehicle type ID (default: myTaxi)",
    )
    parser.add_argument(
        "--color",
        default="yellow",
        help="Taxi colour string recognised by SUMO (default: yellow)",
    )
    parser.add_argument(
        "--person-capacity",
        type=int,
        default=4,
        help="Maximum persons per taxi (default: 4)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for stop assignment (default: 42)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stops = load_stops(args.stops)

    assigned = generate_taxi_rou(
        stops=stops,
        num_taxis=args.num_taxis,
        output_file=args.output,
        depart=args.depart,
        vtype_id=args.vtype_id,
        color=args.color,
        person_capacity=args.person_capacity,
        seed=args.seed,
    )

    print(f"Taxis generated : {len(assigned)}")
    print(f"Output          : {args.output}")
    print(f"Stop pool size  : {len(stops)}")
    print()
    print("Assigned edges:")
    for i, edge in enumerate(assigned):
        print(f"  t_{i:<4} -> {edge}")


if __name__ == "__main__":
    main()
