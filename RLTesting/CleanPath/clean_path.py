import argparse
from dataclasses import dataclass, field
from importlib.resources import path
from pathlib import Path
from typing import Dict, List, Set, Iterable, Optional
import xml.etree.ElementTree as ET
import json
import sumolib


@dataclass
class EdgeStats:
    edge_id: str
    unreachable_count: int = 0
    reachable_to: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "edge_id": self.edge_id,
            "unreachable_count": self.unreachable_count,
            "reachable_to": self.reachable_to,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EdgeStats":
        return cls(
            edge_id=data["edge_id"],
            unreachable_count=data.get("unreachable_count", 0),
            reachable_to=data.get("reachable_to", []),
        )

    def reachable_count(self) -> int:
        return len(self.reachable_to)


class SumoConnectivityChecker:
    def __init__(
        self,
        net_file: str,
        vclass: str = "taxi",
        allow_internal: bool = False,
    ) -> None:
        self.net_file = Path(net_file).resolve()
        self.vclass = vclass
        self.allow_internal = allow_internal

        if not self.net_file.exists():
            raise FileNotFoundError(f"Network file not found: {self.net_file}")

        self.net = sumolib.net.readNet(self.net_file.as_posix())

    def is_valid_edge(self, edge) -> bool:
        edge_id = edge.getID()

        if not self.allow_internal and edge_id.startswith(":"):
            return False

        if hasattr(edge, "getFunction"):
            try:
                if not self.allow_internal and edge.getFunction() == "internal":
                    return False
            except Exception:
                pass

        return edge.allows(self.vclass) and edge.allows("passenger")

    def get_edges_from_net(self) -> List[str]:
        edge_ids = []
        for edge in self.net.getEdges():
            if self.is_valid_edge(edge):
                edge_ids.append(edge.getID())
        print(f"Found {len(edge_ids)} valid edges in the network.")
        return sorted(edge_ids)

    def get_edges_from_route_file(self, route_file: str) -> List[str]:
        route_file = Path(route_file).resolve()
        tree = ET.parse(route_file)
        root = tree.getroot()
        edge_ids: Set[str] = set()

        for person in root.findall("person"):
            ride = person.find("ride")
            if ride is None:
                continue

            from_edge = ride.get("from")
            to_edge = ride.get("to")

            if from_edge and self._edge_exists_and_allowed(from_edge):
                edge_ids.add(from_edge)
            if to_edge and self._edge_exists_and_allowed(to_edge):
                edge_ids.add(to_edge)

        print(f"Found {len(edge_ids)} unique edges in the route file.")
        return sorted(edge_ids)

    def _edge_exists_and_allowed(self, edge_id: str) -> bool:
        try:
            edge = self.net.getEdge(edge_id)
            return self.is_valid_edge(edge)
        except Exception:
            return False

    def analyze(self, edge_ids: Iterable[str]) -> Dict[str, EdgeStats]:
        edge_ids = list(edge_ids)
        results: Dict[str, EdgeStats] = {
            eid: EdgeStats(edge_id=eid) for eid in edge_ids
        }

        for i, source_id in enumerate(edge_ids):
            source_edge = self.net.getEdge(source_id)

            for target_id in edge_ids:
                if source_id == target_id:
                    continue

                target_edge = self.net.getEdge(target_id)
                path, _ = self.net.getShortestPath(source_edge, target_edge, vClass=self.vclass)
                if path:
                    results[source_id].reachable_to.append(target_id)
                else:
                    results[source_id].unreachable_count += 1

            print(
                f"[{i+1}/{len(edge_ids)}] checked {source_id}: "
                f"{results[source_id].unreachable_count} unreachable, "
                f"{results[source_id].reachable_count()} reachable"
            )

        return results

    @staticmethod
    def save_json(results: Dict[str, EdgeStats], output_file: str, total_candidates: int) -> None:
        output_path = Path(output_file).resolve()
        data = {
            "total_candidates": total_candidates,
            "results": {
                eid: stats.to_dict()
                for eid, stats in results.items()
            }
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def load_json(input_file: str) -> tuple[Dict[str, EdgeStats], int]:
        input_path = Path(input_file).resolve()

        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        total_candidates = data["total_candidates"]

        results = {
            eid: EdgeStats.from_dict(stats)
            for eid, stats in data["results"].items()
        }

        return results, total_candidates

    @staticmethod
    def clean_route_file(
        input_route_file: str,
        output_route_file: str,
        bad_edges: Set[str]
    ) -> None:
        input_path = Path(input_route_file).resolve()
        output_path = Path(output_route_file).resolve()

        tree = ET.parse(input_path)
        root = tree.getroot()

        for person in list(root.findall("person")):
            ride = person.find("ride")
            if ride is None:
                continue

            from_edge = ride.get("from")
            to_edge = ride.get("to")

            if from_edge in bad_edges or to_edge in bad_edges:
                root.remove(person)

        tree.write(output_path, encoding="utf-8", xml_declaration=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SUMO edge-connectivity analysis and write a JSON report.",
    )
    parser.add_argument(
        "-n", "--net",
        required=True,
        help="Path to the SUMO .net.xml input file.",
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help=(
            "Output path. If a directory, 'connectivity_report.json' is written inside it; "
            "if a file path, it is used as-is."
        ),
    )
    parser.add_argument(
        "--vclass",
        default="taxi",
        help="Vehicle class to analyze (default: taxi).",
    )
    parser.add_argument(
        "--allow-internal",
        action="store_true",
        help="Include internal edges (IDs starting with ':').",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    net_path = Path(args.net).resolve()
    output_arg = Path(args.output).resolve()
    if output_arg.is_dir() or output_arg.suffix == "":
        output_arg.mkdir(parents=True, exist_ok=True)
        report_path = output_arg / "connectivity_report.json"
    else:
        output_arg.parent.mkdir(parents=True, exist_ok=True)
        report_path = output_arg

    print("Net path:   ", net_path)
    print("Report path:", report_path)

    checker = SumoConnectivityChecker(
        net_file=net_path.as_posix(),
        vclass=args.vclass,
        allow_internal=args.allow_internal,
    )

    edge_ids = checker.get_edges_from_net()
    results = checker.analyze(edge_ids)

    checker.save_json(
        results,
        str(report_path),
        total_candidates=len(edge_ids),
    )
    
    
    