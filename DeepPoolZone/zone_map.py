"""
zone_map.py
-----------
Partitions a SUMO road network into a rectangular grid of zones.

Reads the SUMO .net.xml file (without needing TraCI/SUMO running) to extract
each edge's midpoint coordinates, then assigns every edge to a grid cell.

Grid layout
-----------
The SUMO network uses a local Cartesian coordinate system whose bounding box
is given in the <location convBoundary="xmin,ymin,xmax,ymax"> tag.

We lay a GRID_COLS × GRID_ROWS grid over that box:

    zone_id = row * GRID_COLS + col     (row 0 = bottom, col 0 = left)

Public API
----------
    zm = ZoneMap("osm.net.xml", grid_cols=10, grid_rows=7)
    zone_id = zm.edge_to_zone("edge_id")      # -1 if edge has no geometry
    edges   = zm.zone_to_edges(zone_id)       # list of edge IDs in that zone
    (col, row) = zm.zone_to_cell(zone_id)
    centroid_xy = zm.zone_centroid(zone_id)   # (x, y) in SUMO network coords
    random_edge = zm.sample_edge(zone_id)     # random accessible edge in zone
"""

from __future__ import annotations

import json
import random
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# Vehicle classes that taxis belong to.  An edge is taxi-accessible if at
# least one of its lanes has an `allow` list that intersects this set,
# OR has no `allow` attribute at all (uses `disallow` only, which keeps taxis).
_TAXI_CLASSES: frozenset[str] = frozenset({
    "passenger", "taxi", "private", "emergency", "authority",
    "army", "vip", "hov", "bus", "coach", "delivery",
    "truck", "trailer", "motorcycle", "evehicle", "custom1", "custom2",
})

# Very short lanes are fragile idle targets for SUMO taxis. Keep reposition
# stops on lanes long enough to accommodate a stable waiting position.
_MIN_STOP_LANE_LENGTH = 25.0


def _lane_allows_taxi(allow_str: str) -> bool:
    """
    Return True if a lane's `allow` attribute permits taxi-class vehicles.

    Called only when the lane actually has an `allow` attribute.  A lane with
    no `allow` attribute (uses `disallow` instead) is assumed accessible.
    """
    classes = set(allow_str.split())
    return bool(classes & _TAXI_CLASSES)


@dataclass(frozen=True)
class ZoneStop:
    edge_id: str
    lane_index: int
    pos: float
    lane_length: float


class ZoneMap:
    """
    Static mapping between SUMO edges and rectangular grid zones.

    Only taxi-accessible edges are stored — edges where every lane has an
    explicit `allow` list that excludes passenger vehicles (e.g. footways,
    cycleways, service paths) are silently dropped.

    Parameters
    ----------
    net_xml_path : str
        Path to the SUMO network file (.net.xml).
    grid_cols : int
        Number of columns in the grid (east-west direction).
    grid_rows : int
        Number of rows in the grid (north-south direction).
    """

    def __init__(
        self,
        net_xml_path: str,
        grid_cols: int = 10,
        grid_rows: int = 7,
    ) -> None:
        self.grid_cols = grid_cols
        self.grid_rows = grid_rows
        self.n_zones = grid_cols * grid_rows

        # Parse the network XML
        self._xmin, self._ymin, self._xmax, self._ymax = self._parse_bounds(net_xml_path)
        self._cell_w = (self._xmax - self._xmin) / grid_cols
        self._cell_h = (self._ymax - self._ymin) / grid_rows

        # Build edge ↔ zone mappings
        self._edge_to_zone: Dict[str, int] = {}          # edge_id → zone_id
        self._zone_to_edges: Dict[int, List[str]] = defaultdict(list)  # zone_id → [edge_ids]
        self._edge_midpoints: Dict[str, Tuple[float, float]] = {}
        self._edge_stop_meta: Dict[str, ZoneStop] = {}

        # Stop-only mappings (populated by load_stops)
        self._all_stops: List[ZoneStop] = []
        self._zone_to_stops: Dict[int, List[ZoneStop]] = defaultdict(list)

        self._parse_edges(net_xml_path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_bounds(net_xml_path: str) -> Tuple[float, float, float, float]:
        """
        Extract xmin, ymin, xmax, ymax from the <location> tag's convBoundary.

        Example tag:
            <location convBoundary="0.00,0.00,5526.21,3690.78" .../>
        """
        for event, elem in ET.iterparse(net_xml_path, events=("start",)):
            if elem.tag == "location":
                cb = elem.get("convBoundary", "0,0,1,1")
                parts = [float(v) for v in cb.split(",")]
                return parts[0], parts[1], parts[2], parts[3]
        return 0.0, 0.0, 1.0, 1.0

    def _parse_edges(self, net_xml_path: str) -> None:
        """
        Walk the XML and record each taxi-accessible <edge>'s midpoint.

        SUMO edges have one or more <lane> children.  We track two things per
        lane: its shape (for midpoint) and its `allow` attribute (for access
        filtering).

        An edge is kept only if at least one lane is taxi-accessible:
          - Lane has NO `allow` attribute  →  accessible (uses disallow only)
          - Lane HAS `allow` attribute     →  accessible only if it contains a
                                              passenger/taxi vehicle class

        Internal junction edges (id starts with ':') are always skipped.
        """
        current_edge_id: Optional[str] = None
        current_lane_shapes: List[str] = []
        current_edge_taxi_ok: bool = False   # True once one accessible lane found
        current_stop_meta: Optional[ZoneStop] = None

        for event, elem in ET.iterparse(net_xml_path, events=("start", "end")):
            if event == "start":
                if elem.tag == "edge":
                    edge_id = elem.get("id", "")
                    if edge_id.startswith(":"):
                        current_edge_id = None
                    else:
                        current_edge_id = edge_id
                        current_lane_shapes = []
                        current_edge_taxi_ok = False
                        current_stop_meta = None

                elif elem.tag == "lane" and current_edge_id is not None:
                    shape_str = elem.get("shape", "")
                    if shape_str:
                        current_lane_shapes.append(shape_str)

                    # Check taxi accessibility for this lane
                    allow_str = elem.get("allow", None)
                    if allow_str is None:
                        # No explicit allow → uses disallow only → taxis OK
                        current_edge_taxi_ok = True
                        if current_stop_meta is None:
                            current_stop_meta = self._build_stop_meta(current_edge_id, elem)
                    elif _lane_allows_taxi(allow_str):
                        current_edge_taxi_ok = True
                        if current_stop_meta is None:
                            current_stop_meta = self._build_stop_meta(current_edge_id, elem)

            elif event == "end":
                if elem.tag == "edge" and current_edge_id is not None:
                    if current_lane_shapes and current_edge_taxi_ok:
                        mx, my = self._shape_midpoint(current_lane_shapes[0])
                        self._edge_midpoints[current_edge_id] = (mx, my)
                        if current_stop_meta is not None:
                            self._edge_stop_meta[current_edge_id] = current_stop_meta
                        zone_id = self._xy_to_zone(mx, my)
                        if zone_id >= 0:
                            self._edge_to_zone[current_edge_id] = zone_id
                            self._zone_to_edges[zone_id].append(current_edge_id)
                    current_edge_id = None

    @staticmethod
    def _build_stop_meta(edge_id: str, lane_elem: ET.Element) -> ZoneStop:
        lane_index = int(lane_elem.get("index", "0"))
        lane_length = float(lane_elem.get("length", "10.0"))
        if lane_length <= 2.0:
            pos = max(0.5, lane_length * 0.5)
        else:
            pos = min(lane_length - 1.0, max(1.0, lane_length * 0.5))
        return ZoneStop(edge_id=edge_id, lane_index=lane_index, pos=pos, lane_length=lane_length)

    @staticmethod
    def _shape_midpoint(shape_str: str) -> Tuple[float, float]:
        """
        Parse a SUMO lane shape string ("x1,y1 x2,y2 ...") and return the
        coordinates of the midpoint along the shape polyline.
        """
        points = []
        for token in shape_str.strip().split():
            parts = token.split(",")
            if len(parts) >= 2:
                try:
                    points.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    continue
        if not points:
            return 0.0, 0.0
        if len(points) == 1:
            return points[0]
        # Midpoint of the whole polyline: average of first and last point
        mx = (points[0][0] + points[-1][0]) / 2.0
        my = (points[0][1] + points[-1][1]) / 2.0
        return mx, my

    def _xy_to_zone(self, x: float, y: float) -> int:
        """
        Convert network coordinates (x, y) to a zone_id.
        Returns -1 if the point is outside the network bounds.
        """
        if x < self._xmin or x >= self._xmax or y < self._ymin or y >= self._ymax:
            # Clamp boundary edges (floating-point fuzz at the border)
            x = max(self._xmin, min(x, self._xmax - 1e-6))
            y = max(self._ymin, min(y, self._ymax - 1e-6))

        col = int((x - self._xmin) / self._cell_w)
        row = int((y - self._ymin) / self._cell_h)

        col = max(0, min(col, self.grid_cols - 1))
        row = max(0, min(row, self.grid_rows - 1))

        return row * self.grid_cols + col

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def edge_to_zone(self, edge_id: str) -> int:
        """Return zone_id for an edge, or -1 if unknown."""
        return self._edge_to_zone.get(edge_id, -1)

    def zone_to_edges(self, zone_id: int) -> List[str]:
        """Return all edge IDs mapped to this zone (may be empty)."""
        return self._zone_to_edges.get(zone_id, [])

    def zone_to_cell(self, zone_id: int) -> Tuple[int, int]:
        """Convert zone_id → (col, row)."""
        row = zone_id // self.grid_cols
        col = zone_id % self.grid_cols
        return col, row

    def cell_to_zone(self, col: int, row: int) -> int:
        """Convert (col, row) → zone_id.  No bounds checking."""
        return row * self.grid_cols + col

    def zone_centroid(self, zone_id: int) -> Tuple[float, float]:
        """Return the (x, y) centroid of a zone cell in network coordinates."""
        col, row = self.zone_to_cell(zone_id)
        x = self._xmin + (col + 0.5) * self._cell_w
        y = self._ymin + (row + 0.5) * self._cell_h
        return x, y

    def sample_edge(self, zone_id: int, rng: Optional[random.Random] = None) -> Optional[str]:
        """
        Return a random edge from the given zone, or None if the zone is empty.

        Parameters
        ----------
        zone_id : int
        rng : optional random.Random instance (uses module-level random if None)
        """
        edges = self._zone_to_edges.get(zone_id, [])
        if not edges:
            return None
        return (rng or random).choice(edges)

    def nearest_non_empty_zone(self, zone_id: int) -> int:
        """
        Find the closest zone (by grid distance) that has at least one edge.
        Useful as a fallback when the agent selects an empty zone.
        """
        if self._zone_to_edges.get(zone_id):
            return zone_id

        col0, row0 = self.zone_to_cell(zone_id)
        best_zone = zone_id
        best_dist = float("inf")

        for zid, edges in self._zone_to_edges.items():
            if not edges:
                continue
            c, r = self.zone_to_cell(zid)
            dist = abs(c - col0) + abs(r - row0)  # Manhattan distance
            if dist < best_dist:
                best_dist = dist
                best_zone = zid

        return best_zone

    # ------------------------------------------------------------------
    # Stop-based API  (call load_stops() once before using these)
    # ------------------------------------------------------------------

    def load_stops(self, stops_json_path: str) -> None:
        """
        Read stops.json and build zone → stops index.

        Each stop is a SUMO edge ID that is verified to be reachable from
        anywhere on the main road network.  After calling this, use
        zone_to_stops() instead of zone_to_edges() for repositioning targets.
        """
        with open(stops_json_path, "r") as f:
            data = json.load(f)
        raw_stops = data.get("stops", [])

        self._all_stops.clear()
        self._zone_to_stops.clear()

        unzoned = 0
        skipped_short = 0
        for raw_stop in raw_stops:
            stop = self._coerce_stop(raw_stop)
            if stop is None:
                skipped_short += 1
                continue
            edge_id = stop.edge_id
            zone_id = self._edge_to_zone.get(edge_id, -1)
            if zone_id < 0:
                # Stop edge wasn't in the taxi-accessible edge set (rare).
                # Still keep it in _all_stops as a global fallback, but don't
                # add to any zone bucket.
                unzoned += 1
            else:
                self._zone_to_stops[zone_id].append(stop)
            self._all_stops.append(stop)

        n_zones_with_stops = sum(1 for v in self._zone_to_stops.values() if v)
        print(f"[ZoneMap] Loaded {len(self._all_stops)} stops "
              f"({unzoned} unzoned, {skipped_short} too-short) across {n_zones_with_stops} zones")

    def _coerce_stop(self, raw_stop: object) -> Optional[ZoneStop]:
        if isinstance(raw_stop, str):
            stop = self._edge_stop_meta.get(raw_stop)
            if stop is None or stop.lane_length < _MIN_STOP_LANE_LENGTH:
                return None
            return stop
        if isinstance(raw_stop, dict):
            edge_id = str(raw_stop.get("edge_id", ""))
            if not edge_id:
                return None
            default_meta = self._edge_stop_meta.get(edge_id, ZoneStop(edge_id=edge_id, lane_index=0, pos=1.0, lane_length=10.0))
            lane_index = int(raw_stop.get("lane_index", default_meta.lane_index))
            lane_length = float(raw_stop.get("lane_length", default_meta.lane_length))
            if lane_length < _MIN_STOP_LANE_LENGTH:
                return None
            if lane_length <= 2.0:
                default_pos = max(0.5, lane_length * 0.5)
            else:
                default_pos = min(lane_length - 1.0, max(1.0, lane_length * 0.5))
            pos = float(raw_stop.get("pos", default_pos))
            return ZoneStop(edge_id=edge_id, lane_index=lane_index, pos=pos, lane_length=lane_length)
        return None

    def zone_to_stops(self, zone_id: int) -> List[ZoneStop]:
        """Return stop edges in this zone (empty list if none).  Requires load_stops()."""
        return self._zone_to_stops.get(zone_id, [])

    def nearest_stop(self, zone_id: int, rng: Optional[random.Random] = None) -> Optional[ZoneStop]:
        """
        Return a random stop from the nearest zone (by grid Manhattan distance)
        that has at least one stop.  Falls back to a global random stop if no
        zone has stops (shouldn't happen with a valid stops.json).
        """
        # Check preferred zone first
        stops = self._zone_to_stops.get(zone_id, [])
        if stops:
            return (rng or random).choice(stops)

        # Expand outward zone by zone
        col0, row0 = self.zone_to_cell(zone_id)
        for dist in range(1, max(self.grid_cols, self.grid_rows) + 1):
            for drow in range(-dist, dist + 1):
                for dcol in range(-dist, dist + 1):
                    if abs(drow) != dist and abs(dcol) != dist:
                        continue
                    c, r = col0 + dcol, row0 + drow
                    if not (0 <= c < self.grid_cols and 0 <= r < self.grid_rows):
                        continue
                    z = self.cell_to_zone(c, r)
                    s = self._zone_to_stops.get(z, [])
                    if s:
                        return (rng or random).choice(s)

        # Ultimate fallback: any stop
        return (rng or random).choice(self._all_stops) if self._all_stops else None

    # ------------------------------------------------------------------
    # Debug helpers
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable summary of the zone map."""
        populated = sum(1 for edges in self._zone_to_edges.values() if edges)
        return (
            f"ZoneMap({self.grid_cols}×{self.grid_rows}) "
            f"bounds=({self._xmin:.0f},{self._ymin:.0f})→"
            f"({self._xmax:.0f},{self._ymax:.0f}) "
            f"cell=({self._cell_w:.0f}m×{self._cell_h:.0f}m) "
            f"edges={len(self._edge_to_zone)} "
            f"zones_with_edges={populated}/{self.n_zones}"
        )

    def print_zone_grid(self) -> None:
        """Print an ASCII grid showing edge counts per zone (rows top→bottom)."""
        print(f"\nZone grid ({self.grid_cols} cols × {self.grid_rows} rows):")
        for row in range(self.grid_rows - 1, -1, -1):  # top row first
            cells = []
            for col in range(self.grid_cols):
                zid = self.cell_to_zone(col, row)
                n = len(self._zone_to_edges.get(zid, []))
                cells.append(f"{n:3d}")
            print(f"  row {row:2d} | {'  '.join(cells)}")
        print()
