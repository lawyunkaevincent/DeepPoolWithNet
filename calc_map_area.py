import argparse
import math
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_location(net_xml_path: Path):
    for _, elem in ET.iterparse(net_xml_path, events=("start",)):
        if elem.tag == "location":
            return elem.attrib
    raise ValueError(f"No <location> element found in {net_xml_path}")


def geodesic_bbox_area_km2(min_lon, min_lat, max_lon, max_lat):
    R = 6371.0088
    lat1 = math.radians(min_lat)
    lat2 = math.radians(max_lat)
    dlon = math.radians(max_lon - min_lon)
    return R * R * dlon * (math.sin(lat2) - math.sin(lat1))


def main():
    ap = argparse.ArgumentParser(description="Report the area covered by a SUMO .net.xml map.")
    ap.add_argument(
        "net_xml",
        nargs="?",
        default="SunwaySmallMap/osm.net.xml",
        help="Path to the SUMO .net.xml file (default: SunwaySmallMap/osm.net.xml)",
    )
    args = ap.parse_args()

    path = Path(args.net_xml)
    loc = parse_location(path)

    conv = [float(v) for v in loc["convBoundary"].split(",")]
    cx_min, cy_min, cx_max, cy_max = conv
    width_m = cx_max - cx_min
    height_m = cy_max - cy_min
    area_m2 = width_m * height_m
    area_km2 = area_m2 / 1_000_000

    orig = [float(v) for v in loc["origBoundary"].split(",")]
    o_lon_min, o_lat_min, o_lon_max, o_lat_max = orig
    geo_area_km2 = geodesic_bbox_area_km2(o_lon_min, o_lat_min, o_lon_max, o_lat_max)

    print(f"File: {path}")
    print(f"Projection: {loc.get('projParameter', '(none)')}")
    print()
    print("Projected bounding box (UTM metres):")
    print(f"  x: {cx_min:.2f} -> {cx_max:.2f}  (width  = {width_m:.2f} m)")
    print(f"  y: {cy_min:.2f} -> {cy_max:.2f}  (height = {height_m:.2f} m)")
    print(f"  area = {area_m2:,.2f} m^2  =  {area_km2:.4f} km^2")
    print()
    print("Original geographic bounding box (WGS84):")
    print(f"  lon: {o_lon_min:.6f} -> {o_lon_max:.6f}  (delta = {o_lon_max - o_lon_min:.6f} deg)")
    print(f"  lat: {o_lat_min:.6f} -> {o_lat_max:.6f}  (delta = {o_lat_max - o_lat_min:.6f} deg)")
    print(f"  geodesic bbox area = {geo_area_km2:.4f} km^2")


if __name__ == "__main__":
    main()
