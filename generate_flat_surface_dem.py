#!/usr/bin/env python3
"""
generate_flat_surface_dem.py
----------------------------
Generate a flat-surface DEM (GLB mesh + companion JSON + GeoJSON footprint)
from one or more SRT or AirData CSV flight logs.

The output is compatible with the BAMBI QGIS plugin and can be used as the
DEM input in place of a real elevation model (e.g. for aquatic / near-surface
surveys such as shark or manta-ray flights).

The GLB mesh is centred on the GPS bounding box of all input files, spans
±extent_m in local X and Y at local z = 0 (the elevation_msl value becomes
the z-origin so that camera altitudes H m MSL map to local z = H − elevation_msl).

Usage
-----
    python generate_flat_surface_dem.py --inputs <file1> [<file2> ...] [options]

Examples
--------
    # Single SRT file, surface at sea level, output next to the SRT
    python generate_flat_surface_dem.py --inputs flight.SRT

    # Multiple SRT files, 2.5 m surface elevation, custom output folder
    python generate_flat_surface_dem.py --inputs DJI_0001.SRT DJI_0002.SRT \\
        --elevation 2.5 --output /data/dem_output

    # AirData CSV with explicit UTM zone
    python generate_flat_surface_dem.py --inputs flight_airdata.csv \\
        --elevation 0 --epsg 32643 --output /data/dem_output

Requirements
------------
    pip install numpy pyproj gltflib
    (bambi package must be importable for SRT/AirData parsing)
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import List, Tuple


# ──────────────────────────── GPS extraction ────────────────────────────────

def _read_gps_from_srt(path: str) -> List[Tuple[float, float]]:
    """Return (lat, lon) pairs from an SRT flight log."""
    try:
        from bambi.srt.srt_parser import SrtParser
    except ImportError:
        print(f"  [warn] bambi package not importable; cannot parse SRT: {path}", file=sys.stderr)
        return []

    coords = []
    for frame in SrtParser().parse(path):
        lat = frame.latitude
        lon = frame.longitude
        if lat is not None and lon is not None:
            if -90 <= lat <= 90 and -180 <= lon <= 180 and (lat != 0 or lon != 0):
                coords.append((lat, lon))
    return coords


def _read_gps_from_airdata(path: str) -> List[Tuple[float, float]]:
    """Return (lat, lon) pairs from an AirData CSV (video rows only)."""
    try:
        from bambi.airdata.air_data_parser import AirDataParser
    except ImportError:
        print(f"  [warn] bambi package not importable; cannot parse AirData: {path}", file=sys.stderr)
        return []

    coords = []
    try:
        for frame in AirDataParser().parse(path):
            if not frame.isVideo:
                continue
            lat, lon = frame.latitude, frame.longitude
            if lat is not None and lon is not None:
                if -90 <= lat <= 90 and -180 <= lon <= 180 and (lat != 0 or lon != 0):
                    coords.append((lat, lon))
    except Exception as exc:
        print(f"  [warn] Could not read AirData GPS from {path}: {exc}", file=sys.stderr)
    return coords


def _read_gps_from_file(path: str) -> List[Tuple[float, float]]:
    """Auto-detect file type by extension and return (lat, lon) pairs."""
    ext = Path(path).suffix.lower()
    if ext == ".srt":
        return _read_gps_from_srt(path)
    if ext == ".csv":
        return _read_gps_from_airdata(path)
    # Unknown extension: try SRT, fall back to AirData CSV
    coords = _read_gps_from_srt(path)
    return coords if coords else _read_gps_from_airdata(path)


# ──────────────────────────── mesh generation ────────────────────────────────

def generate_flat_surface_mesh(
    lat: float,
    lon: float,
    elevation_msl: float,
    extent_m: float,
    output_glb: str,
    output_json: str,
    epsg: int = 0,
) -> Tuple[str, str]:
    """Generate a flat horizontal GLB mesh and its companion JSON + GeoJSON.

    Parameters
    ----------
    lat, lon        : WGS-84 centroid of the mesh.
    elevation_msl   : Surface altitude in metres above MSL (becomes the z-origin
                      of the DEM so that camera GPS altitude H maps to local
                      z = H − elevation_msl).
    extent_m        : Half-side length of the square mesh in metres.
    output_glb      : Destination path for the GLB file.
    output_json     : Destination path for the companion JSON file.
    epsg            : Target projected CRS (EPSG code).  When 0 the UTM zone
                      is auto-detected from the centroid longitude.

    Returns
    -------
    (output_glb, output_json)
    """
    import gltflib as gl
    from pyproj import Transformer

    import numpy as np

    if not epsg:
        utm_zone = int((lon + 180) / 6) + 1
        epsg = 32600 + utm_zone if lat >= 0 else 32700 + utm_zone

    x_utm, y_utm = Transformer.from_crs(
        "EPSG:4326", f"EPSG:{epsg}", always_xy=True
    ).transform(lon, lat)

    # Flat quad at local z = 0, spanning ±extent_m in X and Y
    e = float(extent_m)
    vertices = np.array([
        [-e, -e, 0.0],
        [ e, -e, 0.0],
        [ e,  e, 0.0],
        [-e,  e, 0.0],
    ], dtype=np.float32)
    uvs     = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    indices = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)

    vb = vertices.tobytes()
    ub = uvs.tobytes()
    ib = indices.flatten().tobytes()

    def _align4(n): return (4 - n % 4) % 4
    u_off = len(vb) + _align4(len(vb))
    i_off = u_off  + len(ub) + _align4(len(ub))
    total = i_off  + len(ib) + _align4(len(ib))
    buf = bytearray(total)
    buf[0:len(vb)]              = vb
    buf[u_off:u_off + len(ub)] = ub
    buf[i_off:i_off + len(ib)] = ib

    model = gl.GLTFModel(
        asset=gl.Asset(version="2.0"), scene=0,
        scenes=[gl.Scene(nodes=[0])],
        nodes=[gl.Node(mesh=0)],
        meshes=[gl.Mesh(primitives=[gl.Primitive(
            attributes=gl.Attributes(POSITION=0, TEXCOORD_0=1), indices=2,
        )])],
        bufferViews=[
            gl.BufferView(buffer=0, byteOffset=0,     byteLength=len(vb)),
            gl.BufferView(buffer=0, byteOffset=u_off, byteLength=len(ub)),
            gl.BufferView(buffer=0, byteOffset=i_off, byteLength=len(ib)),
        ],
        accessors=[
            gl.Accessor(bufferView=0,
                        componentType=gl.ComponentType.FLOAT.value, count=4,
                        type=gl.AccessorType.VEC3.value,
                        min=vertices.min(axis=0).tolist(),
                        max=vertices.max(axis=0).tolist()),
            gl.Accessor(bufferView=1,
                        componentType=gl.ComponentType.FLOAT.value, count=4,
                        type=gl.AccessorType.VEC2.value),
            gl.Accessor(bufferView=2,
                        componentType=gl.ComponentType.UNSIGNED_INT.value, count=6,
                        type=gl.AccessorType.SCALAR.value),
        ],
        buffers=[gl.Buffer(byteLength=total)],
    )
    gl.GLTF(model=model, resources=[gl.GLBResource(data=bytes(buf))]).export_glb(output_glb)

    dem_meta = {
        "origin": [x_utm, y_utm, float(elevation_msl)],
        "origin_wgs84": {
            "latitude": lat,
            "longitude": lon,
            "altitude": float(elevation_msl),
        },
        "crs": f"EPSG:{epsg}",
    }
    with open(output_json, "w") as f:
        json.dump(dem_meta, f, indent=2)

    # GeoJSON footprint for visualization in QGIS / any GIS tool
    inv = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    corners_utm = [
        (x_utm - e, y_utm - e),
        (x_utm + e, y_utm - e),
        (x_utm + e, y_utm + e),
        (x_utm - e, y_utm + e),
    ]
    ring = [[round(lo, 8), round(la, 8)]
            for lo, la in (inv.transform(cx, cy) for cx, cy in corners_utm)]
    ring.append(ring[0])
    geojson_path = output_json.replace(".json", ".geojson")
    with open(geojson_path, "w") as f:
        json.dump({
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [ring]},
                "properties": {
                    "elevation_msl": float(elevation_msl),
                    "extent_m": float(extent_m),
                    "crs": f"EPSG:{epsg}",
                },
            }],
        }, f, indent=2)

    return output_glb, output_json


# ──────────────────────────────── CLI ────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a flat-surface DEM from SRT or AirData CSV flight logs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_flat_surface_dem.py --inputs flight.SRT
  python generate_flat_surface_dem.py --inputs DJI_0001.SRT DJI_0002.SRT --elevation 2.5
  python generate_flat_surface_dem.py --inputs flight_airdata.csv --epsg 32643 --output /data/dem
""",
    )
    parser.add_argument(
        "--inputs", nargs="+", metavar="FILE", required=True,
        help="One or more SRT (.srt) or AirData (.csv) flight log files",
    )
    parser.add_argument(
        "--elevation", "-e", type=float, default=0.0, metavar="MSL",
        help="Flat surface elevation in metres above MSL (default: 0.0)",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None, metavar="DIR",
        help="Output folder (default: folder of the first input file)",
    )
    parser.add_argument(
        "--epsg", type=int, default=0,
        help="Target projected CRS as EPSG code (default: 0 = auto-detect UTM zone)",
    )
    parser.add_argument(
        "--margin", type=float, default=50.0, metavar="M",
        help="Extra margin in metres added to the GPS bounding-box half-diagonal (default: 50.0)",
    )
    parser.add_argument(
        "--name", type=str, default="flat_surface_dem",
        help="Base name for output files without extension (default: flat_surface_dem)",
    )

    args = parser.parse_args()

    for p in args.inputs:
        if not os.path.isfile(p):
            print(f"Error: file not found: {p}", file=sys.stderr)
            sys.exit(1)

    # Collect GPS coordinates from all input files
    lats: List[float] = []
    lons: List[float] = []
    for p in args.inputs:
        print(f"Reading GPS from: {p}")
        coords = _read_gps_from_file(p)
        if coords:
            file_lats = [c[0] for c in coords]
            file_lons = [c[1] for c in coords]
            lats.extend(file_lats)
            lons.extend(file_lons)
            print(f"  {len(coords)} GPS fixes — "
                  f"lat=[{min(file_lats):.5f}, {max(file_lats):.5f}]  "
                  f"lon=[{min(file_lons):.5f}, {max(file_lons):.5f}]")
        else:
            print(f"  [warn] No GPS data found in {p}")

    if not lats:
        print("Error: no GPS coordinates found in any of the input files.", file=sys.stderr)
        sys.exit(1)

    # Centroid + extent (same formula as the QGIS plugin)
    lat = (min(lats) + max(lats)) / 2
    lon = (min(lons) + max(lons)) / 2
    delta_y   = (max(lats) - min(lats)) * 111320.0
    delta_x   = (max(lons) - min(lons)) * 111320.0 * math.cos(math.radians(lat))
    half_diag = math.sqrt((delta_x / 2) ** 2 + (delta_y / 2) ** 2)
    extent_m  = max(half_diag + args.margin, args.margin)

    epsg_label = str(args.epsg) if args.epsg else "auto-detect UTM"
    print(f"\nCentroid : lat={lat:.6f}, lon={lon:.6f}")
    print(f"Elevation: {args.elevation:.1f} m MSL")
    print(f"Extent   : ±{extent_m:.1f} m  (half-diagonal {half_diag:.1f} m + {args.margin:.0f} m margin)")
    print(f"CRS      : EPSG:{epsg_label}")

    out_dir = args.output or str(Path(args.inputs[0]).parent)
    os.makedirs(out_dir, exist_ok=True)

    output_glb  = os.path.join(out_dir, f"{args.name}.glb")
    output_json = os.path.join(out_dir, f"{args.name}.json")

    print("\nGenerating flat surface DEM …")
    glb_path, json_path = generate_flat_surface_mesh(
        lat, lon, args.elevation, extent_m, output_glb, output_json, epsg=args.epsg
    )

    geojson_path = json_path.replace(".json", ".geojson")
    print("\nOutput files:")
    print(f"  GLB     : {glb_path}")
    print(f"  JSON    : {json_path}")
    print(f"  GeoJSON : {geojson_path}")


if __name__ == "__main__":
    main()
