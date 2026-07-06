"""
Export geo-referenced fish-school outline/hole polygons from TRex posture
tracklets (``*_fschool_posture_id<N>.npz``).

Mirrors export_georeferenced_tracklets.py's approach for shark pose
key-points (undistort + ray-cast onto the same flat/DEM plane trex_to_bambi.py
uses, keep every original field, add new "_geor" fields with the result) but
for the *outline* and *hole* polygons instead of a fixed 9 key-points -
see fschool_posture.py for how those are packed into (and unpacked from) the
npz's ragged arrays.

Outline/hole polygons are raw per-pixel contour traces (thousands of points
per frame - see fschool_posture.decimate_polygon) which is both unnecessary
resolution for georeferencing/display and far too slow to ray-cast one point
at a time, so every polygon is decimated to --max-polygon-points before
projection. The georeferenced output therefore has its own (shorter)
"*_geor" length arrays alongside the "*_geor" point arrays - the original,
full-resolution pixel-space fields are left untouched.

Example
-------
python export_georeferenced_fschool.py \
    --npz-dir    "/media/.../sequences-yolo26/data" \
    --sequence   sequence_20240303_060726300_DJI_0242 \
    --video      "/media/.../sequence_20240303_060726300_DJI_0242.mp4" \
    --dem-json   ".../dem/flat_surface_dem.json" \
    --poses      ".../frames_w/poses.json" \
    --calib      ".../blue_drone_combined.json" \
    --mask       ".../frames_w/mask_W.png" \
    --out-dir    ".../fschool_georeferenced_w" \
    --flat-surface-msl 0.0
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

cv2.setNumThreads(int(os.environ.get("OPENCV_NUM_THREADS", "4")))

from alfspy.core.rendering import Resolution
from bambi.util.projection_util import pixel_to_world_coord

from fschool_posture import (
    decimate_polygon,
    find_fschool_posture_files,
    iter_frame_shapes,
)
from trex_to_bambi import (
    Undistorter,
    get_camera_for_frame,
    load_correction,
    load_projection_mesh,
)


def _pack_hole_layout(hole_lengths_per_frame: List[List[int]]) -> np.ndarray:
    """Inverse of fschool_posture.parse_hole_layout: flatten a per-frame list
    of hole-polygon lengths back into the nested [n_holes, len1, len2, ...]
    encoding used by the original npz format."""
    packed: List[int] = []
    for lengths in hole_lengths_per_frame:
        packed.append(len(lengths))
        packed.extend(lengths)
    return np.array(packed, dtype=np.uint64)


def georeference_track(
    path: str, poses: dict, tri_mesh, offsets: Tuple[float, float, float],
    input_resolution: Resolution, cor_rotation_eulers, cor_translation,
    undistorter: Optional[Undistorter], max_polygon_points: int,
) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    x_off, y_off, _z_off = offsets
    n_poses = len(poses["images"])
    aspect_ratio = input_resolution.width / input_resolution.height

    outline_lengths_geor: List[int] = []
    outline_points_geor: List[np.ndarray] = []
    hole_lengths_per_frame: List[List[int]] = []
    hole_points_geor: List[np.ndarray] = []

    camera_cache: Dict[int, object] = {}
    n_outline_hits = n_outline_total = 0
    n_hole_hits = n_hole_total = 0

    for frame_idx, outline_abs, holes_abs in iter_frame_shapes(data):
        outline_dec = decimate_polygon(outline_abs, max_polygon_points)
        holes_dec = [decimate_polygon(h, max_polygon_points) for h in holes_abs]

        if frame_idx >= n_poses:
            outline_lengths_geor.append(len(outline_dec))
            outline_points_geor.append(np.full((len(outline_dec), 2), np.nan))
            hole_lengths_per_frame.append([len(h) for h in holes_dec])
            for h in holes_dec:
                hole_points_geor.append(np.full((len(h), 2), np.nan))
            n_outline_total += len(outline_dec)
            n_hole_total += sum(len(h) for h in holes_dec)
            continue

        # One batched ray-cast per frame for every point (outline + all holes).
        segments = [outline_dec] + holes_dec
        all_pts = np.concatenate(segments, axis=0) if segments else np.empty((0, 2))
        if undistorter is not None and len(all_pts):
            all_pts = undistorter.points(all_pts)

        camera = camera_cache.get(frame_idx)
        if camera is None:
            camera = get_camera_for_frame(poses, frame_idx, cor_rotation_eulers, cor_translation,
                                          aspect_ratio=aspect_ratio)
            camera_cache[frame_idx] = camera

        if len(all_pts):
            hits = pixel_to_world_coord(all_pts[:, 0], all_pts[:, 1], input_resolution.width,
                                        input_resolution.height, tri_mesh, camera, include_misses=True)
            world = np.array(
                [[h[0] + x_off, h[1] + y_off] if h is not None else [np.nan, np.nan] for h in hits],
                dtype=np.float64,
            )
        else:
            world = np.empty((0, 2))

        cursor = 0
        outline_world = world[cursor:cursor + len(outline_dec)]
        cursor += len(outline_dec)
        outline_lengths_geor.append(len(outline_dec))
        outline_points_geor.append(outline_world)
        n_outline_hits += int(np.isfinite(outline_world[:, 0]).sum())
        n_outline_total += len(outline_dec)

        hole_lengths_per_frame.append([len(h) for h in holes_dec])
        for h in holes_dec:
            hole_world = world[cursor:cursor + len(h)]
            cursor += len(h)
            hole_points_geor.append(hole_world)
            n_hole_hits += int(np.isfinite(hole_world[:, 0]).sum())
            n_hole_total += len(h)

    result = dict(data.items())
    result["outline_lengths_geor"] = np.array(outline_lengths_geor, dtype=np.uint64)
    result["outline_points_geor"] = (
        np.concatenate(outline_points_geor, axis=0) if outline_points_geor else np.empty((0, 2))
    )
    result["hole_counts_geor"] = _pack_hole_layout(hole_lengths_per_frame)
    result["hole_points_geor"] = (
        np.concatenate(hole_points_geor, axis=0) if hole_points_geor else np.empty((0, 2))
    )

    print(f"  {Path(path).name}: outline {n_outline_hits}/{n_outline_total}, "
          f"holes {n_hole_hits}/{n_hole_total} point projections ok "
          f"(decimated to <= {max_polygon_points} pts/polygon)")
    return result


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--npz-dir", required=True,
                   help="Folder containing the *_fschool_posture_id<N>.npz tracklets.")
    p.add_argument("--sequence", required=True,
                   help="Sequence name; selects <sequence>_fschool_posture_id<N>.npz files.")
    p.add_argument("--video", required=True,
                   help="Source video (used only to read the raw frame size for undistortion).")
    p.add_argument("--dem-glb", default=None,
                   help="Digital elevation model mesh (GLTF/GLB). Not required when "
                        "--flat-surface-msl is set.")
    p.add_argument("--dem-json", default=None,
                   help="DEM metadata json (origin offsets / CRS). Optional when "
                        "--flat-surface-msl is set.")
    p.add_argument("--poses", required=True,
                   help="Matched poses json (per-frame camera location/rotation/fovy).")
    p.add_argument("--calib", default=None,
                   help="Camera calibration json (mtx/dist), used to undistort points before "
                        "projection. Omit only if points already live in pose-frame pixel space.")
    p.add_argument("--correction", default=None,
                   help="Optional global correction json (translation/rotation).")
    p.add_argument("--mask", default=None,
                   help="Mask image; used to infer the undistorted frame resolution.")
    p.add_argument("--input-resolution", type=int, nargs=2, metavar=("W", "H"), default=None,
                   help="Override the projection input resolution (defaults to the mask size).")
    p.add_argument("--flat-surface-msl", type=float, default=None, metavar="Z_MSL",
                   help="Project onto a flat horizontal plane at this MSL elevation instead of "
                        "the DEM mesh.")
    p.add_argument("--max-polygon-points", type=int, default=150,
                   help="Decimate every outline/hole polygon to at most this many points before "
                        "ray-casting (raw contours run into the thousands of points/frame - see "
                        "fschool_posture.decimate_polygon). Default 150.")
    p.add_argument("--out-dir", required=True,
                   help="Output folder; one npz per input fish-school track is written here.")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    files = find_fschool_posture_files(args.npz_dir, args.sequence)
    if not files:
        print(f"No fschool posture tracklets found for {args.sequence} in {args.npz_dir}")
        return

    print("1. Loading DEM + poses")
    tri_mesh, offsets = load_projection_mesh(args.dem_json, args.dem_glb, args.flat_surface_msl)
    with open(args.poses, "r", encoding="utf-8") as f:
        poses = json.load(f)
    cor_rotation_eulers, cor_translation = load_correction(args.correction)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")
    raw_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    raw_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    extraction_size: Optional[Tuple[int, int]] = None
    if args.input_resolution is not None:
        extraction_size = (int(args.input_resolution[0]), int(args.input_resolution[1]))
    elif args.mask and os.path.exists(args.mask):
        _m = cv2.imread(args.mask, cv2.IMREAD_UNCHANGED)
        if _m is not None:
            extraction_size = (_m.shape[1], _m.shape[0])

    undistorter = None
    if args.calib:
        undistorter = Undistorter(args.calib, (raw_w, raw_h), new_size=extraction_size)
        print(f"   Undistorting {(raw_w, raw_h)} -> {undistorter.new_size} using {os.path.basename(args.calib)}")

    if extraction_size is not None:
        input_resolution = Resolution(extraction_size[0], extraction_size[1])
    elif undistorter is not None:
        input_resolution = undistorter.resolution
    else:
        input_resolution = Resolution(raw_w, raw_h)
    print(f"   Projection input resolution: {input_resolution.width}x{input_resolution.height}")

    print(f"2. Geo-referencing {len(files)} fish-school track(s)")
    os.makedirs(args.out_dir, exist_ok=True)
    for path in files:
        result = georeference_track(
            path, poses, tri_mesh, offsets, input_resolution,
            cor_rotation_eulers, cor_translation, undistorter, args.max_polygon_points,
        )
        out_path = os.path.join(args.out_dir, os.path.basename(path))
        np.savez(out_path, **result)
        print(f"    -> {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
