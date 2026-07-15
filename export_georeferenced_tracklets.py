"""
Export per-individual TRex tracklets augmented with geo-referenced pose key-points.

Reads each ``*_id<N>.npz`` tracklet exactly as trex_to_bambi.py does, keeps every
original field untouched (frame, timestamp, detection_p, id, ...), and adds one
``poseX{i}_geor`` / ``poseY{i}_geor`` pair per existing ``poseX{i}``/``poseY{i}``
key-point: the world-space (DEM CRS) x/y of that key-point, projected through the
matched drone pose for its frame, using the same undistortion + DEM/flat-surface
projection pipeline as trex_to_bambi.py. Key-points that could not be projected
(frame beyond the poses range, or the ray missed the projection surface) are left
as NaN in the ``*_geor`` columns.

Output: one npz per input tracklet, same filename, written to --out-dir.

Example
-------
python export_georeferenced_tracklets.py \
    --npz-dir   ".../npz_input" \
    --dem-json  ".../dem/flat_surface_dem.json" \
    --poses     ".../frames_w/poses.json" \
    --calib     ".../blue_drone_combined.json" \
    --mask      ".../frames_w/mask_W.png" \
    --out-dir   ".../tracklets_georeferenced_w" \
    --flat-surface-msl 0.0
"""

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from alfspy.core.rendering import Resolution
from bambi.util.projection_util import pixel_to_world_coord

from trex_to_bambi import (
    Undistorter,
    _pose_keys,
    get_camera_for_frame,
    load_correction,
    load_projection_mesh,
)


# --------------------------------------------------------------------------- #
# Per-tracklet key-point geo-referencing
# --------------------------------------------------------------------------- #
def georeference_tracklet(
    data,
    pidx: List[int],
    poses: dict,
    tri_mesh,
    offsets: Tuple[float, float, float],
    input_resolution: Resolution,
    cor_rotation_eulers,
    cor_translation,
    undistorter: Optional[Undistorter],
) -> Dict[str, np.ndarray]:
    """Compute poseX{i}_geor / poseY{i}_geor arrays for one tracklet's data."""
    x_off, y_off, _z_off = offsets
    n_poses = len(poses["images"])
    frames = np.asarray(data["frame"]).astype(int)
    n = len(frames)

    pose_x = {i: np.asarray(data[f"poseX{i}"], dtype=float) for i in pidx}
    pose_y = {i: np.asarray(data[f"poseY{i}"], dtype=float) for i in pidx}

    geor: Dict[str, np.ndarray] = {}
    for i in pidx:
        geor[f"poseX{i}_geor"] = np.full(n, np.nan, dtype=np.float64)
        geor[f"poseY{i}_geor"] = np.full(n, np.nan, dtype=np.float64)

    camera_cache: Dict[int, object] = {}

    for row in range(n):
        frame_idx = int(frames[row])
        if frame_idx >= n_poses:
            continue

        valid_idx = [i for i in pidx if np.isfinite(pose_x[i][row]) and np.isfinite(pose_y[i][row])]
        if not valid_idx:
            continue

        pts = np.array([[pose_x[i][row], pose_y[i][row]] for i in valid_idx], dtype=np.float32)
        if undistorter is not None:
            pts = undistorter.points(pts)

        camera = camera_cache.get(frame_idx)
        if camera is None:
            camera = get_camera_for_frame(poses, frame_idx, cor_rotation_eulers, cor_translation,
                                          aspect_ratio=input_resolution.width / input_resolution.height)
            camera_cache[frame_idx] = camera

        # include_misses=True keeps the output index-aligned with valid_idx, so a
        # missed ray for one key-point doesn't shift the others (unlike
        # trex_to_bambi.label_to_world_coordinates, which drops misses).
        hits = pixel_to_world_coord(pts[:, 0], pts[:, 1], input_resolution.width, input_resolution.height,
                                    tri_mesh, camera, include_misses=True)

        for i, hit in zip(valid_idx, hits):
            if hit is None:
                continue
            geor[f"poseX{i}_geor"][row] = float(hit[0]) + x_off
            geor[f"poseY{i}_geor"][row] = float(hit[1]) + y_off

    return geor


def process_tracklet(
    path: str, out_dir: str, poses: dict, tri_mesh, offsets, input_resolution,
    cor_rotation_eulers, cor_translation, undistorter: Optional[Undistorter],
) -> str:
    data = np.load(path, allow_pickle=True)
    out_path = os.path.join(out_dir, os.path.basename(path))

    pidx = _pose_keys(data)
    if not pidx:
        print(f"  WARNING: {Path(path).name} has no pose points, skipped")
        return out_path

    geor = georeference_tracklet(
        data, pidx, poses, tri_mesh, offsets, input_resolution,
        cor_rotation_eulers, cor_translation, undistorter,
    )

    merged = {k: data[k] for k in data.keys()}
    merged.update(geor)
    os.makedirs(out_dir, exist_ok=True)
    np.savez(out_path, **merged)

    n_hits = sum(int(np.isfinite(geor[f"poseX{i}_geor"]).sum()) for i in pidx)
    n_total = sum(len(geor[f"poseX{i}_geor"]) for i in pidx)
    print(f"  {Path(path).name}: {n_hits}/{n_total} key-point projections ok -> {out_path}")
    return out_path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--npz-dir", required=True,
                   help="Folder containing the TRex *.npz tracklets to augment.")
    p.add_argument("--dem-glb", default=None,
                   help="Digital elevation model mesh (GLTF/GLB). Not required when "
                        "--flat-surface-msl is set.")
    p.add_argument("--dem-json", default=None,
                   help="DEM metadata json (origin offsets / CRS). Optional when "
                        "--flat-surface-msl is set (origin then defaults to (0, 0, 0), "
                        "i.e. coordinates stay in the local pose frame).")
    p.add_argument("--poses", required=True,
                   help="Matched poses json (per-frame camera location/rotation/fovy).")
    p.add_argument("--calib", default=None,
                   help="Camera calibration json (mtx/dist), used to undistort key-points "
                        "before projection. Omit only if the npz key-points already live "
                        "in the pose-frame (undistorted) pixel space.")
    p.add_argument("--correction", default=None,
                   help="Optional global correction json (translation/rotation).")
    p.add_argument("--mask", default=None,
                   help="Mask image; used to infer the undistorted frame resolution.")
    p.add_argument("--out-dir", required=True,
                   help="Output folder; one npz per input tracklet is written here.")
    p.add_argument("--input-resolution", type=int, nargs=2, metavar=("W", "H"), default=None,
                   help="Override the projection input resolution (defaults to the "
                        "undistorted square size, or the mask size).")
    p.add_argument("--flat-surface-msl", type=float, default=None, metavar="Z_MSL",
                   help="Project key-points onto a flat horizontal plane at this MSL "
                        "elevation instead of the DEM mesh.")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    files = sorted(glob.glob(os.path.join(args.npz_dir, "*.npz")))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {args.npz_dir}")

    video_size: Optional[Tuple[int, int]] = None
    for path in files:
        data = np.load(path, allow_pickle=True)
        if "video_size" in data:
            vs = data["video_size"]
            video_size = (int(vs[0]), int(vs[1]))
            break

    print("1. Loading DEM + poses")
    tri_mesh, offsets = load_projection_mesh(args.dem_json, args.dem_glb, args.flat_surface_msl)

    with open(args.poses, "r", encoding="utf-8") as f:
        poses = json.load(f)

    cor_rotation_eulers, cor_translation = load_correction(args.correction)

    extraction_size: Optional[Tuple[int, int]] = None
    if args.input_resolution is not None:
        extraction_size = (int(args.input_resolution[0]), int(args.input_resolution[1]))
    elif args.mask and os.path.exists(args.mask):
        _m = cv2.imread(args.mask, cv2.IMREAD_UNCHANGED)
        if _m is not None:
            extraction_size = (_m.shape[1], _m.shape[0])

    undistorter = None
    if args.calib:
        if video_size is None:
            raise ValueError("Raw video size unknown (no npz has 'video_size'); cannot "
                              "undistort. Omit --calib if key-points are already in the "
                              "pose-frame pixel space.")
        undistorter = Undistorter(args.calib, video_size, new_size=extraction_size)
        print(f"   Undistorting {video_size} -> {undistorter.new_size} using {os.path.basename(args.calib)}")

    if extraction_size is not None:
        input_resolution = Resolution(extraction_size[0], extraction_size[1])
    elif undistorter is not None:
        input_resolution = undistorter.resolution
    elif video_size is not None:
        input_resolution = Resolution(video_size[0], video_size[1])
    else:
        raise ValueError("Could not determine the projection input resolution: pass "
                          "--input-resolution or --mask, or ensure the npz files have "
                          "'video_size'.")
    print(f"   Projection input resolution: {input_resolution.width}x{input_resolution.height}")

    print("2. Geo-referencing key-points per tracklet")
    for path in files:
        process_tracklet(
            path, args.out_dir, poses, tri_mesh, offsets, input_resolution,
            cor_rotation_eulers, cor_translation, undistorter,
        )
    print("Done.")


if __name__ == "__main__":
    main()
