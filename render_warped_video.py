"""
Render a drone-motion-corrected ("warped"/stabilized) video: every frame is
projected onto the same flat/DEM ground plane used by trex_to_bambi.py and
composited onto a fixed geographic canvas, so the world stays anchored while
the drone pans/rotates/moves underneath it (only the camera's footprint window
moves around the canvas from frame to frame).

Unlike bambi's full orthomosaic pipeline (moderngl GPU mesh rendering, meant
for a single static mosaic), this derives one 3x3 homography per frame from
the same ray-cast pixel_to_world_coord() used elsewhere in this pipeline
(4 undistorted-frame corners -> world plane -> canvas pixels) and warps with
plain cv2.warpPerspective - no GPU/EGL context required, consistent with the
rest of TRexConnector's "no QGIS/GPU needed" approach.

Two passes:
  1. Corner pass (cheap, no video decode): for every frame in range, project
     the 4 corners of the undistorted frame onto the ground plane, to size a
     canvas that covers the whole requested range without clipping.
  2. Render pass: decode+undistort each frame, warp it into the canvas via
     the per-frame homography, and (optionally) overlay the geo-referenced
     TRex tracks (tracks.csv) and the drone's own trajectory - both already
     in the same world CRS, so no further projection is needed for them.

Example
-------
python render_warped_video.py \
    --video      ".../sequence_....mp4" \
    --dem-json   ".../dem/flat_surface_dem.json" \
    --poses      ".../frames_w/poses.json" \
    --calib      ".../blue_drone_combined.json" \
    --mask       ".../frames_w/mask_W.png" \
    --tracks-csv ".../tracks_w/tracks.csv" \
    --start-frame 5000 --end-frame 15000 \
    --output     ".../sequence_..._warped_f5000-15000.mp4" \
    --flat-surface-msl 0.0
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

cv2.setNumThreads(int(os.environ.get("OPENCV_NUM_THREADS", "4")))

from alfspy.core.rendering import Resolution
from bambi.util.projection_util import pixel_to_world_coord

from trex_to_bambi import (
    Undistorter,
    get_camera_for_frame,
    load_correction,
    load_projection_mesh,
)
from visualize_trex_video_and_map import (
    MapTileProvider,
    draw_axes_on_canvas,
    draw_drone_on_map,
    draw_map_panel,
    load_drone_positions,
    load_geo_tracks,
    make_ffmpeg_writer,
    make_global_canvas,
    pad_extent_to_match_aspect_ratio,
    world_to_canvas,
)


# --------------------------------------------------------------------------- #
# Pass 1: project the undistorted-frame corners onto the ground plane
# --------------------------------------------------------------------------- #
def project_frame_corners(camera, tri_mesh, width: int, height: int,
                          offsets: Tuple[float, float, float]) -> Optional[np.ndarray]:
    """World (x, y) of the 4 corners of a (width x height) undistorted frame,
    in the same order as ``[(0,0), (W,0), (W,H), (0,H)]``, or None if any
    corner ray misses the projection surface (e.g. a steep bank near the
    frame edge sending that ray above the horizon)."""
    x_off, y_off, _z_off = offsets
    xs = np.array([0, width, width, 0], dtype=float)
    ys = np.array([0, 0, height, height], dtype=float)
    hits = pixel_to_world_coord(xs, ys, width, height, tri_mesh, camera, include_misses=True)
    if any(h is None for h in hits):
        return None
    return np.array([[float(h[0]) + x_off, float(h[1]) + y_off] for h in hits], dtype=np.float64)


def smooth_poses(poses: dict, window: int) -> dict:
    """Return a copy of ``poses`` with a temporally-smoothed camera trajectory
    (moving average over ``window`` frames on location and rotation).

    Raw GPS/barometer altitude is typically sampled far more coarsely than
    the video frame rate and then interpolated with short, sharp ramps
    between samples (e.g. a real ~1cm/frame ramp lasting 10 frames, holding
    flat for the next 100-300 frames, then ramping back - see the altitude
    channel of a typical poses.json). Fed straight into a fresh per-frame
    homography with no smoothing, each ramp rescales the whole warped
    footprint within a fraction of a second, which reads as visible
    "breathing"/trembling even though the drone's actual flight is smooth.
    Only used for the corner-projection camera in this script - the raw
    poses are left untouched for the drone-marker overlay and for
    trex_to_bambi.py's own geo-referencing, which must stay exact.
    """
    if window <= 1:
        return poses
    images = poses["images"]
    loc = np.array([im["location"] for im in images], dtype=float)
    # Unwrap before smoothing so a rotation that crosses the 0/360 boundary
    # doesn't produce a spurious spike; get_camera_for_frame() re-wraps
    # with "% 360.0" on its own, so the unwrapped values don't need it here.
    rot = np.rad2deg(np.unwrap(np.deg2rad(np.array([im["rotation"] for im in images], dtype=float)), axis=0))

    pad = window // 2
    kernel = np.ones(window) / window

    def smooth(arr: np.ndarray) -> np.ndarray:
        out = np.empty_like(arr)
        for i in range(arr.shape[1]):
            padded = np.pad(arr[:, i], (pad, pad), mode="edge")
            out[:, i] = np.convolve(padded, kernel, mode="valid")[:len(arr)]
        return out

    loc_s, rot_s = smooth(loc), smooth(rot)
    smoothed_images = [
        {**im, "location": l.tolist(), "rotation": r.tolist()}
        for im, l, r in zip(images, loc_s, rot_s)
    ]
    return {**poses, "images": smoothed_images}


def compute_corner_world_positions(poses: dict, frame_range: range, tri_mesh, offsets,
                                   input_resolution: Resolution, cor_rotation_eulers, cor_translation,
                                   ) -> Dict[int, Optional[np.ndarray]]:
    n_poses = len(poses["images"])
    aspect_ratio = input_resolution.width / input_resolution.height
    corners_by_frame: Dict[int, Optional[np.ndarray]] = {}
    n_miss = 0
    for frame_idx in frame_range:
        if frame_idx >= n_poses:
            corners_by_frame[frame_idx] = None
            continue
        camera = get_camera_for_frame(poses, frame_idx, cor_rotation_eulers, cor_translation,
                                      aspect_ratio=aspect_ratio)
        corners = project_frame_corners(camera, tri_mesh, input_resolution.width, input_resolution.height, offsets)
        corners_by_frame[frame_idx] = corners
        if corners is None:
            n_miss += 1
    if n_miss:
        print(f"   {n_miss}/{len(frame_range)} frames could not be projected "
              f"(ray missed the surface) - they'll be left blank.")
    return corners_by_frame


# --------------------------------------------------------------------------- #
# Pass 2: warp each frame into the shared canvas
# --------------------------------------------------------------------------- #
def warp_frame_onto_canvas(undist_frame: np.ndarray, world_corners: np.ndarray,
                           canvas_cfg: dict, base: np.ndarray) -> np.ndarray:
    h, w = undist_frame.shape[:2]
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    dst = np.array([world_to_canvas(x, y, canvas_cfg) for x, y in world_corners], dtype=np.float32)

    homography = cv2.getPerspectiveTransform(src, dst)
    canvas_size = (canvas_cfg["width"], canvas_cfg["height"])
    warped = cv2.warpPerspective(undist_frame, homography, canvas_size)
    mask = cv2.warpPerspective(
        np.full((h, w), 255, dtype=np.uint8), homography, canvas_size,
    )

    out = base.copy()
    out[mask > 0] = warped[mask > 0]
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True, help="Source (raw, distorted) video file.")
    p.add_argument("--dem-glb", default=None,
                   help="Digital elevation model mesh (GLTF/GLB). Not required when "
                        "--flat-surface-msl is set.")
    p.add_argument("--dem-json", default=None,
                   help="DEM metadata json (origin offsets / CRS). Optional when "
                        "--flat-surface-msl is set (origin then defaults to (0, 0, 0)).")
    p.add_argument("--poses", required=True,
                   help="Matched poses json (per-frame camera location/rotation/fovy).")
    p.add_argument("--calib", required=True,
                   help="Camera calibration json (mtx/dist) used to undistort each frame "
                        "before warping.")
    p.add_argument("--correction", default=None,
                   help="Optional global correction json (translation/rotation).")
    p.add_argument("--mask", default=None,
                   help="Mask image; used to infer the undistorted frame resolution.")
    p.add_argument("--input-resolution", type=int, nargs=2, metavar=("W", "H"), default=None,
                   help="Override the undistorted/projection resolution (defaults to the "
                        "mask size, or a square from the raw video size).")
    p.add_argument("--flat-surface-msl", type=float, default=None, metavar="Z_MSL",
                   help="Project onto a flat horizontal plane at this MSL elevation instead "
                        "of the DEM mesh.")
    p.add_argument("--tracks-csv", default=None,
                   help="Optional geo-referenced tracks CSV (tracks_w/tracks.csv) to overlay "
                        "as boxes + trails on the warped canvas.")
    p.add_argument("--track-ids", type=int, nargs="*", default=None,
                   help="Optional subset of track ids to overlay.")
    p.add_argument("--no-drone-marker", action="store_true",
                   help="Don't draw the drone position/trail on the canvas.")
    p.add_argument("--output", required=True, help="Output video path.")
    p.add_argument("--start-frame", type=int, default=0, help="First source-video frame to render.")
    p.add_argument("--end-frame", type=int, default=None,
                   help="Stop once this source-video frame index is reached (exclusive). "
                        "Defaults to the end of the poses range.")
    p.add_argument("--canvas-size", type=int, default=1400, help="Output canvas size (square).")
    p.add_argument("--epsg", type=int, default=32643, help="EPSG code of the DEM CRS (for the satellite map).")
    p.add_argument("--no-map", action="store_true", help="Disable the satellite background.")
    p.add_argument("--map-cache", default=None, help="Directory to cache downloaded map tiles.")
    p.add_argument("--fps", type=float, default=None, help="Output FPS. Defaults to source FPS.")
    p.add_argument("--pose-smooth-window", type=int, default=25,
                   help="Moving-average window (frames) applied to the camera position/rotation "
                        "before computing each frame's homography, to damp short interpolation "
                        "ramps in the raw telemetry (e.g. coarsely-sampled altitude) that would "
                        "otherwise make the warp visibly tremble even during smooth flight. "
                        "Set to 0 or 1 to use the raw, unsmoothed poses.")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    print("1. Loading DEM + poses")
    tri_mesh, offsets = load_projection_mesh(args.dem_json, args.dem_glb, args.flat_surface_msl)
    with open(args.poses, "r", encoding="utf-8") as f:
        poses = json.load(f)
    cor_rotation_eulers, cor_translation = load_correction(args.correction)
    n_poses = len(poses["images"])

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")
    raw_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    raw_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"   Video: {args.video}  ({raw_w}x{raw_h} @ {src_fps:.1f} fps, {n_video_frames} frames)")

    extraction_size: Optional[Tuple[int, int]] = None
    if args.input_resolution is not None:
        extraction_size = (int(args.input_resolution[0]), int(args.input_resolution[1]))
    elif args.mask and os.path.exists(args.mask):
        _m = cv2.imread(args.mask, cv2.IMREAD_UNCHANGED)
        if _m is not None:
            extraction_size = (_m.shape[1], _m.shape[0])

    undistorter = Undistorter(args.calib, (raw_w, raw_h), new_size=extraction_size)
    print(f"   Undistorting {(raw_w, raw_h)} -> {undistorter.new_size} using {os.path.basename(args.calib)}")
    input_resolution = undistorter.resolution

    end_frame = args.end_frame if args.end_frame is not None else min(n_poses, n_video_frames)
    end_frame = min(end_frame, n_poses, n_video_frames)
    frame_range = range(args.start_frame, end_frame)
    if len(frame_range) == 0:
        raise ValueError(f"Empty frame range: [{args.start_frame}, {end_frame})")
    print(f"   Frame range: {args.start_frame} -> {end_frame} ({len(frame_range)} frames)")

    print("2. Projecting frame corners onto the ground plane (pass 1)")
    smoothed_poses = smooth_poses(poses, args.pose_smooth_window)
    if args.pose_smooth_window > 1:
        print(f"   Smoothing camera trajectory over a {args.pose_smooth_window}-frame window")
    corners_by_frame = compute_corner_world_positions(
        smoothed_poses, frame_range, tri_mesh, offsets, input_resolution, cor_rotation_eulers, cor_translation,
    )

    all_corners = np.concatenate([c for c in corners_by_frame.values() if c is not None], axis=0)
    extent = (
        float(all_corners[:, 0].min()), float(all_corners[:, 0].max()),
        float(all_corners[:, 1].min()), float(all_corners[:, 1].max()),
    )

    # load_geo_tracks()/load_drone_positions() return the extent over the *whole*
    # file - for a short frame range that would pad the canvas out to the entire
    # flight's bounding box while only a tiny slice of it is actually rendered, so
    # the warped footprint ends up a speck in a mostly-empty canvas. Re-derive the
    # extent from only the points that fall inside frame_range instead.
    frame_geo: Dict[int, list] = {}
    if args.tracks_csv:
        frame_geo, _ = load_geo_tracks(args.tracks_csv)
        geo_points = [
            (x, y)
            for frame_idx in frame_range
            for det in frame_geo.get(frame_idx, [])
            for x, y in ((det["gx1"], det["gy1"]), (det["gx2"], det["gy2"]))
        ]
        if geo_points:
            xs, ys = zip(*geo_points)
            extent = (
                min(extent[0], min(xs)), max(extent[1], max(xs)),
                min(extent[2], min(ys)), max(extent[3], max(ys)),
            )

    frame_drone: Dict[int, Tuple[float, float]] = {}
    if not args.no_drone_marker and args.dem_json:
        frame_drone, _ = load_drone_positions(args.poses, args.dem_json)
        drone_points = [frame_drone[frame_idx] for frame_idx in frame_range if frame_idx in frame_drone]
        if drone_points:
            xs, ys = zip(*drone_points)
            extent = (
                min(extent[0], min(xs)), max(extent[1], max(xs)),
                min(extent[2], min(ys)), max(extent[3], max(ys)),
            )

    margin = 60
    padded_extent = pad_extent_to_match_aspect_ratio(extent, args.canvas_size, args.canvas_size, margin)
    canvas_cfg = make_global_canvas(padded_extent, args.canvas_size, args.canvas_size, margin)
    print(f"   Canvas extent E[{padded_extent[0]:.1f},{padded_extent[1]:.1f}] "
          f"N[{padded_extent[2]:.1f},{padded_extent[3]:.1f}]")

    base = np.zeros((args.canvas_size, args.canvas_size, 3), dtype=np.uint8)
    if not args.no_map:
        prov = MapTileProvider(MapTileProvider.ESRI_SATELLITE, args.map_cache, utm_epsg=args.epsg)
        print("   Downloading satellite background ...")
        map_bg = prov.get_map_background(padded_extent, canvas_cfg)
        if map_bg is not None:
            base = (map_bg * 0.55).astype(np.uint8)
        else:
            print("   Could not build map background (offline?) - using blank canvas.")

    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
    track_history: Dict[int, list] = defaultdict(list)
    drone_history: list = []
    out_fps = args.fps or src_fps

    def frame_generator():
        for frame_idx in frame_range:
            ok, raw_frame = cap.read()
            if not ok:
                break

            world_corners = corners_by_frame.get(frame_idx)
            if world_corners is None:
                canvas = base.copy()
                cv2.putText(canvas, "no ground-plane projection this frame", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                undist_frame = undistorter.image(raw_frame)
                canvas = warp_frame_onto_canvas(undist_frame, world_corners, canvas_cfg, base)

            visible_tids: set = set()
            draw_map_panel(canvas, frame_geo.get(frame_idx, []), canvas_cfg,
                           track_history, visible_tids, track_ids=set(args.track_ids) if args.track_ids else None)
            if frame_drone:
                draw_drone_on_map(canvas, frame_drone.get(frame_idx), canvas_cfg, drone_history)
            draw_axes_on_canvas(canvas, canvas_cfg)
            cv2.putText(canvas, f"{Path(args.video).stem} | frame {frame_idx}", (10, canvas.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            yield frame_idx, canvas

    print("3. Warping + writing video (pass 2)")
    writer = make_ffmpeg_writer()
    if writer is not None:
        writer.write(args.output, frame_generator(), target_fps=out_fps)
    else:
        vw = None
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        for _, img in frame_generator():
            if vw is None:
                h, w = img.shape[:2]
                vw = cv2.VideoWriter(args.output, fourcc, out_fps, (w, h))
            vw.write(img)
        if vw is not None:
            vw.release()
    cap.release()
    print(f"Done: {args.output}")


if __name__ == "__main__":
    main()
