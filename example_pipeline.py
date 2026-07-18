"""
Minimal, self-contained, end-to-end example of the TRex -> BAMBI geo-referencing
pipeline, runnable against a small shareable test dataset (one ~65s/3259-frame
clip instead of a full multi-hour survey) rather than requiring the full
production data layout.

Runs, in order (each step skips if its output already exists - safe to re-run):
  1. generate_flat_surface_dem.py  - flat sea-level DEM from the flight's own GPS track
  2. extract_video_frames.py       - undistorted frames + per-frame camera poses
  3. trex_to_bambi.py              - geo-reference each TRex tracklet's *bounding box*
  4. export_georeferenced_tracklets.py - geo-reference each tracklet's *pose key-points*
     (this is the "keypoints" adaptation: step 3 alone only geo-references the 4 bbox
     corners: this augments every input npz with world-space poseX{i}_geor/poseY{i}_geor
     columns for each of its individual pose key-points)
  5. visualize_trex_video_and_map.py - side-by-side pixel-space / geo-space render

Expects --data-root laid out like this repo's test-data/pink/ folder:
    <data-root>/original-video-file/*.MP4   (exactly one video)
    <data-root>/logs/*.SRT                  (exactly one SRT, matching the video)
    <data-root>/logs/*irdata*.csv           (exactly one AirData CSV, UTC timestamps)
    <data-root>/tracking/*.npz              (one or more TRex *_id<N>.npz tracklets -
                                              this is the "multiple files" part: every
                                              tracklet in the folder is geo-referenced
                                              together in one pass, not one at a time)

Fish-school outline/hole ("segmentation") geo-referencing follows the exact same
pattern via export_georeferenced_fschool.py, but isn't included in this particular
example since this test dataset has no *_fschool_posture_id<N>.npz tracklets - point
it at --npz-dir/--sequence for a dataset that has them and it works the same way.

Two Python environments: generate_flat_surface_dem.py/extract_video_frames.py (from
the bambi_detection repo) and trex_to_bambi.py/export_georeferenced_tracklets.py (this
repo) can need *different* interpreters - the former needs whatever env has your
bambi_detection checkout importable as `bambi` (an editable/PYTHONPATH install of the
actual checkout, not just any environment with a package literally named "bambi" -
a separately pip-installed version can drift out of sync with the checked-out source
and fail with confusing TypeErrors on newer keyword arguments), and the latter needs
alfspy+bambi_detection+trimesh+pyrr (see requirements.txt), which is commonly a
separate, more self-contained env. Point --bambi-python/--trex-python at each.
Verified end-to-end (~8 min, mostly frame extraction) with:

    /path/to/bambi-capable/bin/python example_pipeline.py \
        --data-root .../test-data/pink \
        --calib     .../test-data/camera-calibration-data/angela_pink_combined.json \
        --out-dir   .../test-data/pink/example_output \
        --trex-python /path/to/trexconnector-env/bin/python
"""

import argparse
import glob
import json
import os
import subprocess
import sys

import cv2

# Respect these if already set by the caller; default to a sane cap either way -
# cv2/numpy/BLAS otherwise default to spawning up to nproc threads each, which
# compounds badly across the several subprocesses this script launches in a row.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")
os.environ.setdefault("OPENCV_NUM_THREADS", "4")

TREX_REPO = os.path.dirname(os.path.abspath(__file__))


def _one_match(pattern: str, description: str) -> str:
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No {description} found matching {pattern}")
    if len(matches) > 1:
        raise ValueError(f"Expected exactly one {description}, found {len(matches)}: {matches}")
    return matches[0]


def _find_airdata_csv(logs_dir: str) -> str:
    candidates = sorted(glob.glob(os.path.join(logs_dir, "*irdata*.csv")))
    if not candidates:
        raise FileNotFoundError(f"No AirData CSV found in {logs_dir}")
    # extract_video_frames.py expects the original, always-UTC AirData file (its own
    # docstring: "--airdata FILE (always UTC)") - prefer that over an already-localized
    # variant (e.g. "..._maldives_local.csv") that may sit alongside it in the same
    # folder, rather than erroring out on an ambiguity that isn't really ambiguous.
    utc_only = [c for c in candidates if "_local" not in os.path.basename(c).lower()]
    if len(utc_only) == 1:
        return utc_only[0]
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(f"Expected exactly one (UTC) AirData CSV, found {len(candidates)}: {candidates}")


def _poses_frame_count(poses_json_path: str) -> int:
    if not os.path.isfile(poses_json_path):
        return 0
    try:
        with open(poses_json_path, "r", encoding="utf-8") as f:
            poses = json.load(f)
    except (json.JSONDecodeError, OSError):
        return 0
    return len(poses["images"] if "images" in poses else poses)


def run(cmd, label):
    print(f"[run]  {label}")
    print("      ", " ".join(cmd))
    subprocess.run(cmd, check=True, env=os.environ)


def step1_generate_dem(bambi_python, bambi_repo, airdata_csv, out_dir):
    dem_dir = os.path.join(out_dir, "dem")
    dem_json = os.path.join(dem_dir, "flat_surface_dem.json")
    if os.path.isfile(dem_json):
        print(f"[skip] flat-surface DEM already exists: {dem_json}")
        return dem_json
    os.makedirs(dem_dir, exist_ok=True)
    run([
        bambi_python, os.path.join(bambi_repo, "src", "bambi", "generate_flat_surface_dem.py"),
        "--inputs", airdata_csv,
        "--elevation", "0.0",
        "--output", dem_dir,
        "--name", "flat_surface_dem",
    ], "Step 1: generating flat-surface DEM")
    return dem_json


def step2_extract_frames(bambi_python, bambi_repo, video, srt, airdata_csv, calib, dem_json,
                         timezone, out_dir):
    frames_dir = os.path.join(out_dir, "frames_w")
    poses_json = os.path.join(frames_dir, "poses.json")
    cap = cv2.VideoCapture(video)
    n_expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    n_actual = _poses_frame_count(poses_json)
    if n_actual and n_actual >= n_expected:
        print(f"[skip] frames already extracted ({n_actual} frames): {frames_dir}")
        return frames_dir
    if n_actual:
        print(f"[redo] incomplete extraction found ({n_actual}/{n_expected} frames) - re-running")
    run([
        bambi_python, os.path.join(bambi_repo, "src", "bambi", "extract_video_frames.py"),
        "--videos", video,
        "--srts", srt,
        "--airdata", airdata_csv,
        "--camera", "W",
        "--calibration-path", calib,
        "--dem-json", dem_json,
        "--output", frames_dir,
        "--timezone", timezone,
    ], "Step 2: extracting frames + poses")
    return frames_dir


def step3_georeference_bboxes(trex_python, npz_dir, dem_json, poses_json, calib, mask, out_dir):
    tracks_csv = os.path.join(out_dir, "tracks_w", "tracks.csv")
    if os.path.isfile(tracks_csv):
        print(f"[skip] already geo-referenced (bounding boxes): {tracks_csv}")
        return tracks_csv
    run([
        trex_python, os.path.join(TREX_REPO, "trex_to_bambi.py"),
        "--npz-dir", npz_dir,
        "--dem-json", dem_json,
        "--poses", poses_json,
        "--calib", calib,
        "--mask", mask,
        "--out-dir", out_dir,
        "--flat-surface-msl", "0.0",
    ], "Step 3: geo-referencing tracklets (bounding boxes)")
    return tracks_csv


def step3b_georeference_keypoints(trex_python, npz_dir, dem_json, poses_json, calib, mask, out_dir):
    tracklets_out = os.path.join(out_dir, "tracklets_georeferenced_w")
    wanted = {os.path.basename(p) for p in glob.glob(os.path.join(npz_dir, "*.npz"))}
    if os.path.isdir(tracklets_out):
        existing = {f for f in os.listdir(tracklets_out) if f.endswith(".npz")}
        if wanted and wanted.issubset(existing):
            print(f"[skip] geo-referenced key-points already exported: {tracklets_out}")
            return tracklets_out
    run([
        trex_python, os.path.join(TREX_REPO, "export_georeferenced_tracklets.py"),
        "--npz-dir", npz_dir,
        "--dem-json", dem_json,
        "--poses", poses_json,
        "--calib", calib,
        "--mask", mask,
        "--out-dir", tracklets_out,
        "--flat-surface-msl", "0.0",
    ], f"Step 3b: geo-referencing {len(wanted)} tracklet(s) per key-point")
    return tracklets_out


def step4_render_vis(video, npz_dir, tracks_csv, poses_json, dem_json, out_path):
    if os.path.isfile(out_path):
        print(f"[skip] vis video already exists: {out_path}")
        return out_path
    run([
        sys.executable, os.path.join(TREX_REPO, "visualize_trex_video_and_map.py"),
        "--video", video,
        "--tracking-dir", npz_dir,
        "--tracks-csv", tracks_csv,
        "--poses", poses_json,
        "--dem-json", dem_json,
        "--output", out_path,
        "--no-live",
    ], "Step 4: rendering side-by-side vis video")
    return out_path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", required=True,
                   help="Folder shaped like test-data/pink/ (original-video-file/, logs/, tracking/).")
    p.add_argument("--calib", required=True, help="Camera calibration JSON (mtx/dist).")
    p.add_argument("--out-dir", required=True, help="Output folder for every step below.")
    p.add_argument("--bambi-repo", default=os.path.join(os.path.dirname(TREX_REPO), "bambi_detection"),
                   help="Path to a bambi_detection checkout (default: sibling of this repo).")
    p.add_argument("--bambi-python", default=sys.executable,
                   help="Python interpreter for generate_flat_surface_dem.py/extract_video_frames.py "
                        "(default: the interpreter running this script).")
    p.add_argument("--trex-python", default=sys.executable,
                   help="Python interpreter for trex_to_bambi.py/export_georeferenced_tracklets.py - "
                        "these need alfspy+bambi_detection+trimesh+pyrr installed (see requirements.txt). "
                        "Point this at a dedicated env if that's not the interpreter running this script.")
    p.add_argument("--timezone", default="Indian/Maldives",
                   help="IANA timezone the flight was recorded in (default matches this test dataset).")
    args = p.parse_args()

    video = _one_match(os.path.join(args.data_root, "original-video-file", "*.MP4"), "video")
    srt = _one_match(os.path.join(args.data_root, "logs", "*.SRT"), "SRT file")
    airdata_csv = _find_airdata_csv(os.path.join(args.data_root, "logs"))
    npz_dir = os.path.join(args.data_root, "tracking")
    n_tracklets = len(glob.glob(os.path.join(npz_dir, "*.npz")))
    if n_tracklets == 0:
        raise FileNotFoundError(f"No .npz tracklets found in {npz_dir}")

    print(f"Video      : {video}")
    print(f"SRT        : {srt}")
    print(f"AirData    : {airdata_csv}")
    print(f"Tracklets  : {n_tracklets} file(s) in {npz_dir}")
    print(f"Calibration: {args.calib}")
    print()

    os.makedirs(args.out_dir, exist_ok=True)

    dem_json = step1_generate_dem(args.bambi_python, args.bambi_repo, airdata_csv, args.out_dir)
    frames_dir = step2_extract_frames(
        args.bambi_python, args.bambi_repo, video, srt, airdata_csv, args.calib, dem_json,
        args.timezone, args.out_dir,
    )
    poses_json = os.path.join(frames_dir, "poses.json")
    mask = os.path.join(frames_dir, "mask_W.png")

    tracks_csv = step3_georeference_bboxes(
        args.trex_python, npz_dir, dem_json, poses_json, args.calib, mask, args.out_dir,
    )
    step3b_georeference_keypoints(
        args.trex_python, npz_dir, dem_json, poses_json, args.calib, mask, args.out_dir,
    )
    vis_path = os.path.join(args.out_dir, f"{os.path.splitext(os.path.basename(video))[0]}_trex_vis.mp4")
    step4_render_vis(video, npz_dir, tracks_csv, poses_json, dem_json, vis_path)

    print()
    print("Done. Outputs:")
    print(f"  DEM                   : {dem_json}")
    print(f"  Frames + poses        : {frames_dir}")
    print(f"  Geo-referenced tracks : {tracks_csv}")
    print(f"  Geo-referenced key-points : {os.path.join(args.out_dir, 'tracklets_georeferenced_w')}")
    print(f"  Vis video             : {vis_path}")


if __name__ == "__main__":
    main()
