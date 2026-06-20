"""
Read a raw-pixel tracks_pixel.csv (5120x2700 space) and write an undistorted
version in the new 2700x2700 pixel space, using the same calibration +
undistortion parameters as trex_to_bambi.py / BAMBI's frame extractor.

Usage:
  python undistort_pixel_tracks.py \
      --calib angela_pink_combined.json \
      --input tracks_pixel.csv \
      --output tracks_pixel_undistorted.csv \
      --raw-size 5120 2700
"""
import argparse
import json
import csv
import sys
import numpy as np
import cv2


def load_calib(path: str):
    with open(path) as f:
        d = json.load(f)
    # BAMBI calibration files use "mtx"/"dist"; fall back to OpenCV FileStorage names
    mtx_raw = d.get("mtx") or d.get("camera_matrix")
    dist_raw = d.get("dist") or d.get("dist_coefs")
    if mtx_raw is None or dist_raw is None:
        raise KeyError(f"Calibration file must contain 'mtx'+'dist' or 'camera_matrix'+'dist_coefs' keys: {path}")
    mtx = np.array(mtx_raw, dtype=np.float64)
    dist = np.array(dist_raw, dtype=np.float64).flatten()
    return mtx, dist


def build_new_camera_matrix(mtx, dist, raw_w, raw_h,
                            alpha=0.5, center_principal_point=True,
                            force_same_fov=True):
    min_side = min(raw_w, raw_h)
    new_size = (min_side, min_side)
    ncm, _ = cv2.getOptimalNewCameraMatrix(
        mtx, dist, (raw_w, raw_h), alpha, new_size,
        centerPrincipalPoint=center_principal_point,
    )
    if force_same_fov:
        fxy = max(ncm[0, 0], ncm[1, 1])
        ncm[0, 0] = fxy
        ncm[1, 1] = fxy
    return ncm, new_size


def undistort_corners(corners_raw, mtx, dist, ncm):
    """corners_raw: (N,2) float32 in raw pixel space → (N,2) float32 in undistorted space."""
    pts = corners_raw.reshape(-1, 1, 2).astype(np.float32)
    out = cv2.undistortPoints(pts, mtx, dist, P=ncm)
    return out.reshape(-1, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--raw-size", nargs=2, type=int, default=[5120, 2700],
                    metavar=("WIDTH", "HEIGHT"))
    args = ap.parse_args()

    mtx, dist = load_calib(args.calib)
    raw_w, raw_h = args.raw_size
    ncm, new_size = build_new_camera_matrix(mtx, dist, raw_w, raw_h)
    new_w, new_h = new_size
    print(f"Raw size:         {raw_w}×{raw_h}")
    print(f"Undistorted size: {new_w}×{new_h}")
    print(f"new_camera_matrix fxy={ncm[0,0]:.3f}  cx={ncm[0,2]:.3f}  cy={ncm[1,2]:.3f}")

    rows = []
    with open(args.input, newline="") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cols = line.split(",")
            if len(cols) < 7:
                continue
            frame_id = cols[0]
            track_id = cols[1]
            x1, y1, x2, y2 = float(cols[2]), float(cols[3]), float(cols[4]), float(cols[5])
            conf = cols[6]
            rest = cols[7:]
            rows.append((frame_id, track_id, x1, y1, x2, y2, conf, rest))

    if not rows:
        print("No rows found — check CSV format.", file=sys.stderr)
        sys.exit(1)

    # Undistort all corner points in one batched call
    raw_pts = np.array([[r[2], r[3]] for r in rows] + [[r[4], r[5]] for r in rows],
                       dtype=np.float32)
    und_pts = undistort_corners(raw_pts, mtx, dist, ncm)
    n = len(rows)
    tl_pts = und_pts[:n]
    br_pts = und_pts[n:]

    print(f"\nUndistortion stats (x):")
    raw_x1 = raw_pts[:n, 0]
    und_x1 = tl_pts[:, 0]
    print(f"  raw  x1  min={raw_x1.min():.1f}  max={raw_x1.max():.1f}")
    print(f"  und  x1  min={und_x1.min():.1f}  max={und_x1.max():.1f}")
    print(f"  shift x1 mean={np.mean(und_x1 - raw_x1):.1f}  std={np.std(und_x1 - raw_x1):.1f}")

    out_rows = []
    for i, (frame_id, track_id, _, _, _, _, conf, rest) in enumerate(rows):
        ux1, uy1 = tl_pts[i]
        ux2, uy2 = br_pts[i]
        out_rows.append([frame_id, track_id,
                         f"{ux1:.6f}", f"{uy1:.6f}",
                         f"{ux2:.6f}", f"{uy2:.6f}",
                         conf] + rest)

    with open(args.output, "w", newline="") as f:
        for row in out_rows:
            f.write(",".join(row) + "\n")

    print(f"\nWrote {len(out_rows)} rows → {args.output}")


if __name__ == "__main__":
    main()
