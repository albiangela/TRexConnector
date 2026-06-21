"""
Visualize TRex tracklets on the source video (pixel space) next to a map of the
associated geo-referenced tracks.

This is a companion tool to ``trex_to_bambi.py``: it overlays the raw TRex
tracklets on the original video and, side by side, plots the geo-referenced
``tracks.csv`` that ``trex_to_bambi.py`` produces, so a TRex run can be checked
visually end to end.

Left panel : the original video with per-track bounding boxes and ID labels
             drawn in pixel space. The boxes come either from the TRex pose
             key-points (``--tracking-dir``) or from a pixel-space MOT CSV
             (``--pixel-tracks-csv``); exactly one of the two is used.
Right panel: a 2D map of the geo-referenced tracks (projected coordinates from
             ``tracks.csv``) with per-track boxes, trajectory trails and an
             optional satellite background. When a poses JSON (``--poses``) and
             its DEM metadata (``--dem-json``) are supplied, the drone's own
             position and flight trail are drawn on the map too.

The TRex tracking output is one ``*_id<N>.npz`` file per track. Each file holds,
per video frame, the centroid, the 9 pose key-points (``poseX0..8`` /
``poseY0..8``) and the detection confidence/class. Alternatively, the pixel-space
MOT CSV (``tracks_pixel.csv`` from ``trex_to_bambi.py``) carries the raw
(distorted) bounding boxes directly, without key-points. Either way the frame
indices and track ids match the geo-referenced tracks, so the panels stay in sync.

Output is encoded with BAMBI's ``PipeFFMPEGWriter`` (libx264) when available,
falling back to ``cv2.VideoWriter`` (mp4v) otherwise — so the only hard runtime
dependencies are numpy, opencv, and (for the satellite map) requests + pyproj.

Example:
    python visualize_trex_video_and_map.py \
        --video        /path/to/20240307_063012765_DJI_0463.MP4 \
        --tracking-dir /path/to/tracking \
        --tracks-csv   /path/to/output/tracks_w/tracks.csv \
        --epsg 32643

    # or drive the video panel from a pixel-space MOT CSV instead:
    python visualize_trex_video_and_map.py \
        --video            /path/to/20240307_063012765_DJI_0463.MP4 \
        --pixel-tracks-csv /path/to/output/tracks_pixel_w/tracks_pixel.csv \
        --tracks-csv       /path/to/output/tracks_w/tracks.csv \
        --epsg 32643

    or

      python visualize_trex_video_and_map.py
      --video            "C:/Users/P41743/Desktop/lndf/Angela/test-data/pink/qgis5/poses_w.mp4"
      --pixel-tracks-csv "C:/Users/P41743/Desktop/lndf/Angela/test-data/pink/trex14/tracks_pixel_w/tracks_pixel_undistorted.csv"
      --tracks-csv       "C:/Users/P41743/Desktop/lndf/Angela/test-data/pink/trex14/tracks_w/tracks.csv"
      --poses            "C:/Users/P41743/Desktop/lndf/Angela/test-data/pink/qgis5/poses_w.json"
      --dem-json         "C:/Users/P41743/Desktop/lndf/Angela/test-data/pink/qgis5/flat_surface_dem.json"
      --output           "C:/Users/P41743/Desktop/lndf/Angela/test-data/pink/qgis5/poses_w_vis.mp4"
      --epsg 32643
"""

import argparse
import colorsys
import hashlib
import math
import os
from collections import defaultdict
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import json
import numpy as np

try:
    import requests
    from pyproj import CRS, Transformer
    _HAS_MAP_DEPS = True
except Exception:  # pragma: no cover - map background is optional
    _HAS_MAP_DEPS = False


# ============================================================
# UNDISTORTION (raw video pixels -> undistorted frame pixels)
# ============================================================

class Undistorter:
    """
    Mirrors the BAMBI CalibratedVideoFrameAccessor undistortion so that
    raw-video pixel coordinates can be mapped into the undistorted frame
    space used by the exported frames / poses.

    Pass an instance to undistort_frame_dets() after loading pixel tracks.
    """

    def __init__(self, calib_path: str, raw_size: Tuple[int, int],
                 alpha: float = 0.5, center_principal_point: bool = True,
                 force_same_fov: bool = True):
        with open(calib_path, "r", encoding="utf-8") as f:
            calib = json.load(f)
        mtx_raw = calib.get("mtx") or calib.get("camera_matrix")
        dist_raw = calib.get("dist") or calib.get("dist_coefs")
        if mtx_raw is None or dist_raw is None:
            raise KeyError(
                f"Calibration file must contain 'mtx'+'dist' or "
                f"'camera_matrix'+'dist_coefs' keys: {calib_path}"
            )
        self.mtx = np.asarray(mtx_raw, dtype=float)
        self.dist = np.asarray(dist_raw, dtype=float)

        w, h = raw_size
        wh = min(w, h)
        self.new_size = (wh, wh)
        ncm, _ = cv2.getOptimalNewCameraMatrix(
            self.mtx, self.dist, (w, h), alpha, self.new_size,
            centerPrincipalPoint=center_principal_point,
        )
        if force_same_fov:
            fxy = max(ncm[0, 0], ncm[1, 1])
            ncm[0, 0] = ncm[1, 1] = fxy
        self.new_camera_matrix = ncm
        self.raw_size = raw_size

    def points(self, pts_xy: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts_xy, dtype=np.float32).reshape(-1, 1, 2)
        out = cv2.undistortPoints(pts, self.mtx, self.dist, P=self.new_camera_matrix)
        return out.reshape(-1, 2)


def undistort_frame_dets(frame_dets: Dict[int, List[dict]], undistorter: Undistorter) -> None:
    """Undistort all pixel-space bounding boxes and key-points in-place."""
    for dets in frame_dets.values():
        for det in dets:
            corners = np.array([[det["x1"], det["y1"]], [det["x2"], det["y2"]]], dtype=np.float32)
            und = undistorter.points(corners)
            det["x1"], det["y1"] = float(und[0, 0]), float(und[0, 1])
            det["x2"], det["y2"] = float(und[1, 0]), float(und[1, 1])
            det["cx"] = (det["x1"] + det["x2"]) / 2.0
            det["cy"] = (det["y1"] + det["y2"]) / 2.0
            if det.get("keypoints"):
                kp = np.array(det["keypoints"], dtype=np.float32)
                und_kp = undistorter.points(kp)
                det["keypoints"] = list(zip(und_kp[:, 0].tolist(), und_kp[:, 1].tolist()))


# ============================================================
# 0. OPTIONAL BAMBI VIDEO WRITER
# ============================================================

def make_ffmpeg_writer():
    """
    Return a BAMBI ``PipeFFMPEGWriter`` instance, or ``None`` if it (or ffmpeg)
    is unavailable. Constructing the writer probes ``ffmpeg -version``.
    """
    try:
        from bambi.video.video_writer import PipeFFMPEGWriter
        return PipeFFMPEGWriter(silent=True)
    except Exception as e:
        print(f"FFMPEG writer unavailable ({e}); using cv2.VideoWriter fallback.")
        return None


# ============================================================
# 1. DRAWING / COORDINATE HELPERS
# ============================================================

def id_to_color(identifier, saturation=0.65, lightness=0.5):
    """Deterministic BGR colour for a given track id."""
    h = hashlib.sha256(str(identifier).encode("utf-8")).digest()
    hue = int.from_bytes(h[:4], "big") / 2 ** 32
    r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
    return (int(b * 255), int(g * 255), int(r * 255))


def draw_dashed_rectangle(img, pt1, pt2, color, thickness=2, dash_length=10):
    x1, y1 = int(pt1[0]), int(pt1[1])
    x2, y2 = int(pt2[0]), int(pt2[1])
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1

    for x in range(x1, x2, dash_length * 2):
        cv2.line(img, (x, y1), (min(x + dash_length, x2), y1), color, thickness)
        cv2.line(img, (x, y2), (min(x + dash_length, x2), y2), color, thickness)
    for y in range(y1, y2, dash_length * 2):
        cv2.line(img, (x1, y), (x1, min(y + dash_length, y2)), color, thickness)
        cv2.line(img, (x2, y), (x2, min(y + dash_length, y2)), color, thickness)


def pad_extent_to_match_aspect_ratio(extent, width, height, margin):
    min_x, max_x, min_y, max_y = extent
    draw_w = width - 2 * margin
    draw_h = height - 2 * margin
    target_ar = draw_w / draw_h

    data_w = max_x - min_x
    data_h = max_y - min_y
    data_ar = data_w / data_h if data_h > 0 else 1.0

    cx, cy = (min_x + max_x) / 2, (min_y + max_y) / 2

    if data_ar > target_ar:
        new_h = data_w / target_ar
        min_y = cy - new_h / 2
        max_y = cy + new_h / 2
    else:
        new_w = data_h * target_ar
        min_x = cx - new_w / 2
        max_x = cx + new_w / 2

    return (min_x, max_x, min_y, max_y)


def make_global_canvas(global_extent, width=800, height=800, margin=60):
    min_x, max_x, min_y, max_y = global_extent
    span_x = max(max_x - min_x, 1e-6)
    scale = (width - 2 * margin) / span_x
    return {
        "min_x": min_x, "max_x": max_x, "min_y": min_y, "max_y": max_y,
        "scale": scale, "margin": margin, "width": width, "height": height,
    }


def world_to_canvas(x, y, canvas_cfg):
    min_x = canvas_cfg["min_x"]
    min_y = canvas_cfg["min_y"]
    scale = canvas_cfg["scale"]
    margin = canvas_cfg["margin"]
    height = canvas_cfg["height"]
    px = int(margin + (x - min_x) * scale)
    py = int(height - (margin + (y - min_y) * scale))
    return px, py


def draw_axes_on_canvas(map_img, canvas_cfg, num_ticks=4):
    min_x, max_x = canvas_cfg["min_x"], canvas_cfg["max_x"]
    min_y, max_y = canvas_cfg["min_y"], canvas_cfg["max_y"]
    axis_color = (200, 200, 200)
    text_color = (255, 255, 255)

    bl = world_to_canvas(min_x, min_y, canvas_cfg)
    br = world_to_canvas(max_x, min_y, canvas_cfg)
    tl = world_to_canvas(min_x, max_y, canvas_cfg)

    cv2.line(map_img, bl, br, axis_color, 1)
    cv2.line(map_img, bl, tl, axis_color, 1)

    for i in range(num_ticks):
        t = i / (num_ticks - 1)
        val = min_x + t * (max_x - min_x)
        px, py = world_to_canvas(val, min_y, canvas_cfg)
        cv2.line(map_img, (px, py), (px, py + 5), axis_color, 1)
        cv2.putText(map_img, f"{int(val)}", (px - 20, py + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1, cv2.LINE_AA)

    for i in range(num_ticks):
        t = i / (num_ticks - 1)
        val = min_y + t * (max_y - min_y)
        px, py = world_to_canvas(min_x, val, canvas_cfg)
        cv2.line(map_img, (px - 5, py), (px, py), axis_color, 1)
        cv2.putText(map_img, f"{int(val)}", (px - 60, py + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1, cv2.LINE_AA)

    h, w = map_img.shape[:2]
    cv2.putText(map_img, "Easting (X)", (w // 2, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    cv2.putText(map_img, "Northing (Y)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)


# ============================================================
# 2. MAP TILE PROVIDER (optional satellite background)
# ============================================================

class MapTileProvider:
    OPENSTREETMAP = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
    ESRI_SATELLITE = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    CARTO_LIGHT = "https://basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png"
    CARTO_DARK = "https://basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png"

    def __init__(self, tile_url=None, cache_dir=None, utm_epsg=32643):
        self.tile_url = tile_url or self.OPENSTREETMAP
        self.cache_dir = cache_dir
        self.transformer = Transformer.from_crs(CRS.from_epsg(utm_epsg), CRS.from_epsg(4326), always_xy=True)
        self.headers = {'User-Agent': 'VisScript/1.0'}
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def utm_to_latlon(self, x, y):
        return self.transformer.transform(x, y)[::-1]

    def latlon_to_tile(self, lat, lon, zoom):
        lat_rad = math.radians(lat)
        n = 2.0 ** zoom
        x_tile = int((lon + 180.0) / 360.0 * n)
        y_tile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        return x_tile, y_tile

    def tile_to_latlon(self, x, y, zoom):
        n = 2.0 ** zoom
        lon = x / n * 360.0 - 180.0
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
        return math.degrees(lat_rad), lon

    def download_tile(self, x, y, zoom):
        cache_path = None
        if self.cache_dir:
            h = hashlib.md5(self.tile_url.encode()).hexdigest()[:8]
            cache_path = os.path.join(self.cache_dir, f"{h}_{zoom}_{x}_{y}.png")
            if os.path.exists(cache_path):
                return cv2.imread(cache_path)

        url = self.tile_url.format(z=zoom, x=x, y=y)
        try:
            resp = requests.get(url, headers=self.headers, timeout=5)
            if resp.status_code == 200:
                arr = np.frombuffer(resp.content, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if self.cache_dir and img is not None:
                    cv2.imwrite(cache_path, img)
                return img
        except Exception:
            pass
        return None

    def get_map_background(self, global_extent, canvas_cfg):
        min_x, max_x, min_y, max_y = global_extent
        min_lat, min_lon = self.utm_to_latlon(min_x, min_y)
        max_lat, max_lon = self.utm_to_latlon(max_x, max_y)
        if min_lat > max_lat:
            min_lat, max_lat = max_lat, min_lat
        if min_lon > max_lon:
            min_lon, max_lon = max_lon, min_lon

        cw, ch = canvas_cfg["width"], canvas_cfg["height"]
        zoom = 18
        for z in range(19, 12, -1):
            x1, y1 = self.latlon_to_tile(max_lat, min_lon, z)
            x2, y2 = self.latlon_to_tile(min_lat, max_lon, z)
            if (abs(x2 - x1) + 1) * 256 > cw and (abs(y2 - y1) + 1) * 256 > ch:
                zoom = z
                break

        tx1, ty1 = self.latlon_to_tile(max_lat, min_lon, zoom)
        tx2, ty2 = self.latlon_to_tile(min_lat, max_lon, zoom)

        stitch_w = (tx2 - tx1 + 1) * 256
        stitch_h = (ty2 - ty1 + 1) * 256
        stitch = np.zeros((stitch_h, stitch_w, 3), dtype=np.uint8)

        for y in range(ty1, ty2 + 1):
            for x in range(tx1, tx2 + 1):
                t = self.download_tile(x, y, zoom)
                if t is not None:
                    py, px = (y - ty1) * 256, (x - tx1) * 256
                    stitch[py:py + 256, px:px + 256] = t

        top_lat, left_lon = self.tile_to_latlon(tx1, ty1, zoom)
        btm_lat, rgt_lon = self.tile_to_latlon(tx2 + 1, ty2 + 1, zoom)

        def ll2px(lat, lon):
            px = (lon - left_lon) / (rgt_lon - left_lon) * stitch_w
            py = (top_lat - lat) / (top_lat - btm_lat) * stitch_h
            return int(px), int(py)

        px1, py1 = ll2px(max_lat, min_lon)
        px2, py2 = ll2px(min_lat, max_lon)

        px1, py1 = max(0, px1), max(0, py1)
        px2, py2 = min(stitch_w, px2), min(stitch_h, py2)

        if px2 <= px1 or py2 <= py1:
            return None
        crop = stitch[py1:py2, px1:px2]

        iw, ih = cw - 2 * canvas_cfg["margin"], ch - 2 * canvas_cfg["margin"]
        final = np.zeros((ch, cw, 3), dtype=np.uint8)
        try:
            resized_crop = cv2.resize(crop, (iw, ih), interpolation=cv2.INTER_AREA)
            final[canvas_cfg["margin"]:canvas_cfg["margin"] + ih,
                  canvas_cfg["margin"]:canvas_cfg["margin"] + iw] = resized_crop
        except Exception as e:
            print(f"Map resize error: {e}")
            return None

        return final


# ============================================================
# 3. DATA LOADING
# ============================================================

def load_trex_tracklets(tracking_dir: str, video_stem: str,
                        video_w: int, video_h: int,
                        ) -> Tuple[Dict[int, List[dict]], int, Optional[Tuple[int, int]]]:
    """
    Loads all ``<video_stem>_id<N>.npz`` TRex tracklets.

    A pixel-space bounding box is derived from the extent of the valid pose
    key-points (``poseX0..8`` / ``poseY0..8``). Invalid key-points are encoded
    by TRex as ``inf`` or ``0`` and are filtered out.

    :return: (frame_dets, n_tracks, raw_video_size) where ``raw_video_size`` is
             the raw video dimensions read from the npz (or None if not stored).
    """
    files = sorted(glob(os.path.join(tracking_dir, f"{video_stem}_id*.npz")))
    if not files:
        raise FileNotFoundError(
            f"No TRex tracklets matching '{video_stem}_id*.npz' in {tracking_dir}")

    frame_dets: Dict[int, List[dict]] = defaultdict(list)
    raw_video_size: Optional[Tuple[int, int]] = None

    for fp in files:
        d = np.load(fp, allow_pickle=True)
        if raw_video_size is None and "video_size" in d:
            vs = d["video_size"]
            raw_video_size = (int(vs[0]), int(vs[1]))
        tid = int(d["id"][0])

        frames = d["frame"]
        pose_x = np.stack([d[f"poseX{i}"] for i in range(9)], axis=1)  # (N, 9)
        pose_y = np.stack([d[f"poseY{i}"] for i in range(9)], axis=1)
        conf = d["detection_p"] if "detection_p" in d else np.ones(len(frames))

        for r in range(len(frames)):
            xs = pose_x[r]
            ys = pose_y[r]
            valid = (
                np.isfinite(xs) & np.isfinite(ys)
                & (xs > 0) & (ys > 0)
                & (xs <= video_w) & (ys <= video_h)
            )
            if valid.sum() < 2:
                continue

            vx = xs[valid]
            vy = ys[valid]
            x1, x2 = float(vx.min()), float(vx.max())
            y1, y2 = float(vy.min()), float(vy.max())

            frame_dets[int(frames[r])].append({
                "tid": tid,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "cx": (x1 + x2) / 2.0, "cy": (y1 + y2) / 2.0,
                "conf": float(conf[r]),
                "keypoints": list(zip(vx.tolist(), vy.tolist())),
            })

    return frame_dets, len(files), raw_video_size


def load_pixel_tracks(csv_path: str) -> Tuple[Dict[int, List[dict]], int]:
    """
    Loads pixel-space (distorted) tracks from a MOT-style CSV, as written by
    ``trex_to_bambi.py`` (``tracks_pixel.csv``).

    Columns: frame, track_id, x1, y1, x2, y2, confidence, class_id, flag
    (raw video-pixel coordinates; no pose key-points).

    :return: (frame_dets, n_tracks) in the same shape as
             :func:`load_trex_tracklets`, with an empty ``keypoints`` list.
    """
    frame_dets: Dict[int, List[dict]] = defaultdict(list)
    tids: set = set()

    with open(csv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 6:
                continue

            frame = int(float(parts[0]))
            tid = int(float(parts[1]))
            x1, y1 = float(parts[2]), float(parts[3])
            x2, y2 = float(parts[4]), float(parts[5])
            conf = float(parts[6]) if len(parts) > 6 else 1.0
            tids.add(tid)

            frame_dets[frame].append({
                "tid": tid,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "cx": (x1 + x2) / 2.0, "cy": (y1 + y2) / 2.0,
                "conf": conf,
                "keypoints": [],
            })

    return frame_dets, len(tids)


def load_geo_tracks(csv_path: str) -> Tuple[Dict[int, List[dict]], Optional[Tuple[float, float, float, float]]]:
    """
    Loads the geo-referenced tracks from ``tracks.csv``.

    Columns: frame, tid, x1, y1, z1, x2, y2, z2, conf, cls, interpolated
    (coordinates in the projected CRS, e.g. UTM easting/northing).

    :return: (frame_geo, extent) with ``frame_geo`` mapping frame index to a list
             of dicts (tid, gx1, gy1, gx2, gy2, conf, interp) and ``extent`` the
             global (min_x, max_x, min_y, max_y) bounding box.
    """
    frame_geo: Dict[int, List[dict]] = defaultdict(list)
    min_x, max_x = float("inf"), float("-inf")
    min_y, max_y = float("inf"), float("-inf")

    with open(csv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 8:
                continue

            frame = int(float(parts[0]))
            tid = int(float(parts[1]))
            gx1, gy1 = float(parts[2]), float(parts[3])
            gx2, gy2 = float(parts[5]), float(parts[6])
            conf = float(parts[8]) if len(parts) > 8 else 1.0
            interp = int(float(parts[10])) if len(parts) > 10 else 0

            min_x = min(min_x, gx1, gx2)
            max_x = max(max_x, gx1, gx2)
            min_y = min(min_y, gy1, gy2)
            max_y = max(max_y, gy1, gy2)

            frame_geo[frame].append({
                "tid": tid,
                "gx1": gx1, "gy1": gy1, "gx2": gx2, "gy2": gy2,
                "conf": conf, "interp": interp,
            })

    if min_x == float("inf"):
        return frame_geo, None
    return frame_geo, (min_x, max_x, min_y, max_y)


def load_dem_offsets(dem_json_path: str) -> Tuple[float, float, float]:
    """
    Read the DEM metadata JSON and return the ``(x_off, y_off, z_off)`` origin
    offset in the projected CRS, mirroring the logic in ``trex_to_bambi.py``.

    A geographic (degree) origin is auto-converted to UTM so it matches the
    coordinates in ``tracks.csv``.
    """
    import json

    with open(dem_json_path, "r", encoding="utf-8") as f:
        dem_meta = json.load(f)
    x_off, y_off, z_off = (float(v) for v in dem_meta["origin"])

    # Geographic origin (degrees) -> UTM, same fix as trex_to_bambi.py.
    if abs(x_off) <= 180 and abs(y_off) <= 90:
        try:
            from pyproj import Transformer
            wgs84 = dem_meta.get("origin_wgs84", {})
            lat = float(wgs84.get("latitude", y_off))
            lon = float(wgs84.get("longitude", x_off))
            z_off = float(wgs84.get("altitude", z_off))
            crs_str = str(dem_meta.get("crs", "EPSG:32633"))
            epsg_num = int(crs_str.split(":")[-1]) if "EPSG:" in crs_str else 32633
            t = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg_num}", always_xy=True)
            x_off, y_off = t.transform(lon, lat)
            print(f"   DEM geographic origin auto-converted to UTM: ({x_off:.2f}, {y_off:.2f})")
        except Exception as e:
            print(f"   Warning: DEM origin conversion failed ({e}). Drone positions may be wrong.")
    return x_off, y_off, z_off


def load_drone_positions(poses_path: str, dem_json_path: str
                         ) -> Tuple[Dict[int, Tuple[float, float]], Optional[Tuple[float, float, float, float]]]:
    """
    Load per-frame drone positions from a poses JSON.

    The poses ``location`` is in mesh-local coordinates (the same space
    ``trex_to_bambi.py`` projects in); adding the DEM origin offset places it in
    the projected CRS of ``tracks.csv``. The poses image index is the frame
    index, matching the geo-referenced tracks.

    :return: (frame_drone, extent) where ``frame_drone`` maps a frame index to a
             ``(x, y)`` position and ``extent`` is the (min_x, max_x, min_y,
             max_y) bounding box of the trajectory.
    """
    import json

    x_off, y_off, _z_off = load_dem_offsets(dem_json_path)

    with open(poses_path, "r", encoding="utf-8") as f:
        poses = json.load(f)
    images = poses["images"] if "images" in poses else poses

    frame_drone: Dict[int, Tuple[float, float]] = {}
    min_x, max_x = float("inf"), float("-inf")
    min_y, max_y = float("inf"), float("-inf")

    for frame_idx, img in enumerate(images):
        loc = img.get("location")
        if loc is None or len(loc) < 2:
            continue
        x = float(loc[0]) + x_off
        y = float(loc[1]) + y_off
        frame_drone[frame_idx] = (x, y)
        min_x, max_x = min(min_x, x), max(max_x, x)
        min_y, max_y = min(min_y, y), max(max_y, y)

    if min_x == float("inf"):
        return frame_drone, None
    return frame_drone, (min_x, max_x, min_y, max_y)


# ============================================================
# 4. PANEL RENDERING
# ============================================================

def draw_video_panel(frame, dets, draw_scale, draw_keypoints=True, track_ids=None):
    """Draws TRex bounding boxes (and key-points) onto a (resized) video frame."""
    for det in dets:
        tid = det["tid"]
        if track_ids is not None and tid not in track_ids:
            continue
        color = id_to_color(tid)

        x1 = int(det["x1"] * draw_scale)
        y1 = int(det["y1"] * draw_scale)
        x2 = int(det["x2"] * draw_scale)
        y2 = int(det["y2"] * draw_scale)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"ID {tid} {det['conf']:.2f}"
        cv2.putText(frame, label, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        if draw_keypoints:
            for (kx, ky) in det["keypoints"]:
                cv2.circle(frame, (int(kx * draw_scale), int(ky * draw_scale)), 2, color, -1)


def draw_map_panel(map_img, geo_dets, canvas_cfg, track_history, visible_tids, track_ids=None):
    """Draws geo-referenced boxes + trajectory trails onto the map canvas."""
    if geo_dets:
        for det in geo_dets:
            tid = det["tid"]
            if track_ids is not None and tid not in track_ids:
                continue
            visible_tids.add(tid)
            color = id_to_color(tid)

            px1, py1 = world_to_canvas(det["gx1"], det["gy1"], canvas_cfg)
            px2, py2 = world_to_canvas(det["gx2"], det["gy2"], canvas_cfg)

            if det["interp"]:
                draw_dashed_rectangle(map_img, (px1, py1), (px2, py2), color, 2, 6)
            else:
                cv2.rectangle(map_img, (px1, py1), (px2, py2), color, 2)

            cv2.putText(map_img, f"ID {tid}", (px1, max(0, min(py1, py2) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

            cx = (det["gx1"] + det["gx2"]) / 2.0
            cy = (det["gy1"] + det["gy2"]) / 2.0
            track_history[tid].append(world_to_canvas(cx, cy, canvas_cfg))

    # Trajectory trails + labels for tracks no longer visible.
    for tid, pts in track_history.items():
        if track_ids is not None and tid not in track_ids:
            continue
        color = id_to_color(tid)
        if len(pts) > 1:
            cv2.polylines(map_img, [np.array(pts)], False, color, 1)
        if tid not in visible_tids and pts:
            lx, ly = pts[-1]
            cv2.putText(map_img, f"ID {tid}", (lx + 5, ly - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


# Distinct, fixed colour for the drone marker/trail (BGR).
DRONE_COLOR = (0, 215, 255)  # amber


def draw_drone_on_map(map_img, drone_pos, canvas_cfg, drone_history):
    """Draws the drone trajectory trail and its current position on the map."""
    if drone_pos is not None:
        drone_history.append(world_to_canvas(drone_pos[0], drone_pos[1], canvas_cfg))

    if len(drone_history) > 1:
        cv2.polylines(map_img, [np.array(drone_history)], False, DRONE_COLOR, 1)

    if drone_pos is not None:
        px, py = drone_history[-1]
        cv2.circle(map_img, (px, py), 6, DRONE_COLOR, -1)
        cv2.circle(map_img, (px, py), 6, (0, 0, 0), 1)
        cv2.line(map_img, (px - 9, py), (px + 9, py), DRONE_COLOR, 1)
        cv2.line(map_img, (px, py - 9), (px, py + 9), DRONE_COLOR, 1)
        cv2.putText(map_img, "Drone", (px + 8, py - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, DRONE_COLOR, 1, cv2.LINE_AA)


# ============================================================
# 5. MAIN
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True, help="Source video file.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--tracking-dir",
                     help="Directory holding the TRex *_id<N>.npz tracklets (pose key-points).")
    src.add_argument("--pixel-tracks-csv",
                     help="Pixel-space MOT CSV instead of TRex tracklets "
                          "(e.g. tracks_pixel_w/tracks_pixel.csv from trex_to_bambi). "
                          "Bounding boxes only, no pose key-points.")
    p.add_argument("--tracks-csv", required=True,
                   help="Geo-referenced tracks CSV (e.g. tracks_w/tracks.csv from trex_to_bambi).")
    p.add_argument("--poses", default=None,
                   help="Optional poses JSON (e.g. poses_w.json). When given, the drone "
                        "position is drawn on the map. Requires --dem-json for the origin offset.")
    p.add_argument("--dem-json", default=None,
                   help="DEM metadata JSON (e.g. flat_surface_dem.json) providing the origin "
                        "offset used to place the (relative) poses into the tracks CRS. "
                        "Required when --poses is given.")
    p.add_argument("--output", default=None,
                   help="Output video path. Defaults to <video_dir>/<stem>_trex_vis.mp4")
    p.add_argument("--epsg", type=int, default=32643,
                   help="EPSG code of the tracks.csv coordinates (used for the satellite map).")
    p.add_argument("--display-width", type=int, default=1280,
                   help="Width the video panel is downscaled to for drawing/output.")
    p.add_argument("--map-size", type=int, default=900, help="Map canvas size (square).")
    p.add_argument("--fps", type=float, default=None, help="Output FPS. Defaults to source FPS.")
    p.add_argument("--no-map", action="store_true", help="Disable the satellite background.")
    p.add_argument("--no-keypoints", action="store_true", help="Do not draw pose key-points.")
    p.add_argument("--no-live", action="store_true", help="Do not show a live preview window.")
    p.add_argument("--no-video", action="store_true", help="Do not write an output video.")
    p.add_argument("--track-ids", type=int, nargs="*", default=None,
                   help="Optional subset of track ids to display.")
    p.add_argument("--max-frames", type=int, default=None, help="Stop after N frames (debugging).")
    p.add_argument("--map-cache", default=None, help="Directory to cache downloaded map tiles.")
    p.add_argument("--calib", default=None,
                   help="Camera calibration JSON (mtx/dist). When provided, pixel-space bounding "
                        "boxes are undistorted before drawing. Use this when --video is made from "
                        "the QGIS-exported undistorted frames rather than the original raw video.")
    p.add_argument("--raw-size", type=int, nargs=2, metavar=("W", "H"), default=None,
                   help="Raw video dimensions in pixels (e.g. 5120 2700). Required with --calib "
                        "when using --pixel-tracks-csv; inferred from the NPZ files when using "
                        "--tracking-dir.")
    return p.parse_args()


def main():
    args = parse_args()

    video_path = args.video
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    video_stem = Path(video_path).stem

    track_ids = set(args.track_ids) if args.track_ids else None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out_fps = args.fps or src_fps

    print(f"Video      : {video_path}  ({video_w}x{video_h} @ {src_fps:.1f} fps)")

    # --- Load tracking data (TRex tracklets or pixel-space MOT CSV) ---
    npz_raw_size: Optional[Tuple[int, int]] = None
    if args.pixel_tracks_csv:
        frame_dets, n_tracks = load_pixel_tracks(args.pixel_tracks_csv)
        print(f"MOT pixel  : {n_tracks} tracks, detections on {len(frame_dets)} frames "
              f"({os.path.basename(args.pixel_tracks_csv)})")
    else:
        frame_dets, n_tracks, npz_raw_size = load_trex_tracklets(
            args.tracking_dir, video_stem, video_w, video_h)
        print(f"TRex       : {n_tracks} tracklets, detections on {len(frame_dets)} frames")

    # --- Undistort pixel coordinates when --video is the exported undistorted frames ---
    if args.calib:
        raw_size = tuple(args.raw_size) if args.raw_size else npz_raw_size
        if raw_size is None:
            raise ValueError(
                "--calib requires --raw-size when using --pixel-tracks-csv "
                "(the raw video dimensions, e.g. --raw-size 5120 2700)."
            )
        undistorter = Undistorter(args.calib, raw_size)
        print(f"Undistort  : {raw_size[0]}x{raw_size[1]} -> "
              f"{undistorter.new_size[0]}x{undistorter.new_size[1]} "
              f"using {os.path.basename(args.calib)}")
        undistort_frame_dets(frame_dets, undistorter)

    frame_geo, extent = load_geo_tracks(args.tracks_csv)
    if extent is None:
        raise RuntimeError(f"No geo-referenced tracks found in {args.tracks_csv}")
    print(f"Geo tracks : {len(frame_geo)} frames, extent E[{extent[0]:.1f},{extent[1]:.1f}] "
          f"N[{extent[2]:.1f},{extent[3]:.1f}]")

    # --- Optional drone positions from poses JSON ---
    frame_drone: Dict[int, Tuple[float, float]] = {}
    if args.poses:
        if not args.dem_json:
            raise ValueError("--poses requires --dem-json (for the origin offset).")
        print("Drone      : loading poses ...")
        frame_drone, drone_extent = load_drone_positions(args.poses, args.dem_json)
        print(f"Drone      : {len(frame_drone)} positions")
        if drone_extent is not None:
            # Grow the map extent so the whole drone trajectory stays visible.
            extent = (
                min(extent[0], drone_extent[0]), max(extent[1], drone_extent[1]),
                min(extent[2], drone_extent[2]), max(extent[3], drone_extent[3]),
            )

    # --- Map canvas setup ---
    draw_scale = args.display_width / video_w
    disp_h = int(round(video_h * draw_scale))

    map_size = args.map_size
    margin = 60
    padded_extent = pad_extent_to_match_aspect_ratio(extent, map_size, map_size, margin)
    canvas_cfg = make_global_canvas(padded_extent, map_size, map_size, margin)

    map_bg = None
    if not args.no_map:
        if not _HAS_MAP_DEPS:
            print("Map deps (requests/pyproj) unavailable - skipping satellite background.")
        else:
            print("Downloading satellite background ...")
            prov = MapTileProvider(MapTileProvider.ESRI_SATELLITE, args.map_cache, utm_epsg=args.epsg)
            map_bg = prov.get_map_background(padded_extent, canvas_cfg)
            if map_bg is not None:
                map_bg = (map_bg * 0.55).astype(np.uint8)
            else:
                print("Could not build map background (offline?) - using blank canvas.")

    # --- Output path ---
    out_path = args.output
    if out_path is None:
        out_path = os.path.join(os.path.dirname(video_path), f"{video_stem}_trex_vis.mp4")

    show_live = not args.no_live
    create_video = not args.no_video
    if show_live:
        cv2.namedWindow("TRex Vis", cv2.WINDOW_NORMAL)

    track_history: Dict[int, list] = defaultdict(list)
    drone_history: list = []

    def frame_generator():
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.max_frames is not None and frame_idx >= args.max_frames:
                break

            # --- Video panel (pixel space) ---
            vid_panel = cv2.resize(frame, (args.display_width, disp_h), interpolation=cv2.INTER_AREA)
            draw_video_panel(vid_panel, frame_dets.get(frame_idx, []), draw_scale,
                             draw_keypoints=not args.no_keypoints, track_ids=track_ids)
            cv2.putText(vid_panel, "Video (pixel space)", (10, disp_h - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # --- Map panel (geo space) ---
            map_img = map_bg.copy() if map_bg is not None else \
                np.zeros((map_size, map_size, 3), dtype=np.uint8)
            visible_tids: set = set()
            draw_map_panel(map_img, frame_geo.get(frame_idx, []), canvas_cfg,
                           track_history, visible_tids, track_ids=track_ids)
            if frame_drone:
                draw_drone_on_map(map_img, frame_drone.get(frame_idx), canvas_cfg, drone_history)
            draw_axes_on_canvas(map_img, canvas_cfg)
            cv2.putText(map_img, f"Geo tracks (EPSG:{args.epsg})", (10, map_size - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # --- Combine side by side (match heights) ---
            map_scale = disp_h / map_size
            map_resized = cv2.resize(map_img, (int(map_size * map_scale), disp_h))
            combined = np.hstack([vid_panel, map_resized])

            # libx264/yuv420p requires even width & height.
            ch, cw = combined.shape[:2]
            combined = combined[:ch - (ch % 2), :cw - (cw % 2)]

            cv2.putText(combined, f"{video_stem} | frame {frame_idx}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            if show_live:
                cv2.imshow("TRex Vis", combined)
                k = cv2.waitKey(1)
                if k == 27 or k == ord("q"):
                    raise KeyboardInterrupt

            yield frame_idx, combined
            frame_idx += 1

    try:
        if create_video:
            writer = make_ffmpeg_writer()
            print(f"Writing video -> {out_path}")
            if writer is not None:
                writer.write(out_path, frame_generator(), target_fps=out_fps)
            else:
                # cv2.VideoWriter fallback (mp4v).
                vw = None
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                for _, img in frame_generator():
                    if vw is None:
                        h, w = img.shape[:2]
                        vw = cv2.VideoWriter(out_path, fourcc, out_fps, (w, h))
                    vw.write(img)
                if vw is not None:
                    vw.release()
            print(f"Done: {out_path}")
        else:
            # Live preview only.
            for _ in frame_generator():
                pass
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        cap.release()
        if show_live:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
