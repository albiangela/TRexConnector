"""
Convert TRex bounding-box tracklets (.npz) into

1. a flat detection file            -> detections.txt
2. a pixel-space (distorted) MOT track file -> tracks_pixel.csv (raw x1/y1/x2/y2)
3. a geo-referenced detection file  -> georeferenced.txt   (global x/y/z)
4. a geo-referenced MOT track file  -> tracks.csv

The geo-referencing replicates the bambi_detection projection pipeline
(``label_to_world_coordinates`` against the digital elevation model, using the
per-frame camera taken from the matched ``poses`` file and an optional global
correction).  Because the TRex detections live in *raw* video pixel space
(e.g. 5120x2700) while the bambi poses correspond to the *undistorted* square
frame (e.g. 2700x2700), the detection corners are first undistorted with the
camera calibration, exactly as bambi did when it produced the frames/poses.

Output coordinates are in the DEM CRS (world = mesh-local + DEM origin offset),
matching the reference ``georeferenced.txt`` / ``tracks.csv`` formats.

Example
-------
python trex_to_bambi.py ^
    --npz-dir       "C:/.../pink/tracking" ^
    --dem-glb       "C:/.../pink/original-video-file/qgis/DJI_..._DEM.glb" ^
    --dem-json      "C:/.../pink/original-video-file/qgis/DJI_..._DEM.json" ^
    --poses         "C:/.../pink/original-video-file/qgis/poses_w.json" ^
    --calib         "C:/.../M30T/W_calib.json" ^
    --out-dir       "C:/.../pink/qgis4"
"""

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from alfspy.core.rendering import Camera, Resolution
from alfspy.render.render import read_gltf
from bambi.util.projection_util import label_to_world_coordinates
from pyrr import Quaternion, Vector3
from trimesh import Trimesh


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #
class Detection:
    """A single bounding box on a single frame, belonging to a track."""

    __slots__ = ("frame", "track_id", "x1", "y1", "x2", "y2", "confidence", "class_id")

    def __init__(self, frame, track_id, x1, y1, x2, y2, confidence, class_id):
        self.frame = int(frame)
        self.track_id = int(track_id)
        self.x1 = float(x1)
        self.y1 = float(y1)
        self.x2 = float(x2)
        self.y2 = float(y2)
        self.confidence = float(confidence)
        self.class_id = int(class_id)


# --------------------------------------------------------------------------- #
# Step 1: read TRex .npz tracklets
# --------------------------------------------------------------------------- #
def _pose_keys(data) -> List[int]:
    """Return the sorted indices ``i`` for which both poseX{i} and poseY{i} exist."""
    idxs = []
    for key in data.keys():
        if key.startswith("poseX"):
            suffix = key[len("poseX"):]
            if suffix.isdigit() and f"poseY{suffix}" in data:
                idxs.append(int(suffix))
    return sorted(idxs)


def read_npz_tracklets(npz_dir: str, track_id_offset: int = 0) -> Tuple[List[Detection], Tuple[int, int]]:
    """
    Read every ``*.npz`` tracklet in ``npz_dir`` and turn it into a list of
    :class:`Detection`.  The bounding box of each frame is the axis-aligned box
    spanning the (finite) pose points poseX*/poseY*.

    Returns the detections and the raw video size ``(width, height)`` found in
    the npz files.
    """
    files = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {npz_dir}")

    detections: List[Detection] = []
    video_size: Optional[Tuple[int, int]] = None

    for path in files:
        data = np.load(path, allow_pickle=True)

        if "video_size" in data and video_size is None:
            vs = data["video_size"]
            video_size = (int(vs[0]), int(vs[1]))

        # track id: prefer the embedded id, fall back to the idN in the file name
        if "id" in data and len(data["id"]):
            track_id = int(np.asarray(data["id"]).ravel()[0])
        else:
            stem = Path(path).stem
            track_id = int(stem.split("id")[-1]) if "id" in stem else len(detections)
        track_id += track_id_offset

        frames = np.asarray(data["frame"]).astype(int)
        conf = np.nan_to_num(np.asarray(data["detection_p"], dtype=float), nan=0.0)
        cls = np.nan_to_num(np.asarray(data["detection_class"], dtype=float), nan=0.0).astype(int)

        pidx = _pose_keys(data)
        if not pidx:
            print(f"  WARNING: {Path(path).name} has no pose points, skipped")
            continue
        pose_x = np.stack([np.asarray(data[f"poseX{i}"], dtype=float) for i in pidx], axis=1)
        pose_y = np.stack([np.asarray(data[f"poseY{i}"], dtype=float) for i in pidx], axis=1)

        n = len(frames)
        kept = 0
        for i in range(n):
            xs = pose_x[i]
            ys = pose_y[i]
            mask = np.isfinite(xs) & np.isfinite(ys)
            if not mask.any():
                continue  # no valid pose point -> no bounding box for this frame
            xs = xs[mask]
            ys = ys[mask]
            detections.append(
                Detection(
                    frame=frames[i],
                    track_id=track_id,
                    x1=xs.min(), y1=ys.min(), x2=xs.max(), y2=ys.max(),
                    confidence=conf[i] if i < len(conf) else 1.0,
                    class_id=cls[i] if i < len(cls) else 0,
                )
            )
            kept += 1
        print(f"  {Path(path).name}: track {track_id}, {kept} detections")

    # deterministic ordering: by frame, then track id (matches reference files)
    detections.sort(key=lambda d: (d.frame, d.track_id))
    if video_size is None:
        video_size = (0, 0)
    return detections, video_size


# --------------------------------------------------------------------------- #
# Step 1 output: detections.txt
# --------------------------------------------------------------------------- #
def write_detections_txt(detections: List[Detection], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# frame x1 y1 x2 y2 confidence class_id\n")
        for d in detections:
            f.write(
                f"{d.frame} {d.x1:.2f} {d.y1:.2f} {d.x2:.2f} {d.y2:.2f} "
                f"{d.confidence:.4f} {d.class_id}\n"
            )
    print(f"Wrote {len(detections)} detections -> {out_path}")


# --------------------------------------------------------------------------- #
# Step 2: pixel-space MOT tracks (distorted or undistorted)
# --------------------------------------------------------------------------- #
def write_pixel_tracks_csv(detections: List[Detection], out_path: str,
                           undistorter: Optional["Undistorter"] = None) -> None:
    """
    Write pixel-space, non-geo-referenced tracks in MOT format.

    When ``undistorter`` is None the raw (distorted) video-pixel coordinates
    from the npz tracklets are written directly.  When an ``Undistorter`` is
    supplied the four bounding-box corners are mapped through the same
    undistortion pipeline used by the BAMBI frame extractor, producing
    coordinates in the undistorted square-frame pixel space.

    Columns: frame(8d), track_id, x1, y1, x2, y2, confidence, class_id, flag
    (flag is always 0 = real detection, matching tracks.csv).
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for d in detections:
            if undistorter is not None:
                corners = np.array(
                    [[d.x1, d.y1], [d.x2, d.y1], [d.x2, d.y2], [d.x1, d.y2]],
                    dtype=np.float32,
                )
                und = undistorter.points(corners)
                x1 = float(und[:, 0].min())
                y1 = float(und[:, 1].min())
                x2 = float(und[:, 0].max())
                y2 = float(und[:, 1].max())
            else:
                x1, y1, x2, y2 = d.x1, d.y1, d.x2, d.y2
            f.write(
                f"{d.frame:08d},{d.track_id},{x1:.6f},{y1:.6f},"
                f"{x2:.6f},{y2:.6f},{d.confidence:.6f},{d.class_id},0\n"
            )
    print(f"Wrote {len(detections)} pixel-space track rows -> {out_path}")


# --------------------------------------------------------------------------- #
# Undistortion (raw video pixels -> undistorted square frame pixels)
# --------------------------------------------------------------------------- #
class Undistorter:
    """
    Reproduces the bambi ``CalibratedVideoFrameAccessor`` undistortion so that
    detection coordinates measured on the *raw* video can be mapped into the
    *undistorted* frame the poses/cameras were built for.
    """

    def __init__(self, calib_path: str, raw_size: Tuple[int, int],
                 new_size: Optional[Tuple[int, int]] = None,
                 alpha: float = 0.5, center_principal_point: bool = True,
                 force_same_fov: bool = True):
        with open(calib_path, "r", encoding="utf-8") as f:
            calib = json.load(f)
        self.mtx = np.asarray(calib["mtx"], dtype=float)
        self.dist = np.asarray(calib["dist"], dtype=float)

        w, h = raw_size
        # The undistortion target must equal the *extracted-frame* size the poses were
        # built for. bambi keeps the full aspect ratio when preserve_aspect_ratio=True
        # (e.g. 5120x2700) or forces a square (min(w,h)) otherwise. Pass new_size (the
        # mask / extracted-frame resolution) to match; only fall back to square when
        # unknown. Projecting a non-square frame as a forced square offsets/compresses
        # the georeferenced boxes horizontally.
        if new_size is None:
            wh = min(w, h)
            new_size = (wh, wh)
        self.new_size = new_size
        ncm, _roi = cv2.getOptimalNewCameraMatrix(
            self.mtx, self.dist, (w, h), alpha, self.new_size,
            centerPrincipalPoint=center_principal_point,
        )
        # force_same_fov requires a square frame (see prepare_undistort assertion).
        if force_same_fov and self.new_size[0] == self.new_size[1]:
            fxy = max(ncm[0, 0], ncm[1, 1])
            ncm[0, 0] = ncm[1, 1] = fxy
        self.new_camera_matrix = ncm

    @property
    def resolution(self) -> Resolution:
        return Resolution(self.new_size[0], self.new_size[1])

    def points(self, pts_xy: np.ndarray) -> np.ndarray:
        """Undistort an (N, 2) array of raw pixel coordinates."""
        pts = np.asarray(pts_xy, dtype=np.float32).reshape(-1, 1, 2)
        out = cv2.undistortPoints(pts, self.mtx, self.dist, P=self.new_camera_matrix)
        return out.reshape(-1, 2)


# --------------------------------------------------------------------------- #
# Camera for a frame (mirrors bambi / georeference_deepsort_mot)
# --------------------------------------------------------------------------- #
def get_camera_for_frame(poses: dict, frame_idx: int,
                         cor_rotation_eulers: Vector3, cor_translation: Vector3,
                         aspect_ratio: float = 1.0) -> Camera:
    image_data = poses["images"][frame_idx]
    fovy = image_data["fovy"][0]
    position = Vector3(image_data["location"]) + cor_translation
    rotation_eulers = (
        Vector3([np.deg2rad(v % 360.0) for v in image_data["rotation"]]) - cor_rotation_eulers
    ) * -1
    rotation = Quaternion.from_eulers(rotation_eulers)
    return Camera(fovy=fovy, aspect_ratio=aspect_ratio, position=position, rotation=rotation)


def load_correction(path: Optional[str]) -> Tuple[Vector3, Vector3]:
    """Return (rotation_eulers, translation) correction vectors; identity if no file."""
    translation = {"x": 0.0, "y": 0.0, "z": 0.0}
    rotation = {"x": 0.0, "y": 0.0, "z": 0.0}
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            corr = json.load(f)
        translation = corr.get("translation", translation)
        rotation = corr.get("rotation", rotation)
    cor_translation = Vector3([translation["x"], translation["y"], translation["z"]], dtype="f4")
    cor_rotation_eulers = Vector3([rotation["x"], rotation["y"], rotation["z"]], dtype="f4")
    return cor_rotation_eulers, cor_translation


# --------------------------------------------------------------------------- #
# Steps 2 + 3: georeference + write georeferenced.txt and tracks.csv
# --------------------------------------------------------------------------- #
def georeference(
    detections: List[Detection],
    poses: dict,
    tri_mesh: Trimesh,
    offsets: Tuple[float, float, float],
    input_resolution: Resolution,
    cor_rotation_eulers: Vector3,
    cor_translation: Vector3,
    undistorter: Optional[Undistorter],
    georef_path: str,
    tracks_path: str,
) -> None:
    x_off, y_off, z_off = offsets
    n_frames = len(poses["images"])

    os.makedirs(os.path.dirname(georef_path), exist_ok=True)
    os.makedirs(os.path.dirname(tracks_path), exist_ok=True)

    n_ok = 0
    n_fail = 0
    # camera is the same for every detection on a frame -> cache per frame
    camera_cache: Dict[int, Camera] = {}

    with open(georef_path, "w", encoding="utf-8") as gf, \
            open(tracks_path, "w", encoding="utf-8") as tf:
        gf.write("# idx frame min_x min_y min_z max_x max_y max_z confidence class_id\n")

        for idx, d in enumerate(detections):
            if d.frame >= n_frames:
                gf.write(f"{idx} {d.frame} -1 -1 -1 -1 -1 -1 {d.confidence:.4f} {d.class_id}\n")
                n_fail += 1
                continue

            # 4 bbox corners in raw pixel space
            corners = np.array(
                [[d.x1, d.y1], [d.x2, d.y1], [d.x2, d.y2], [d.x1, d.y2]], dtype=np.float32
            )
            if undistorter is not None:
                corners = undistorter.points(corners)
            poly = corners.reshape(-1).tolist()  # x1 y1 x2 y1 x2 y2 x1 y2

            camera = camera_cache.get(d.frame)
            if camera is None:
                camera = get_camera_for_frame(poses, d.frame, cor_rotation_eulers, cor_translation,
                                             aspect_ratio=input_resolution.width / input_resolution.height)
                camera_cache[d.frame] = camera

            world = label_to_world_coordinates(poly, input_resolution, tri_mesh, camera)

            if world is None or len(world) == 0:
                gf.write(f"{idx} {d.frame} -1 -1 -1 -1 -1 -1 {d.confidence:.4f} {d.class_id}\n")
                n_fail += 1
                continue

            xx = world[:, 0] + x_off
            yy = world[:, 1] + y_off
            zz = world[:, 2] + z_off
            min_x, max_x = float(xx.min()), float(xx.max())
            min_y, max_y = float(yy.min()), float(yy.max())
            min_z, max_z = float(zz.min()), float(zz.max())

            gf.write(
                f"{idx} {d.frame} {min_x:.6f} {min_y:.6f} {min_z:.6f} "
                f"{max_x:.6f} {max_y:.6f} {max_z:.6f} {d.confidence:.4f} {d.class_id}\n"
            )
            # geo-referenced MOT row: frame(8) , track , 3D bbox , conf , class , flag(0=real)
            tf.write(
                f"{d.frame:08d},{d.track_id},{min_x:.6f},{min_y:.6f},{min_z:.6f},"
                f"{max_x:.6f},{max_y:.6f},{max_z:.6f},{d.confidence:.6f},{d.class_id},0\n"
            )
            n_ok += 1

            if (idx + 1) % 2000 == 0:
                print(f"  georeferenced {idx + 1}/{len(detections)} (ok={n_ok}, fail={n_fail})")

    print(f"Wrote {n_ok} georeferenced detections (could not project {n_fail}).")
    print(f"  -> {georef_path}")
    print(f"  -> {tracks_path}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _default(*parts: str) -> str:
    return os.path.join(*parts)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    base = r"C:/Users/P41743/Desktop/lndf/Angela/test-data/pink"
    qgis = _default(base, "original-video-file", "qgis")

    p.add_argument("--npz-dir", default=_default(base, "tracking"),
                   help="Folder containing the TRex *.npz tracklets")
    p.add_argument("--dem-glb", default=_default(qgis, "DJI_202403071703_002_athtongue_DEM.glb"),
                   help="Digital elevation model mesh (GLTF/GLB). "
                        "Not required when --flat-surface-msl is set.")
    p.add_argument("--dem-json", default=_default(qgis, "DJI_202403071703_002_athtongue_DEM.json"),
                   help="DEM metadata json (provides the origin offsets / CRS)")
    p.add_argument("--poses", default=_default(qgis, "poses_w.json"),
                   help="Matched poses json (per-frame camera location/rotation/fovy)")
    p.add_argument("--calib", default=r"C:/Users/P41743/Desktop/lndf/M30T/W_calib.json",
                   help="Camera calibration json (mtx/dist) for undistorting detections")
    p.add_argument("--correction", default=None,
                   help="Optional global correction json (translation/rotation)")
    p.add_argument("--mask", default=_default(qgis, "mask_W.png"),
                   help="Mask image, used to infer the undistorted frame resolution "
                        "when undistortion is disabled")
    p.add_argument("--out-dir", default=_default(base, "qgis4"),
                   help="Output folder; detections/georeferenced/tracks subfolders are created")

    p.add_argument("--no-undistort", action="store_true",
                   help="Do not undistort detections (use only if they already live in "
                        "the pose-frame pixel space)")
    p.add_argument("--track-id-offset", type=int, default=0,
                   help="Value added to every track id (use 1 for 1-based MOT ids)")
    p.add_argument("--input-resolution", type=int, nargs=2, metavar=("W", "H"), default=None,
                   help="Override the projection input resolution (defaults to the "
                        "undistorted square size, or the mask size)")
    p.add_argument("--flat-surface-msl", type=float, default=None, metavar="Z_MSL",
                   help="Project detections onto a flat horizontal plane at this MSL elevation "
                        "instead of the DEM mesh. Use 0.0 for sea-surface surveys where the DEM "
                        "is the seafloor and animals are near the water surface.")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    det_path = _default(args.out_dir, "detections_w", "detections.txt")
    pixel_tracks_path = _default(args.out_dir, "tracks_pixel_w", "tracks_pixel.csv")
    pixel_tracks_und_path = _default(args.out_dir, "tracks_pixel_w", "tracks_pixel_undistorted.csv")
    georef_path = _default(args.out_dir, "georeferenced_w", "georeferenced.txt")
    tracks_path = _default(args.out_dir, "tracks_w", "tracks.csv")

    # --- Step 1: npz -> detections -------------------------------------------------
    print("1. Reading TRex tracklets")
    detections, raw_size = read_npz_tracklets(args.npz_dir, track_id_offset=args.track_id_offset)
    print(f"   {len(detections)} detections, raw video size {raw_size}")
    write_detections_txt(detections, det_path)
    write_pixel_tracks_csv(detections, pixel_tracks_path)

    # --- prepare geo-referencing ---------------------------------------------------
    print("2. Loading DEM + poses")
    with open(args.dem_json, "r", encoding="utf-8") as f:
        dem_meta = json.load(f)
    offsets = tuple(float(v) for v in dem_meta["origin"])
    # Detect geographic (degree) origin and convert to UTM — same fix as in run_export_geotiffs.
    # Without this, a geographic JSON gives z_off=-103.61 (DEM min) instead of -98.46 (origin
    # altitude), causing a ~5 m z-plane difference vs the GeoTIFF export and lateral box drift.
    x_off, y_off, z_off = offsets
    if abs(x_off) <= 180 and abs(y_off) <= 90:
        try:
            from pyproj import Transformer
            wgs84 = dem_meta.get("origin_wgs84", {})
            lat = float(wgs84.get("latitude", y_off))
            lon = float(wgs84.get("longitude", x_off))
            z_off = float(wgs84.get("altitude", z_off))
            crs_str = str(dem_meta.get("crs", "EPSG:32633"))
            epsg_num = int(crs_str.split(":")[-1]) if "EPSG:" in crs_str else 32633
            _t = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg_num}", always_xy=True)
            x_off, y_off = _t.transform(lon, lat)
            offsets = (x_off, y_off, z_off)
            print(f"   Geographic origin auto-converted to UTM: ({x_off:.2f}, {y_off:.2f})")
        except Exception as _e:
            print(f"   Warning: origin conversion failed ({_e}). Coordinates may be wrong.")
    print(f"   DEM CRS {dem_meta.get('crs')}, origin offset {offsets}")

    with open(args.poses, "r", encoding="utf-8") as f:
        poses = json.load(f)

    if args.flat_surface_msl is not None:
        z_off = float(offsets[2])
        z_local = args.flat_surface_msl - z_off
        half = 1_000_000.0
        verts = np.array([
            [-half, -half, z_local],
            [ half, -half, z_local],
            [ half,  half, z_local],
            [-half,  half, z_local],
        ], dtype=np.float32)
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        tri_mesh = Trimesh(vertices=verts, faces=faces)
        print(f"   Flat-surface projection: {args.flat_surface_msl:.1f} m MSL "
              f"(= {z_local:.2f} m local). DEM mesh skipped.")
    else:
        mesh_data, _texture = read_gltf(args.dem_glb)
        tri_mesh = Trimesh(vertices=mesh_data.vertices, faces=mesh_data.indices)

    cor_rotation_eulers, cor_translation = load_correction(args.correction)
    print(cor_rotation_eulers, cor_translation)

    # Determine the extracted-frame resolution the poses were built for (the mask shares
    # the extracted-frame dimensions). --input-resolution overrides it. Used both as the
    # undistortion target size and the projection input resolution so detections live in
    # the exact pixel space of the poses.
    extraction_size: Optional[Tuple[int, int]] = None
    if args.input_resolution is not None:
        extraction_size = (int(args.input_resolution[0]), int(args.input_resolution[1]))
    elif os.path.exists(args.mask):
        _m = cv2.imread(args.mask, cv2.IMREAD_UNCHANGED)
        if _m is not None:
            extraction_size = (_m.shape[1], _m.shape[0])

    undistorter = None
    if not args.no_undistort:
        if raw_size == (0, 0):
            raise ValueError("Raw video size unknown; cannot undistort. Pass --no-undistort "
                             "or ensure the npz files contain 'video_size'.")
        undistorter = Undistorter(args.calib, raw_size, new_size=extraction_size)
        print(f"   Undistorting {raw_size} -> {undistorter.new_size} using {os.path.basename(args.calib)}")
        write_pixel_tracks_csv(detections, pixel_tracks_und_path, undistorter)
        print(f"   -> undistorted pixel tracks: {pixel_tracks_und_path}")

    if extraction_size is not None:
        input_resolution = Resolution(extraction_size[0], extraction_size[1])
    elif undistorter is not None:
        input_resolution = undistorter.resolution
    else:
        input_resolution = Resolution(raw_size[0], raw_size[1])
    print(f"   Projection input resolution: {input_resolution.width}x{input_resolution.height}")

    # --- Steps 2 + 3: georeference + tracks ---------------------------------------
    print("3. Georeferencing detections and writing MOT tracks")
    georeference(
        detections, poses, tri_mesh, offsets, input_resolution,
        cor_rotation_eulers, cor_translation, undistorter,
        georef_path, tracks_path,
    )
    print("Done.")


if __name__ == "__main__":
    main()
