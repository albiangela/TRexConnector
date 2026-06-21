# TRexConnector

A connector script that bridges **[TRex](https://github.com/mooch443/trex)** tracking
output with the **[BAMBI](https://github.com/bambi-eco/bambi_detection)** wildlife-detection
and geo-referencing workflow.

TRex produces per-tracklet `.npz` files (bounding boxes / pose points per frame, one file
per track). BAMBI projects image-space detections onto a digital elevation model (DEM) to
obtain global world coordinates. This connector reads the TRex tracklets, converts them into
the detection format BAMBI expects, geo-references them through the BAMBI projection pipeline,
and exports a geo-referenced multi-object-tracking (MOT) file. Like that, a TRex run can be consumed
e.g. directly by the BAMBI QGIS workflow. Recommendation: Run the frame extraction in QGIS first and
afterwards use the TRexConnector to geo-reference detections and tracks.

```
TRex (.npz tracklets)  ──►  TRexConnector  ──►  detections.txt
                                            ──►  tracks_pixel.csv   (pixel-space MOT)
                                            ──►  georeferenced.txt  (global x/y/z)
                                            ──►  tracks.csv         (geo-referenced MOT)
```

## What it produces

For a folder of TRex `*.npz` tracklets the script writes, under `--out-dir`:

1. **`detections_w/detections.txt`** — flat per-frame detections
   `# frame x1 y1 x2 y2 confidence class_id`.
   Each frame's bounding box is the axis-aligned box spanning the tracklet's `poseX*/poseY*`
   points.
2. **`tracks_pixel_w/tracks_pixel.csv`** — pixel-space (non-geo-referenced) MOT in the raw
   video pixel space:
   `frame(8-digit),track_id,x1,y1,x2,y2,confidence,class_id,flag`.
   When undistortion is enabled, **`tracks_pixel_w/tracks_pixel_undistorted.csv`** is also
   written, with the same columns but in the undistorted square-frame pixel space.
3. **`georeferenced_w/georeferenced.txt`** — the detections projected onto the DEM
   `# idx frame min_x min_y min_z max_x max_y max_z confidence class_id`,
   in the DEM CRS (world = mesh-local + DEM origin offset).
4. **`tracks_w/tracks.csv`** — geo-referenced MOT, carrying each detection's track id:
   `frame(8-digit),track_id,min_x,min_y,min_z,max_x,max_y,max_z,confidence,class_id,flag`.

Steps 1 and 2 are independent of the DEM and always run; steps 3 and 4 require the DEM/poses
inputs described below.

## How the geo-referencing works

The connector reuses BAMBI's `label_to_world_coordinates` (ray-casting against the DEM mesh
using the per-frame camera from the matched `poses` file, plus the DEM origin offset and an
optional global correction).

TRex detections are measured on the **raw** video (e.g. `5120×2700`), whereas the BAMBI
poses/cameras correspond to the **undistorted, square** frame (e.g. `2700×2700`). The script
therefore undistorts each detection's corners with the camera calibration (`mtx`/`dist`),
reproducing exactly what BAMBI did when it generated the frames and poses. Disable this with
`--no-undistort` only if your detections are already in the pose-frame pixel space.

### Aquatic / marine surveys — flat-surface projection

When the DEM represents the **seafloor** (e.g. a bathymetric model) and the animals are near
the **water surface**, projecting onto the DEM will place detections far from their true
positions — systematically displaced outward from the camera nadir by an amount proportional
to `seafloor_depth × tan(off-nadir_angle)` (tens of metres for typical drone altitudes over
shallow reefs).

Use `--flat-surface-msl <elevation>` to project onto a flat horizontal plane at the given MSL
elevation instead of the DEM mesh. For sea-surface surveys set it to `0.0`:

```bash
python trex_to_bambi.py \
    --npz-dir   /path/to/tracking \
    --dem-json  /path/to/DEM.json \
    --poses     /path/to/poses_w.json \
    --calib     /path/to/W_calib.json \
    --out-dir   /path/to/output \
    --flat-surface-msl 0.0
```

The `--dem-glb` argument is not required (and the mesh file is not loaded) when
`--flat-surface-msl` is set, which also makes the script run significantly faster.

## Installation

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pulls BAMBI (and the `alfspy` rendering/geo library it builds on) directly
from git, alongside the connector's runtime dependencies:

```
bambi_detection @ git+https://github.com/bambi-eco/bambi_detection.git
alfspy @ git+https://github.com/bambi-eco/alfs_py.git
```

## Usage

```bash
python trex_to_bambi.py \
    --npz-dir   /path/to/tracking \
    --dem-glb   /path/to/DEM.glb \
    --dem-json  /path/to/DEM.json \
    --poses     /path/to/poses_w.json \
    --calib     /path/to/W_calib.json \
    --out-dir   /path/to/output
```

### Key options

| Option | Description |
| --- | --- |
| `--npz-dir` | Folder containing the TRex `*.npz` tracklets. |
| `--dem-glb` / `--dem-json` | DEM mesh (GLTF/GLB) and its metadata (origin offsets / CRS). |
| `--poses` | Matched poses JSON (per-frame camera location/rotation/fovy). |
| `--calib` | Camera calibration JSON (`mtx`/`dist`) used to undistort detections. |
| `--correction` | Optional flight-specific correction JSON (`translation`/`rotation`). |
| `--mask` | Mask image, used to infer the undistorted frame resolution when undistortion is disabled. |
| `--out-dir` | Output folder; `detections_w/`, `tracks_pixel_w/`, `georeferenced_w/`, `tracks_w/` are created. |
| `--flat-surface-msl Z` | Project onto a flat plane at `Z` metres MSL instead of the DEM mesh. Use `0.0` for sea-surface surveys where the DEM is the seafloor. `--dem-glb` is not required when this is set. |
| `--no-undistort` | Skip undistortion (detections already in pose-frame space). |
| `--track-id-offset` | Added to every track id (use `1` for 1-based MOT ids). |
| `--input-resolution W H` | Override the projection input resolution. |

## Requirements on the input data

The DEM mesh and the `poses` file **must be in the same coordinate convention** — both
local-to-origin metres, with the global anchor stored in `DEM.json`'s `origin`. When the DEM
mesh and the camera poses live in incompatible coordinate ranges, no detection can be
projected and the geo-referenced outputs will be empty. (The `detections.txt` step is
independent of the DEM and always works.)

## Visualizing the results

`visualize_trex_video_and_map.py` is a companion tool for checking a run end to end. It
renders a side-by-side video:

- **Left — pixel space:** the original video with per-track bounding boxes, ID labels and
  confidence. The boxes come either from the TRex pose key-points (`--tracking-dir`, which
  also draws the `poseX*/poseY*` key-points) or from a pixel-space MOT CSV
  (`--pixel-tracks-csv`, e.g. the `tracks_pixel.csv` produced above) — exactly one of the two
  is required.
- **Right — geo space:** a 2D map of the geo-referenced `tracks.csv` (per-track boxes,
  trajectory trails and axis ticks) over an optional satellite background. When a poses JSON
  (`--poses`) and its DEM metadata (`--dem-json`) are supplied, the drone's own position and
  flight trail are drawn on the map too.

The two panels stay in sync because the TRex tracklets and `tracks.csv` share the same frame
indices and track ids.

```bash
python visualize_trex_video_and_map.py \
    --video        /path/to/20240307_063012765_DJI_0463.MP4 \
    --tracking-dir /path/to/tracking \
    --tracks-csv   /path/to/output/tracks_w/tracks.csv \
    --epsg 32643
```

By default it writes `<video_dir>/<stem>_trex_vis.mp4` and shows a live preview window (press
`q`/`Esc` to quit). The video is encoded with BAMBI's `PipeFFMPEGWriter` (libx264) when
available and falls back to `cv2.VideoWriter` (mp4v) otherwise, so it needs only numpy,
opencv, and — for the satellite map — `pyproj` + `requests`.

### Key options

| Option | Description |
| --- | --- |
| `--video` | Source video file. |
| `--tracking-dir` | Folder containing the TRex `*_id<N>.npz` tracklets (pose key-points). One of `--tracking-dir` / `--pixel-tracks-csv` is required. |
| `--pixel-tracks-csv` | Pixel-space MOT CSV (e.g. `tracks_pixel_w/tracks_pixel.csv`) to drive the left panel instead of TRex tracklets. |
| `--tracks-csv` | Geo-referenced tracks CSV (the `tracks_w/tracks.csv` produced above). Required. |
| `--poses` | Optional poses JSON (e.g. `poses_w.json`); when given, the drone position is drawn on the map. Requires `--dem-json`. |
| `--dem-json` | DEM metadata JSON providing the origin offset used to place the poses into the tracks CRS. |
| `--epsg` | EPSG code of the `tracks.csv` coordinates, used to place the satellite map (default `32643`, UTM 43N). |
| `--output` | Output video path (defaults to `<video_dir>/<stem>_trex_vis.mp4`). |
| `--fps` | Output FPS (defaults to the source FPS). |
| `--calib` | Camera calibration JSON (`mtx`/`dist`); undistorts pixel-space boxes before drawing. Use when the video is made from the QGIS-exported undistorted frames. |
| `--raw-size W H` | Raw video dimensions (e.g. `5120 2700`). Required with `--calib` when using `--pixel-tracks-csv`; inferred from the NPZ files with `--tracking-dir`. |
| `--track-ids` | Optional subset of track ids to display. |
| `--no-map` | Disable the satellite background (no network needed). |
| `--no-keypoints` | Draw only the bounding boxes, not the pose key-points. |
| `--no-live` / `--no-video` | Skip the preview window / skip writing the output video. |
| `--display-width` / `--map-size` | Video panel width and (square) map canvas size in pixels. |
| `--max-frames` | Stop after N frames (useful for a quick check). |
| `--map-cache` | Directory to cache downloaded map tiles. |
