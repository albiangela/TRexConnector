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
                                            ──►  georeferenced.txt  (global x/y/z)
                                            ──►  tracks.csv         (geo-referenced MOT)
```

## What it produces

For a folder of TRex `*.npz` tracklets the script writes three files:

1. **`detections_w/detections.txt`** — flat per-frame detections
   `# frame x1 y1 x2 y2 confidence class_id`.
   Each frame's bounding box is the axis-aligned box spanning the tracklet's `poseX*/poseY*`
   points.
2. **`georeferenced_w/georeferenced.txt`** — the same detections projected onto the DEM
   `# idx frame min_x min_y min_z max_x max_y max_z confidence class_id`,
   in the DEM CRS (world = mesh-local + DEM origin offset).
3. **`tracks_w/tracks.csv`** — geo-referenced MOT, carrying each detection's track id:
   `frame(8-digit),track_id,min_x,min_y,min_z,max_x,max_y,max_z,confidence,class_id,flag`.

## How the geo-referencing works

The connector reuses BAMBI's `label_to_world_coordinates` (ray-casting against the DEM mesh
using the per-frame camera from the matched `poses` file, plus the DEM origin offset and an
optional global correction).

TRex detections are measured on the **raw** video (e.g. `5120×2700`), whereas the BAMBI
poses/cameras correspond to the **undistorted, square** frame (e.g. `2700×2700`). The script
therefore undistorts each detection's corners with the camera calibration (`mtx`/`dist`),
reproducing exactly what BAMBI did when it generated the frames and poses. Disable this with
`--no-undistort` only if your detections are already in the pose-frame pixel space.

## Installation

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pulls BAMBI in directly from git:

```
bambi_detection @ git+https://github.com/bambi-eco/bambi_detection.git
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
| `--out-dir` | Output folder; `detections_w/`, `georeferenced_w/`, `tracks_w/` are created. |
| `--no-undistort` | Skip undistortion (detections already in pose-frame space). |
| `--track-id-offset` | Added to every track id (use `1` for 1-based MOT ids). |
| `--input-resolution W H` | Override the projection input resolution. |
| `--bambi-path` | Path to a local `bambi_detection` checkout (see note above). |

## Requirements on the input data

The DEM mesh and the `poses` file **must be in the same coordinate convention** — both
local-to-origin metres, with the global anchor stored in `DEM.json`'s `origin`. When the DEM
mesh and the camera poses live in incompatible coordinate ranges, no detection can be
projected and the geo-referenced outputs will be empty. (The `detections.txt` step is
independent of the DEM and always works.)
