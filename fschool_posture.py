"""
Parsing helpers for TRex fish-school posture tracklets (``*_fschool_posture_id<N>.npz``).

Unlike the shark tracklets (a fixed 9 pose key-points per frame), a fish-school
detection's shape is a variable-length outline polygon plus zero or more
variable-length "hole" polygons (gaps inside the school shape), stored as
flattened, ragged arrays - reverse-engineered from real data since TRex
doesn't ship a format spec for this export:

- ``outline_points`` (sum(outline_lengths), 2): every frame's outline points
  concatenated in frame order; ``outline_lengths[i]`` is frame ``i``'s point
  count (one outline per frame, so this part is a plain ragged array).
- ``hole_points`` (total hole-point count, 2): every frame's hole polygons
  concatenated, holes within a frame concatenated in turn.
- ``hole_counts``: a *nested* ragged index, not one entry per frame. Read
  sequentially, frame ``i``'s entry is
  ``[n_holes_i, len(hole_1), ..., len(hole_{n_holes_i})]``. Consuming this
  sequentially for every frame in order exactly exhausts ``hole_counts`` and
  exactly accounts for every point in ``hole_points`` (verified against real
  data: the running cursor lands exactly on ``len(hole_counts)``, and the sum
  of every extracted hole length equals ``len(hole_points)`` exactly).

All outline/hole points are stored *relative* to a per-frame ``offset``
``(N, 2)`` array - add ``offset[i]`` to get absolute raw-video pixel
coordinates (verified: without the offset, points cluster near local
``(0, 0)``; with it, the resulting bounding box falls inside the raw video
frame bounds).
"""

import glob
import os
import re
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np


def find_fschool_posture_files(npz_dir: str, seq: str) -> List[str]:
    """All ``<seq>_fschool_posture_id<N>.npz`` files for a sequence, sorted by id."""
    files = glob.glob(os.path.join(npz_dir, f"{seq}_fschool_posture_id*.npz"))

    def _id(path: str) -> int:
        m = re.search(r"_id(\d+)\.npz$", path)
        return int(m.group(1)) if m else -1

    return sorted(files, key=_id)


def track_id_from_path(path: str) -> int:
    m = re.search(r"_id(\d+)\.npz$", path)
    if not m:
        raise ValueError(f"Could not find a trailing '_id<N>.npz' in {path}")
    return int(m.group(1))


def parse_hole_layout(hole_counts: np.ndarray, n_frames: int) -> List[List[int]]:
    """Consume ``hole_counts`` sequentially and return, per frame, the list of
    hole-polygon point-counts for that frame (an empty list when the frame has
    no holes). Raises if the array doesn't exactly match ``n_frames`` frames'
    worth of entries - a sign the npz doesn't match the expected format."""
    hole_counts = np.asarray(hole_counts, dtype=np.int64)
    cursor = 0
    layout: List[List[int]] = []
    for _ in range(n_frames):
        if cursor >= len(hole_counts):
            raise ValueError(
                f"hole_counts exhausted after {len(layout)}/{n_frames} frames "
                f"- the npz may not match the expected fschool posture format."
            )
        n_holes = int(hole_counts[cursor])
        cursor += 1
        lengths = hole_counts[cursor:cursor + n_holes].tolist()
        cursor += n_holes
        layout.append(lengths)
    if cursor != len(hole_counts):
        raise ValueError(
            f"hole_counts layout mismatch: consumed {cursor} of {len(hole_counts)} entries "
            f"for {n_frames} frames - the npz may not match the expected format."
        )
    return layout


def iter_frame_shapes(
    data,
    outline_points_key: str = "outline_points",
    outline_lengths_key: str = "outline_lengths",
    hole_points_key: str = "hole_points",
    hole_counts_key: str = "hole_counts",
    offset_key: Optional[str] = "offset",
) -> Iterator[Tuple[int, np.ndarray, List[np.ndarray]]]:
    """Yield ``(frame_idx, outline, holes)`` for every frame in a
    ``*_fschool_posture_id<N>.npz``-shaped dataset, where ``outline`` is an
    ``(L, 2)`` array and ``holes`` is a list of ``(Lh, 2)`` arrays.

    The default field names read the raw pixel-space fields. ``outline_points``
    is stored *relative* to the per-frame ``offset`` and needs it added to land
    in absolute raw-video pixel coordinates - but ``hole_points`` is stored
    *already absolute* (verified against real data: a frame's hole points fall
    squarely inside that frame's absolute outline bounding box with no offset
    added at all, and land far outside it - even outside the raw video frame -
    if the offset is added). So the offset is only ever applied to the outline,
    never to holes, regardless of ``offset_key``.

    export_georeferenced_fschool.py writes a second set of fields
    (``outline_points_geor``/``outline_lengths_geor``/``hole_points_geor``/
    ``hole_counts_geor``) using this exact same ragged-encoding convention but
    already in world coordinates for both outline and holes (no offset to add
    to either) - pass ``offset_key=None`` and the ``*_geor`` field names to
    read those instead.
    """
    frames = np.asarray(data["frames"]).astype(int)
    offset = np.asarray(data[offset_key], dtype=float) if offset_key is not None else None
    outline_lengths = np.asarray(data[outline_lengths_key], dtype=np.int64)
    outline_points = np.asarray(data[outline_points_key], dtype=float)
    hole_points = np.asarray(data[hole_points_key], dtype=float)
    hole_layout = parse_hole_layout(data[hole_counts_key], len(frames))

    o_cursor = 0
    h_cursor = 0
    for i, frame_idx in enumerate(frames):
        off = offset[i] if offset is not None else 0.0
        olen = int(outline_lengths[i])
        outline_abs = outline_points[o_cursor:o_cursor + olen] + off
        o_cursor += olen

        holes_abs = []
        for hlen in hole_layout[i]:
            hole_abs = hole_points[h_cursor:h_cursor + hlen]  # already absolute - no offset
            h_cursor += hlen
            holes_abs.append(hole_abs)

        yield int(frame_idx), outline_abs, holes_abs


def load_frame_shapes(data) -> Dict[int, Tuple[np.ndarray, List[np.ndarray]]]:
    """Same as :func:`iter_frame_shapes` but materialized into a
    ``{frame_idx: (outline_abs, holes_abs)}`` dict for random access."""
    return {frame_idx: (outline, holes) for frame_idx, outline, holes in iter_frame_shapes(data)}


def decimate_polygon(points: np.ndarray, max_points: int) -> np.ndarray:
    """Stride-subsample a closed polygon down to roughly ``max_points`` points.

    These are raw per-pixel contour traces (verified on real data: outlines
    average ~7400 points/frame, up to 19000; individual holes average ~550,
    up to ~7800) - full resolution is unnecessary for georeferencing/display
    and prohibitively slow to ray-cast one point at a time, so both the
    export script and the visualizer decimate through this same function to
    stay consistent with each other.
    """
    if max_points <= 0 or len(points) <= max_points:
        return points
    stride = max(1, len(points) // max_points)
    return points[::stride]
