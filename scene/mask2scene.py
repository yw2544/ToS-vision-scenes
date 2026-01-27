#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mask -> TDW scene (shared-wall; zero-width bands collapsed)

Fix:
- Seal exterior perimeter after door cuts to remove small gaps on outer walls.
"""

import argparse, json
from pathlib import Path
from typing import List, Tuple, Dict, Optional

from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.add_ons.image_capture import ImageCapture
from tdw.librarian import ModelRecord
from tdw.librarian import ModelLibrarian, MaterialLibrarian

# ---------------- visuals ----------------
TRIM_OPEN_ENDS = True

GRAY = {"r": 0.86, "g": 0.86, "b": 0.86, "a": 1.0}
FLOOR_COLORS = [
    {"r": 0.82, "g": 0.82, "b": 0.82, "a": 1.0},
    {"r": 0.75, "g": 0.95, "b": 0.80, "a": 1.0},
    {"r": 0.95, "g": 0.85, "b": 0.75, "a": 1.0},
    {"r": 0.95, "g": 0.75, "b": 0.90, "a": 1.0},
    {"r": 0.75, "g": 0.80, "b": 0.95, "a": 1.0},
]
DOOR_COLOR = {"r": 0.95, "g": 0.25, "b": 0.25, "a": 1.0}
DOOR_H = 1.4
EPS = 1e-6
DOOR_RECORD_JSON = Path("/Users/songshe/ai2thor/tdw_import_md/output_fbx/Door/record.json")
DOOR_UNIFORM_SCALE = 0.12     # Uniform scale (adjust as needed)
DOOR_PRESERVE_MATERIALS = False  # Preserve materials (do not set_color)
DOOR_Y_BY_PIVOT_BOTTOM = True   # If pivot at bottom, place by door height/2; else set False

# ---- Door filler (lintel) config ----
FILLER_BOTTOM_Y = 1.4  # the top of the (short) door; filler goes from this y to the wall top
USE_WALL_THICKNESS = True  # keep filler thickness same as wall (recommended)
FIXED_THICKNESS = 0.04     # only used if USE_WALL_THICKNESS=False
USE_DOOR_WIDTH = False      # keep filler width same as door width (recommended)
FIXED_WIDTH = 0.8          # only used if USE_DOOR_WIDTH=False

# ====== Wall material config (NEW) ======
WALL_USE_MATERIAL = False
WALL_MATERIAL_NAME = "plastic_vinyl_glossy_white"           # Material name
WALL_MATERIAL_LIB  = "materials_med.json" # Library: materials_low/med/high.json
WHITE = {"r": 1, "g": 1, "b": 1, "a": 1}
def is_room(v: int) -> bool: return 1 <= v <= 99
def is_door(v: int) -> bool: return v >= 100

# ====== Floor material config (NEW) ======
FLOOR_USE_MATERIAL = True
FLOOR_MATERIAL_NAME = "ceramic_tiles_floral_white"   # Floor material name
FLOOR_MATERIAL_LIB  = "materials_med.json"      # Material library

# ---------------- TDW primitive ----------------
def add_cube(ctrl: Controller, *, pos: Dict, scale: Dict, color: Dict, rot: Dict = None, kin: bool = True):
    # Add a kinematic colored cube primitive
    if rot is None: rot = {"x": 0, "y": 0, "z": 0}
    oid = ctrl.get_unique_id()
    cmds = [
        ctrl.get_add_object("prim_cube", library="models_special.json",
                            object_id=oid, position=pos, rotation=rot),
        {"$type": "scale_object", "id": oid, "scale_factor": scale},
        {"$type": "set_color", "id": oid, "color": color},
    ]
    if kin:
        cmds.append({"$type": "set_kinematic_state", "id": oid,
                     "is_kinematic": True, "use_gravity": False})
    ctrl.communicate(cmds)
    return oid  # Return the object ID

def apply_material_to_special_primitive(ctrl: Controller,
                                        object_id: int,
                                        model_name: str = "prim_cube",
                                        material_name: str = "concrete",
                                        material_library: str = "materials_med.json",
                                        model_library: str = "models_special.json"):
    """
    Apply material to primitives in models_special.json:
    1) add_material (library name must match)
    2) replace all submesh/material_index via set_visual_material
    """
    # Get substructure
    mlib = ModelLibrarian(library=model_library)
    rec = mlib.get_record(model_name)
    sub = rec.substructure

    # Confirm material exists and load it
    matlib = MaterialLibrarian(library=material_library)
    assert any(r.name == material_name for r in matlib.records), \
        f"[ERROR] Material {material_name} is not in {material_library}."
    cmds = [ctrl.get_add_material(material_name, library=material_library)]

    # Set to white first to avoid prior set_color tinting
    cmds.append({"$type": "set_color", "id": object_id, "color": {"r":1, "g":1, "b":1, "a":1}})

    # Apply to all submeshes and slots
    for s in sub:
        submesh = s["name"]
        for i in range(len(s["materials"])):
            cmds.append({
                "$type": "set_visual_material",
                "id": object_id,
                "object_name": submesh,
                "material_name": material_name,
                "material_index": i
            })
            cmds.append({
                "$type": "set_texture_scale",
                "id": object_id,
                "object_name": submesh,
                "material_index": i,
                "scale": {"x": 3.5, "y": 3.5}  # Slightly adjust texture scale
            })
            
    ctrl.communicate(cmds)

# ---------------- segment set ----------------
class SegSet:
    """Maintain a set of merged [a,b] segments on a line."""
    def __init__(self): self.segs: List[Tuple[float, float]] = []
    def add(self, a: float, b: float):
        if b <= a + EPS: return
        a, b = (a, b) if a <= b else (b, a)
        out: List[Tuple[float, float]] = []
        for (u, v) in sorted(self.segs + [(a, b)]):
            if not out: out = [(u, v)]; continue
            pu, pv = out[-1]
            if u <= pv + EPS: out[-1] = (pu, max(pv, v))
            else: out.append((u, v))
        self.segs = out
    def cut(self, a: float, b: float):
        if b <= a + EPS: return
        a, b = (a, b) if a <= b else (b, a)
        out: List[Tuple[float, float]] = []
        for (u, v) in self.segs:
            if v <= a + EPS or u >= b - EPS: out.append((u, v))
            else:
                if u < a - EPS: out.append((u, a))
                if v > b + EPS: out.append((b, v))
        self.segs = out
    def overlaps(self, a: float, b: float) -> bool:
        for (u, v) in self.segs:
            if not (v <= a + EPS or u >= b - EPS):
                return True
        return False
    def __iter__(self): return iter(self.segs)

# ---------------- compression ----------------
def build_compression_maps(mask: List[List[int]]):
    rows, cols = len(mask), len(mask[0])
    keep_cols = [any(is_room(mask[r][c]) for r in range(rows)) for c in range(cols)]
    keep_rows = [any(is_room(mask[r][c]) for c in range(cols)) for r in range(rows)]
    col_map: Dict[int, int] = {}
    row_map: Dict[int, int] = {}
    kept_cols = []; kept_rows = []
    j = 0
    for c in range(cols):
        if keep_cols[c]:
            col_map[c] = j; kept_cols.append(c); j += 1
    ncols_eff = j
    i = 0
    for r in range(rows):
        if keep_rows[r]:
            row_map[r] = i; kept_rows.append(r); i += 1
    nrows_eff = i
    return keep_rows, keep_cols, row_map, col_map, kept_rows, kept_cols, nrows_eff, ncols_eff

def compressed_mask(mask: List[List[int]], kept_rows: List[int], kept_cols: List[int]) -> List[List[int]]:
    return [[mask[r][c] for c in kept_cols] for r in kept_rows]

# ---------------- coordinates (compressed) ----------------
def x_center_comp(j: float, ncols_eff: int, cell: float) -> float:
    return (j - (ncols_eff - 1) / 2.0) * cell
def z_center_comp(i: float, nrows_eff: int, cell: float) -> float:
    return ((nrows_eff - 1) / 2.0 - i) * cell
def x_boundary_comp(jb: int, ncols_eff: int, cell: float) -> float:
    return ((jb + 0.5) - (ncols_eff - 1) / 2.0) * cell
def z_boundary_comp(ib: int, nrows_eff: int, cell: float) -> float:
    return (((nrows_eff - 1) / 2.0) - (ib + 0.5)) * cell

# ---------------- nearest-room helpers ----------------
def nearest_room_col(mask: List[List[int]], r: int, start_c: int, step: int) -> Optional[int]:
    cols = len(mask[0]); c = start_c
    while 0 <= c < cols and not is_room(mask[r][c]): c += step
    return c if 0 <= c < cols and is_room(mask[r][c]) else None
def nearest_room_row(mask: List[List[int]], c: int, start_r: int, step: int) -> Optional[int]:
    rows = len(mask); r = start_r
    while 0 <= r < rows and not is_room(mask[r][c]): r += step
    return r if 0 <= r < rows and is_room(mask[r][c]) else None
def nearest_kept_index(index_map: Dict[int, int], total: int, idx: int, up: bool) -> Optional[int]:
    if idx in index_map: return index_map[idx]
    if up:
        k = idx - 1
        while k >= 0 and k not in index_map: k -= 1
        return index_map[k] if k >= 0 else None
    else:
        k = idx + 1
        while k < total and k not in index_map: k += 1
        return index_map[k] if k < total else None

# ---------------- debug ----------------
def dump_debug(mask, kept_rows, kept_cols, row_map, col_map,
               nrows_eff, ncols_eff, V_idx, H_idx, cell, doors_dbg=None):
    print("\n[DEBUG] kept_rows (orig->comp):", [(r, row_map[r]) for r in kept_rows])
    print("[DEBUG] kept_cols (orig->comp):", [(c, col_map[c]) for c in kept_cols])
    cmask = compressed_mask(mask, kept_rows, kept_cols)
    print("[DEBUG] cmask shape:", len(cmask), "x", len(cmask[0]))
    for i, row in enumerate(cmask):
        print(f"  i={i:2d}:", " ".join(f"{v:3d}" for v in row))
    print("\n[DEBUG] vertical walls (jb -> x, segs[z0,z1]):")
    for jb in sorted(V_idx.keys()):
        x = x_boundary_comp(jb, ncols_eff, cell)
        print(f"  jb={jb:2d}, x={x: .3f}, segs={list(V_idx[jb].segs)}")
    print("\n[DEBUG] horizontal walls (ib -> z, segs[x0,x1]):")
    for ib in sorted(H_idx.keys()):
        z = z_boundary_comp(ib, nrows_eff, cell)
        print(f"  ib={ib:2d}, z={z: .3f}, segs={list(H_idx[ib].segs)}")
    if doors_dbg:
        print("\n[DEBUG] doors handled:")
        for d in doors_dbg: print(" ", d)
    print()

def heal_outer_single_cell_gaps(*, V_idx, H_idx, cell, nrows_eff, ncols_eff):
    """
    Heal only single-cell gaps on the OUTER boundary:
    - Horizontal: ib == -1 (top) or ib == nrows_eff - 1 (bottom)
    - Vertical:   jb == -1 (left) or jb == ncols_eff - 1 (right)
    This won't create new long walls across void; it just bridges tiny holes.
    """
    THRESH = cell * 1.05  # allow small numeric slack

    def heal_segset(segset: SegSet):
        segs = sorted(segset.segs)
        for k in range(len(segs) - 1):
            a0, b0 = segs[k]
            a1, b1 = segs[k + 1]
            gap = a1 - b0
            if gap > EPS and gap <= THRESH:
                # bridge the 1-cell hole
                segset.add(b0, a1)

    # top / bottom horizontal edges
    if -1 in H_idx:
        heal_segset(H_idx[-1])
    if (nrows_eff - 1) in H_idx:
        heal_segset(H_idx[nrows_eff - 1])

    # left / right vertical edges
    if -1 in V_idx:
        heal_segset(V_idx[-1])
    if (ncols_eff - 1) in V_idx:
        heal_segset(V_idx[ncols_eff - 1])

# ---------------- coordinates (non-compressed) ----------------
def x_center(r: int, cell: float, rows: int) -> float:
    """Convert mask row to world X coordinate (cell center)
    New coordinate system: x=row, z=col
    """
    offset_x = -(rows // 2)  # INTEGER offset for rows (X axis)
    return (r + offset_x) * cell

def z_center(c: int, cell: float, cols: int) -> float:
    """Convert mask col to world Z coordinate (cell center)
    New coordinate system: x=row, z=col
    """
    offset_z = -(cols // 2)  # INTEGER offset for cols (Z axis)
    return (c + offset_z) * cell

def x_boundary(r: int, cell: float, rows: int) -> float:
    """Convert mask row to world X coordinate (cell boundary)
    New coordinate system: x=row, z=col
    """
    offset_x = -(rows // 2)  # INTEGER offset for rows (X axis)
    return (r + offset_x) * cell - cell/2

def z_boundary(c: int, cell: float, cols: int) -> float:
    """Convert mask col to world Z coordinate (cell boundary)
    New coordinate system: x=row, z=col
    """
    offset_z = -(cols // 2)  # INTEGER offset for cols (Z axis)
    return (c + offset_z) * cell - cell/2


# ---------------- build ----------------
def build_scene(ctrl,
                mask,
                cell,
                wall_t,
                wall_h,
                door_w,
                door_colors=None,
                floor_y=0,
                floor_thickness=0.05,
                floor_to_wall_face=False,
                add_door_mesh=False,
                debug=False):
    rows, cols = len(mask), len(mask[0])

    # 1) One floor slab per room (use room bbox, optionally shrink to wall inner face)
    _build_rect_floors_by_room(ctrl=ctrl,
                               mask=mask,
                               cell=cell,
                               floor_y=floor_y,
                               floor_thickness=floor_thickness,
                               wall_t=wall_t,
                               floor_to_wall_face=floor_to_wall_face)

    # 2) Continuous walls + precise door openings (including door model and lintel)
    door_ids, filler_ids, door_pos, filler_pos = _build_continuous_walls_with_precise_doors(ctrl, mask, rows, cols,
                                               cell, wall_t, wall_h, door_w,
                                               door_colors)
    
    # Return door/filler tracking info (if needed)
    return (door_ids, filler_ids, door_pos, filler_pos)

# ---- Segment set for precise wall cutting ----
class SegSet:
    """Manage line segments with add/cut operations."""
    def __init__(self):
        self.segments = []  # Store (start, end) segments
    
    def add(self, start, end):
        """Add a segment."""
        if end > start:
            self.segments.append((start, end))
    
    def cut(self, cut_start, cut_end):
        """Cut out an interval from all segments."""
        new_segments = []
        for start, end in self.segments:
            if cut_end <= start or cut_start >= end:
                # No overlap; keep the segment
                new_segments.append((start, end))
            else:
                # Overlap; split segment
                if start < cut_start:
                    # Keep left part
                    new_segments.append((start, cut_start))
                if cut_end < end:
                    # Keep right part
                    new_segments.append((cut_end, end))
        self.segments = new_segments
    
    def __iter__(self):
        """Iterate segments."""
        return iter(self.segments)


# ---- Wall-band detection and orientation ----
def _is_band(v: int) -> bool:
    """Check whether a cell is a wall band (wall or door)."""
    return v == 0 or is_door(v)


def _nearest_room_row(mask, col, start_row, direction):
    """Find nearest room cell along a column in a direction."""
    rows = len(mask)
    r = start_row
    while 0 <= r < rows:
        if is_room(mask[r][col]):
            return r
        r += direction
    return None


def _nearest_room_col(mask, row, start_col, direction):
    """Find nearest room cell along a row in a direction."""
    cols = len(mask[0])
    c = start_col
    while 0 <= c < cols:
        if is_room(mask[row][c]):
            return c
        c += direction
    return None



 
# Treat only value 0 as a pure wall cell
def _is_zero(v: int) -> bool:
    return v == 0

# Compute continuous run length along an axis from (r, c)
# only_zero=True counts only 0; False counts wall band (0 or door)
def _run_len(mask, r, c, axis: str, only_zero: bool, THRESH=3) -> int:
    rows, cols = len(mask), len(mask[0])
    def ok(rr, cc):
        v = mask[rr][cc]
        if only_zero:
            return _is_zero(v)
        else:
            return (v == 0) or is_door(v)

    run = 1
    if axis == "h":
        cc = c - 1
        while cc >= 0 and ok(r, cc): run += 1; cc -= 1
        cc = c + 1
        while cc < cols and ok(r, cc): run += 1; cc += 1
    else:  # "v"
        rr = r - 1
        while rr >= 0 and ok(rr, c): run += 1; rr -= 1
        rr = r + 1
        while rr < rows and ok(rr, c): run += 1; rr += 1
    return run


def _choose_orientation(mask, r, c):
    if not _is_band(mask[r][c]):
        return (False, False)

    rows, cols = len(mask), len(mask[0])

    # Check if any direction can reach a room (original definition)
    up    = _nearest_room_row(mask, c, r - 1, -1) is not None
    down  = _nearest_room_row(mask, c, r + 1, +1) is not None
    left  = _nearest_room_col(mask, r, c - 1, -1) is not None
    right = _nearest_room_col(mask, r, c + 1, +1) is not None

    # ---- Boundary override ----
    if r == 0 or r == rows - 1:
        return (True, False)      # Top/bottom row: horizontal
    if c == 0 or c == cols - 1:
        return (False, True)      # Left/right column: vertical

    THRESH = 3  # Threshold: three consecutive zeros

    # Prefer pure-zero runs (stronger evidence)
    h0 = _run_len(mask, r, c, "h", only_zero=True,  THRESH=THRESH)
    v0 = _run_len(mask, r, c, "v", only_zero=True,  THRESH=THRESH)
    if v0 >= THRESH and h0 < THRESH:
        return (False, True)      # Stronger vertical zero band
    if h0 >= THRESH and v0 < THRESH:
        return (True, False)      # Stronger horizontal zero band
    if h0 >= THRESH and v0 >= THRESH:
        return (True, True)       # Cross; generate both

    # Fallback to wall-band runs (0/door)
    hb = _run_len(mask, r, c, "h", only_zero=False, THRESH=THRESH)
    vb = _run_len(mask, r, c, "v", only_zero=False, THRESH=THRESH)
    if vb >= THRESH and hb < THRESH:
        return (False, True)
    if hb >= THRESH and vb < THRESH:
        return (True, False)
    if hb >= THRESH and vb >= THRESH:
        return (True, True)

    # Fallback to room adjacency rules (avoid extreme fragmentation)
    # Both sides rooms -> clear separator
    if (left and right) and not (up and down):
        return (False, True)
    if (up and down) and not (left and right):
        return (True, False)

    # Rooms on all sides -> cross intersection
    if (up and down) and (left and right):
        return (True, True)

    # Single-side room
    if (left ^ right):
        return (False, True)
    if (up ^ down):
        return (True, False)

    # Any-side fallback
    if (left or right) and not (up or down):
        return (False, True)
    if (up or down) and not (left or right):
        return (True, False)

    return (False, False)



def _build_dir_maps(mask):
    rows, cols = len(mask), len(mask[0])
    dirH = [[False]*cols for _ in range(rows)]
    dirV = [[False]*cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            h, v = _choose_orientation(mask, r, c)
            dirH[r][c] = h
            dirV[r][c] = v
    return dirH, dirV


# ---- Wall placement helpers ----
def _place_wall_segments_along_z(ctrl, *, x_const, z0, z1, door_z_list, wall_t, wall_h, door_w):
    """Place wall segments along Z with precise door openings."""
    EPS = 1e-6
    segs = SegSet()
    segs.add(z0, z1)
    
    # Cut door openings (door_w) at each door position
    for dz in door_z_list:
        segs.cut(dz - door_w/2.0, dz + door_w/2.0)
    
    # Place remaining wall segments
    for a, b in segs:
        if b - a <= EPS:
            continue
        oid=add_cube(ctrl,
                pos={"x": x_const, "y": wall_h/2, "z": (a+b)/2},
                scale={"x": wall_t, "y": wall_h, "z": (b-a)},
                color=GRAY)
          # Key: apply material to a white base, not GRAY
        if WALL_USE_MATERIAL:
            apply_material_to_special_primitive(ctrl,
                                                object_id=oid,
                                                model_name="prim_cube",
                                                material_name=WALL_MATERIAL_NAME,
                                                material_library=WALL_MATERIAL_LIB,
                                                model_library="models_special.json")



def _place_wall_segments_along_x(ctrl, *, z_const, x0, x1, door_x_list, wall_t, wall_h, door_w):
    """Place wall segments along X with precise door openings."""
    EPS = 1e-6
    segs = SegSet()
    segs.add(x0, x1)
    
    # Cut door openings (door_w) at each door position
    for dx in door_x_list:
        segs.cut(dx - door_w/2.0, dx + door_w/2.0)
    
    # Place remaining wall segments
    for a, b in segs:
        if b - a <= EPS:
            continue
        oid=add_cube(ctrl,
                pos={"x": (a+b)/2, "y": wall_h/2, "z": z_const},
                scale={"x": (b-a), "y": wall_h, "z": wall_t},
                color=GRAY)
        # oid = add_cube(ctrl,
        #        pos={...},
        #        scale={...},
        #        color=GRAY)  # Key: apply material to a white base, not GRAY
        if WALL_USE_MATERIAL:
            apply_material_to_special_primitive(ctrl,
                                                object_id=oid,
                                                model_name="prim_cube",
                                                material_name=WALL_MATERIAL_NAME,
                                                material_library=WALL_MATERIAL_LIB,
                                                model_library="models_special.json")


def _add_door_filler(ctrl, *, x, z, wall_h, wall_t, door_w, horizontal, bottom_y=FILLER_BOTTOM_Y):
    filler_h = max(0.0, wall_h - bottom_y)
    if filler_h <= 1e-4:
        return None  # Return None if no filler created

    thickness = wall_t if USE_WALL_THICKNESS else FIXED_THICKNESS
    width = door_w if USE_DOOR_WIDTH else FIXED_WIDTH

    # Keep axis orientation consistent with prior logic
    scale = {"x": thickness, "y": filler_h, "z": width} if horizontal else \
            {"x": width, "y": filler_h, "z": thickness}

    pos_y = bottom_y + filler_h / 2.0

    # Key: use add_cube (prim_cube + set_color(GRAY)) to match wall appearance
    oid=add_cube(ctrl,
             pos={"x": x, "y": pos_y, "z": z},
             scale=scale,
             color=GRAY,
             rot={"x": 0, "y": 0, "z": 0},
             kin=True)
    # oid = add_cube(ctrl,
    #            pos={"x": x, "y": pos_y, "z": z},
    #            scale=scale,
    #            color=GRAY)
    if WALL_USE_MATERIAL:
        apply_material_to_special_primitive(ctrl, oid,
                                            model_name="prim_cube",
                                            material_name=WALL_MATERIAL_NAME,
                                            material_library=WALL_MATERIAL_LIB)
    
    return oid  # Return the filler object ID



def _add_door_visual(ctrl, *, x, z, wall_h, wall_t, door_w, horizontal, color=None):
    """
    If a custom door record.json is provided, load the real model; otherwise fall back to a colored cube.
    Returns (door_id, filler_id), where filler_id may be None.
    """
    door_id = None
    filler_id = None
    
    try:
        if DOOR_RECORD_JSON and DOOR_RECORD_JSON.exists():
            rec = ModelRecord(json.loads(DOOR_RECORD_JSON.read_text()))
            model_name = rec.name
            model_url = rec.get_url()
            rot_y = 90 if horizontal else 0

            # Vertical placement: if pivot is at bottom, use half door height; if centered, use 0
            door_h = DOOR_H  # Could use wall_h, but 2.1m DOOR_H is more realistic
            y = (door_h/2) if DOOR_Y_BY_PIVOT_BOTTOM else 0.0

            oid = ctrl.get_unique_id()
            door_id = oid
            ctrl.communicate([ctrl.get_add_material("plastic_vinyl_glossy_white", library="materials_med.json")])
            cmds = [
                { "$type": "add_object",
                  "name": model_name,
                  "url": model_url,
                  "id": oid,
                  "position": {"x": x, "y": y, "z": z},
                  "rotation": {"x": 0, "y": rot_y, "z": 0},
                  "category": "door" },
                { "$type": "set_kinematic_state", "id": oid, "is_kinematic": True, "use_gravity": False },
                { "$type": "scale_object", "id": oid,
                  "scale_factor": {"x": DOOR_UNIFORM_SCALE, "y": DOOR_UNIFORM_SCALE, "z": 0.1} },
                
                { "$type": "set_visual_material", "id": oid,"material_name": "plastic_vinyl_glossy_white", "object_name":"default",},
                { "$type": "set_color", "id": oid, "color": color }
            ]
            # Key: do not set_color on real models to avoid overriding materials
            ctrl.communicate(cmds)
            filler_id = _add_door_filler(ctrl,
                    x=x, z=z,
                    wall_h=wall_h,
                    wall_t=wall_t,
                    door_w=door_w,
                    horizontal=horizontal,
                    bottom_y=FILLER_BOTTOM_Y)
            return (door_id, filler_id)
    except Exception as e:
        print(f"[WARN] Failed to add custom door model, fallback to cube: {e}")

    # ---- Fallback: keep legacy logic, use a cube door (allow coloring) ----
    if horizontal:
        scale = {"x": wall_t, "y": wall_h, "z": door_w}
    else:
        scale = {"x": door_w, "y": wall_h, "z": wall_t}
    door_color = color if color is not None else DOOR_COLOR
    door_id = add_cube(ctrl, pos={"x": x, "y": wall_h/2, "z": z}, scale=scale, color=door_color)
    # ... after placing the door (custom model or cube) ...
    
    return (door_id, filler_id)
    
def _build_rect_floors_by_room(ctrl: Controller,
                               mask: List[List[int]],
                               cell: float,
                               floor_y: float,
                               floor_thickness: float,
                               wall_t: float,
                               floor_to_wall_face: bool):
    """
    For each room id (1..99), build ONE rectangular floor that spans the room bbox.
    """
    rows, cols = len(mask), len(mask[0])
    half_th = floor_thickness / 2.0

    # 1) Collect bbox per room id (row/col are inclusive)
    bboxes: Dict[int, Dict[str, int]] = {}
    for r in range(rows):
        for c in range(cols):
            v = mask[r][c]
            if is_room(v):
                b = bboxes.get(v)
                if b is None:
                    b = {"r0": r, "r1": r, "c0": c, "c1": c}
                else:
                    b["r0"] = min(b["r0"], r); b["r1"] = max(b["r1"], r)
                    b["c0"] = min(b["c0"], c); b["c1"] = max(b["c1"], c)
                bboxes[v] = b

    # 2) Place one floor per room bbox (optionally shrink to wall inner face)
    for rid, b in bboxes.items():
        r0, r1, c0, c1 = b["r0"], b["r1"], b["c0"], b["c1"]

        # Boundary to world: use cell boundary coordinates without extra expansion
        x0 = x_boundary(r0,     cell, rows)
        x1 = x_boundary(r1 + 1, cell, rows)
        z0 = z_boundary(c0,     cell, cols)
        z1 = z_boundary(c1 + 1, cell, cols)

        # Optional: inset floor to wall inner face to avoid overlap
        if floor_to_wall_face:
            inset = wall_t / 2.0
            x0 += inset; x1 -= inset
            z0 += inset; z1 -= inset

        x_len = max(0.0, x1 - x0)
        z_len = max(0.0, z1 - z0)
        if x_len <= 1e-6 or z_len <= 1e-6:
            continue  # Too thin/empty, skip

        pos = {"x": (x0 + x1) / 2.0, "y": floor_y + half_th, "z": (z0 + z1) / 2.0}
        scale = {"x": x_len+1, "y": floor_thickness, "z": z_len+1}

        # Create prim_cube floor (white base, then apply material)
        oid = add_cube(ctrl, pos=pos, scale=scale, color=WHITE, kin=True)
        if FLOOR_USE_MATERIAL:
            apply_material_to_special_primitive(ctrl,
                                                object_id=oid,
                                                model_name="prim_cube",
                                                material_name=FLOOR_MATERIAL_NAME,
                                                material_library=FLOOR_MATERIAL_LIB,
                                                model_library="models_special.json")
        
    


# ---- Main wall generation function ----
def _build_continuous_walls_with_precise_doors(ctrl, mask, rows, cols, cell, wall_t, wall_h, door_w, door_colors=None):
    """Build continuous walls with precise door openings.
    Returns (door_object_ids, filler_object_ids, door_positions, filler_positions) for teleport tracking.
    """
    
    # Build direction maps
    dirH, dirV = _build_dir_maps(mask)
    used_doors = set()  # Avoid duplicate door visuals
    
    # Track door/filler object IDs and positions
    door_object_ids = []
    filler_object_ids = []
    door_positions = {}  # {object_id: position}
    filler_positions = {}  # {object_id: position}
    
    # Generate horizontal walls (row scan)
    for r in range(rows):
        c = 0
        while c < cols:

            if not _is_band(mask[r][c]):
                c += 1
                continue

            start_c = c
            door_cols = []
            seenH = False

            # Consume the full wall-band run and track if any dirH=True appears
            while c < cols and _is_band(mask[r][c]):
                if dirH[r][c]:
                    seenH = True
                if is_door(mask[r][c]):
                    door_cols.append(c)
                c += 1

            end_c = c - 1

            # Segment eligibility: output horizontal wall only if dirH=True appears in the run
            if not seenH:
                continue
            

            # Compute world-space span (extend half-cell on both ends to cover corners)
            x_const = x_center(r, cell, rows)
            # Check whether adjacent cell is open (out of bounds or -1)
            left_is_open  = (start_c - 1 < 0)   or (mask[r][start_c - 1] < 0)
            right_is_open = (end_c + 1 >= cols) or (mask[r][end_c + 1]   < 0)

            # Check whether adjacent cell is a room
            left_has_room  = (start_c - 1 >= 0)  and is_room(mask[r][start_c - 1])
            right_has_room = (end_c + 1   <  cols) and is_room(mask[r][end_c + 1])

            # Endpoints trim condition: open or room
            left_trim  = left_is_open  or left_has_room
            right_trim = right_is_open or right_has_room

            # If trim, shrink by half-cell; otherwise expand by half-cell to align corners
            l_ext = (-cell / 2.0) if left_trim  else (cell / 2.0)
            r_ext = (-cell / 2.0) if right_trim else (cell / 2.0)

            z0 = z_boundary(start_c,   cell, cols) - l_ext
            z1 = z_boundary(end_c + 1, cell, cols) + r_ext

            door_z_list = [z_center(dc, cell, cols) for dc in door_cols]
            
            # Place wall segments (cut door openings precisely)
            _place_wall_segments_along_z(ctrl,
                                        x_const=x_const, z0=z0, z1=z1,
                                        door_z_list=door_z_list,
                                        wall_t=wall_t, wall_h=wall_h, door_w=door_w)
            
            # Place door visuals
            for dc in door_cols:
                if (r, dc) not in used_doors:
                    # Get door color from mapping if available
                    door_id = mask[r][dc]  # Door ID from mask
                    door_color = None
                    if door_colors and door_id in door_colors:
                        door_color = door_colors[door_id]
                    
                    door_x = x_const
                    door_z = z_center(dc, cell, cols)
                    door_obj_id, filler_obj_id = _add_door_visual(ctrl,
                                    x=door_x, z=door_z,
                                    wall_h=wall_h, wall_t=wall_t, door_w=door_w,
                                    horizontal=True, color=door_color)
                    
                    # Record door object ID and position
                    if door_obj_id is not None:
                        door_object_ids.append(door_obj_id)
                        door_h = DOOR_H if DOOR_Y_BY_PIVOT_BOTTOM else 0.0  # Set Y based on pivot
                        door_y = (door_h/2) if DOOR_Y_BY_PIVOT_BOTTOM else 0.0
                        door_positions[door_obj_id] = {"x": door_x, "y": door_y, "z": door_z}
                    
                    # Record filler object ID and position
                    if filler_obj_id is not None:
                        filler_object_ids.append(filler_obj_id)
                        filler_y = FILLER_BOTTOM_Y + (wall_h - FILLER_BOTTOM_Y) / 2.0
                        filler_positions[filler_obj_id] = {"x": door_x, "y": filler_y, "z": door_z}
                    
                    used_doors.add((r, dc))
    
    # Generate vertical walls (column scan)
    for c in range(cols):
        r = 0
        while r < rows:
            # Start segment if this is a wall band (0 or door)
            if not _is_band(mask[r][c]):
                r += 1
                continue

            start_r = r
            door_rows = []
            seenV = False

            # Consume the full wall-band run and track if any dirV appears
            while r < rows and _is_band(mask[r][c]):
                if dirV[r][c]:
                    seenV = True
                if is_door(mask[r][c]):
                    door_rows.append(r)
                r += 1
            end_r = r - 1

            # Segment eligibility: output vertical wall only if dirV appears in the run
            if not seenV:
                continue

            z_const = z_center(c, cell, cols)
            # Check whether adjacent cell is open (out of bounds or -1)
            top_is_open    = (start_r - 1 < 0)   or (mask[start_r - 1][c] < 0)
            bottom_is_open = (end_r + 1 >= rows) or (mask[end_r + 1][c]   < 0)

            # Check whether adjacent cell is a room
            top_has_room    = (start_r - 1 >= 0)  and is_room(mask[start_r - 1][c])
            bottom_has_room = (end_r + 1   <  rows) and is_room(mask[end_r + 1][c])

            # Endpoints trim condition: open or room
            top_trim    = top_is_open    or top_has_room
            bottom_trim = bottom_is_open or bottom_has_room

            # If trim, shrink by half-cell; otherwise expand by half-cell
            t_ext = (-cell / 2.0) if top_trim    else (cell / 2.0)
            b_ext = (-cell / 2.0) if bottom_trim else (cell / 2.0)

            x0 = x_boundary(start_r,   cell, rows) - t_ext
            x1 = x_boundary(end_r + 1, cell, rows) + b_ext

            door_x_list = [x_center(dr, cell, rows) for dr in door_rows]
            
            # Place wall segments (cut door openings precisely)
            _place_wall_segments_along_x(ctrl,
                                        z_const=z_const, x0=x0, x1=x1,
                                        door_x_list=door_x_list,
                                        wall_t=wall_t, wall_h=wall_h, door_w=door_w)
            
            # Place door visuals
            for dr in door_rows:
                if (dr, c) not in used_doors:
                    # Get door color from mapping if available
                    door_id = mask[dr][c]  # Door ID from mask
                    door_color = None
                    if door_colors and door_id in door_colors:
                        door_color = door_colors[door_id]
                    
                    door_x = x_center(dr, cell, rows)
                    door_z = z_const
                    door_obj_id, filler_obj_id = _add_door_visual(ctrl,
                                    x=door_x, z=door_z,
                                    wall_h=wall_h, wall_t=wall_t, door_w=door_w,
                                    horizontal=False, color=door_color)
                    
                    # Record door object ID and position
                    if door_obj_id is not None:
                        door_object_ids.append(door_obj_id)
                        door_h = DOOR_H if DOOR_Y_BY_PIVOT_BOTTOM else 0.0  # Set Y based on pivot
                        door_y = (door_h/2) if DOOR_Y_BY_PIVOT_BOTTOM else 0.0
                        door_positions[door_obj_id] = {"x": door_x, "y": door_y, "z": door_z}
                    
                    # Record filler object ID and position
                    if filler_obj_id is not None:
                        filler_object_ids.append(filler_obj_id)
                        filler_y = FILLER_BOTTOM_Y + (wall_h - FILLER_BOTTOM_Y) / 2.0
                        filler_positions[filler_obj_id] = {"x": door_x, "y": filler_y, "z": door_z}
                    
                    used_doors.add((dr, c))

    # Return all collected door/filler info
    return (door_object_ids, filler_object_ids, door_positions, filler_positions)




# ---------------- IO ----------------
def load_mask(path: Path) -> List[List[int]]:
    text = path.read_text(encoding="utf-8").strip()
    if path.suffix.lower() == ".json":
        data = json.loads(text)
        assert isinstance(data, list) and isinstance(data[0], list)
        return data
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line: continue
        rows.append([int(x) for x in line.split()])
    w = len(rows[0])
    assert all(len(r) == w for r in rows), "Mask must be a rectangular matrix."
    return rows

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask_path", type=str, default="")
    ap.add_argument("--cell_size", type=float, default=1.0)
    ap.add_argument("--wall_thickness", type=float, default=0.15)
    ap.add_argument("--wall_height", type=float, default=3.0)
    ap.add_argument("--door_width", type=float, default=0.8)
    ap.add_argument("--floor_y", type=float, default=0.0)
    ap.add_argument("--floor_thickness", type=float, default=0.05)
    ap.add_argument("--floor_to_wall_face", action="store_true")
    ap.add_argument("--add_door_mesh", action="store_true")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--out", type=str, default="./tdw_mask_out")
    args = ap.parse_args()

    # default mask unchanged
    DEFAULT_MASK = [
        [-1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [-1, -1, -1,  0,  1,  1,  1,  1,  1,  1,  0,  2,  2,  2,  2,  2,  2,  2,  2,  0],
        [-1, -1, -1,  0,  1,  1,  1,  1,  1,  1, 100, 2,  2,  2,  2,  2,  2,  2,  2,  0],
        [-1, -1, -1,  0,  1,  1,  1,  1,  1,  1,  0,  2,  2,  2,  2,  2,  2,  2,  2,  0],
        [-1, -1, -1,  0,  1,  1,  1,  1,  1,  1,  0,  2,  2,  2,  2,  2,  2,  2,  2,  0],
        [-1, -1, -1,  0,  1,  1,  1,  1,  1,  1,  0,  2,  2,  2,  2,  2,  2,  2,  2,  0],
        [-1, -1, -1,  0,  1,  1,  1,  1,  1,  1,  0,  2,  2,  2,  2,  2,  2,  2,  2,  0],
        [ 0,  0,  0,  0,  0,  0,  0, 101, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  3,  3,  3,  3,  3,  3,  3,  3,  0,  4,  4,  4,  4,  4,  4,  4,  4,  4,  0],
        [ 0,  3,  3,  3,  3,  3,  3,  3,  3, 102, 4,  4,  4,  4,  4,  4,  4,  4,  4,  0],
        [ 0,  3,  3,  3,  3,  3,  3,  3,  3,  0,  4,  4,  4,  4,  4,  4,  4,  4,  4,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]
    ]

    mask = DEFAULT_MASK if not args.mask_path else load_mask(Path(args.mask_path))
    rows, cols = len(mask), len(mask[0])
    print(f"[INFO] Mask loaded: {rows}x{cols}")

    # canvas from compressed size
    _, _, _, _, kept_rows, kept_cols, nrows_eff, ncols_eff = build_compression_maps(mask)
    extent_x = max(40, int(cols * args.cell_size * 2))
    extent_z = max(40, int(rows * args.cell_size * 2))

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    ctrl = Controller(launch_build=True)


    cap = ImageCapture(avatar_ids=[], path=str(out_dir), pass_masks=["_img"], png=True)
    ctrl.add_ons.append(cap)
    ctrl.communicate([
        {"$type": "set_screen_size", "width": 768, "height": 768},
        TDWUtils.create_empty_room(extent_x, extent_z)
    ])

    # Build scene and get door/filler tracking info (though not used in main)
    try:
        door_ids, filler_ids, door_pos, filler_pos = build_scene(ctrl=ctrl,
                    mask=mask,
                    cell=args.cell_size,
                    wall_t=args.wall_thickness,
                    wall_h=args.wall_height,
                    door_w=args.door_width,
                    add_door_mesh=args.add_door_mesh,
                    floor_y=args.floor_y,
                    floor_thickness=args.floor_thickness,
                    floor_to_wall_face=args.floor_to_wall_face,
                    debug=args.debug)
        print(f"[INFO] Scene built with {len(door_ids)} doors and {len(filler_ids)} fillers")
    except TypeError:
        # Handle case where build_scene doesn't return the new values (backwards compatibility)
        print("[WARN] Using legacy build_scene without door tracking")
        build_scene(ctrl=ctrl,
                    mask=mask,
                    cell=args.cell_size,
                    wall_t=args.wall_thickness,
                    wall_h=args.wall_height,
                    door_w=args.door_width,
                    add_door_mesh=args.add_door_mesh,
                    floor_y=args.floor_y,
                    floor_thickness=args.floor_thickness,
                    floor_to_wall_face=args.floor_to_wall_face,
                    debug=args.debug)

    cam_h = max(12.0, max(ncols_eff, nrows_eff) * args.cell_size * 1.5)
    top_cam = ThirdPersonCamera(avatar_id="top_down",
                                position={"x": 0.0, "y": cam_h, "z": 0.0},
                                look_at={"x": 0.0, "y": 0.0, "z": 0.0},
                                field_of_view=60)
    ctrl.add_ons.append(top_cam)
    cap.avatar_ids = ["top_down"]; cap.set(frequency="always"); ctrl.communicate([]); cap.set(frequency="never")

    (out_dir / "layout.json").write_text(json.dumps({
        "ncols_eff": ncols_eff, "nrows_eff": nrows_eff, "cell_size": args.cell_size,
        "wall_thickness": args.wall_thickness, "wall_height": args.wall_height, "door_width": args.door_width
    }, indent=2), encoding="utf-8")
    print(f"[OK] Saved images and layout.json to: {out_dir.resolve()}")
    ctrl.communicate({"$type": "terminate"})

if __name__ == "__main__":
    main()
