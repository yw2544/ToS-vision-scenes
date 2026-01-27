#!/usr/bin/env python3
"""
Apply VAGEN ObjectModifier-style changes to an existing ToS meta_data.json.

Steps:
1) Load meta_data.json (ToS format).
2) Convert to a VAGEN Room (mask + objects on integer grid; orientation as 4-way vectors).
3) Run ObjectModifier(seed, n_changes, mod_type) to get modified room + change list.
4) Write a new meta_data JSON with updated object positions/rotations and a _fb_changes note.

Usage (example):
  python scene/apply_false_belief_to_meta.py \
    --input /Users/songshe/objaverse_import/ToS2/multi_room_gen/tos_dataset_1214_3room_100runs/run00/meta_data.json \
    --output /tmp/meta_data_fb_seed0.json \
    --seed 0 \
    --n-changes 2 \
    --mod-type auto

mod-type: auto | move | rotate
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np

from vagen.env.spatial.Base.tos_base.utils.room_modifier import ObjectModifier, ChangedObject
from vagen.env.spatial.Base.tos_base.core.room import Room
from vagen.env.spatial.Base.tos_base.core.object import Object as VObject, Agent as VAgent

_DEFAULT_YAW_MAP: Dict[str, float] = {}


def _load_default_yaw_map() -> None:
    """Load default_rotation.y for each model_name from custom/buildin model configs."""
    global _DEFAULT_YAW_MAP
    if _DEFAULT_YAW_MAP:
        return
    root = Path(__file__).resolve().parent.parent
    models_dir = root / "models"
    for fname in ("custom_models.json", "builtin_models.json"):
        path = models_dir / fname
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        for entry in data:
            model_name = entry.get("model_name")
            if not model_name:
                continue
            yaw = (entry.get("default_rotation") or {}).get("y")
            if yaw is None:
                continue
            _DEFAULT_YAW_MAP[model_name] = float(yaw)


def _get_default_yaw_for_model(model_name: str) -> float:
    if not _DEFAULT_YAW_MAP:
        _load_default_yaw_map()
    return float(_DEFAULT_YAW_MAP.get(model_name, 0.0))


def orientation_str_to_vec(ori: str) -> np.ndarray:
    ori_map = {
        "north": np.array([0, 1]),
        "east": np.array([1, 0]),
        "south": np.array([0, -1]),
        "west": np.array([-1, 0]),
    }
    return ori_map.get(str(ori).lower(), np.array([0, 1]))


def vec_to_angle(vec: np.ndarray) -> int:
    v = tuple(int(x) for x in vec)
    if v == (1, 0):
        return 90
    if v == (-1, 0):
        return 270
    if v == (0, -1):
        return 180
    return 0  # default north


def yaw_to_orientation(yaw: int) -> str:
    yaw = int(yaw) % 360
    return {
        0: "north",
        90: "east",
        180: "south",
        270: "west",
    }.get(yaw, "north")


def load_tos_meta(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _ensure_mask(mask: np.ndarray, min_row: int, min_col: int) -> np.ndarray:
    """Ensure mask is large enough; pad with 1s (room) if needed."""
    if mask.ndim != 2:
        mask = np.ones((max(20, min_row + 1), max(20, min_col + 1)), dtype=int)
    rows, cols = mask.shape
    need_rows = max(rows, min_row + 1)
    need_cols = max(cols, min_col + 1)
    if need_rows > rows or need_cols > cols:
        new_mask = np.zeros((need_rows, need_cols), dtype=int)
        new_mask[:rows, :cols] = mask
        # fill new areas as room (1)
        new_mask[rows:, :] = 1
        new_mask[:, cols:] = 1
        mask = new_mask
    return mask


def to_vagen_room(meta: Dict[str, Any]) -> Tuple[Room, VAgent]:
    raw_mask = meta.get("mask")
    mask = np.array(raw_mask) if raw_mask is not None else np.zeros((0, 0), dtype=int)
    offset = meta.get("offset", [0, 0])
    off_x, off_z = int(offset[0]), int(offset[1])

    min_r = 0
    min_c = 0
    objects = []
    for obj in meta.get("objects", []):
        name_lower = str(obj.get("name", "")).lower()
        attrs = obj.get("attributes", {}) or {}
        # 跳过门（door 不参与 false-belief 改动）
        if "door" in name_lower or "connected_rooms" in attrs:
            continue
        pos = obj.get("pos", {})
        has_ori = bool(attrs.get("has_orientation", False))
        rot_y = None
        if has_ori and "rot" in obj:
            rot_y = obj.get("rot", {}).get("y")
        if has_ori and rot_y is not None:
            ori_vec = orientation_str_to_vec(yaw_to_orientation(rot_y))
        else:
            ori_str = attrs.get("orientation", "north")
            ori_vec = orientation_str_to_vec(ori_str) if has_ori else np.array([0, 1])
        room_id = int(attrs.get("room_id", 1))
        # world -> mask indices (row from x, col from z) using offset (no rounding)
        # Example: x=-6,z=-1 with offset[7,8] -> row=1, col=7 maps to mask[1][7]
        row = int(pos.get("x", 0) + off_x)
        col = int(pos.get("z", 0) + off_z)
        min_r = max(min_r, row)
        min_c = max(min_c, col)
        v_name = str(obj.get("object_id", obj.get("name", "obj")))
        objects.append(
            VObject(
                name=v_name,
                pos=np.array([row, col], dtype=int),
                ori=ori_vec,
                room_id=room_id,
                has_orientation=has_ori,
            )
        )

    mask = _ensure_mask(mask, min_r, min_c)

    # If any object falls on a non-room cell (<=0), raise with details.
    for o in objects:
        r, c = int(o.pos[0]), int(o.pos[1])
        if r < 0 or c < 0 or r >= mask.shape[0] or c >= mask.shape[1]:
            raise ValueError(f"Object {o.name} maps out of mask bounds: (row={r}, col={c}), mask shape={mask.shape}")
        if mask[r, c] <= 0:
            raise ValueError(f"Object {o.name} maps to non-room cell: mask[{r},{c}]={mask[r,c]}")

    agent_obj = None
    agent_pos = None
    for cam in meta.get("cameras", []):
        if cam.get("id") == "agent":
            agent_pos = cam.get("position", {})
            break
    if agent_pos:
        row = float(agent_pos.get("x", 0) + off_x)
        col = float(agent_pos.get("z", 0) + off_z)
        agent_obj = VAgent(
            name="agent",
            pos=np.array([row, col], dtype=float),
            ori=np.array([0, 1], dtype=float),
            room_id=1,
            init_pos=np.array([row, col], dtype=float),
            init_ori=np.array([0, 1], dtype=float),
            init_room_id=1,
        )

    room = Room(objects=objects, mask=mask, name="fb_room")
    return room, agent_obj


def apply_changes_to_meta(meta: Dict[str, Any], modified_room: Room, changes: List[ChangedObject]) -> Dict[str, Any]:
    # Build a map from (name) to new pos/ori from modified_room
    name_to_state = {}
    for o in modified_room.objects:
        yaw_base = vec_to_angle(o.ori)
        name_to_state[o.name] = {
            "pos": {"row": int(o.pos[0]), "col": int(o.pos[1])},
            "ori_vec": o.ori.copy(),
            "yaw_base": yaw_base,
        }

    change_map = {c.name: c for c in changes}
    changed_names = set(change_map.keys())
    # cache old positions/rotations for reporting; key by object_id (string) if available
    old_pos = {}
    for o in meta.get("objects", []):
        key = str(o.get("object_id")) if o.get("object_id") is not None else o.get("name")
        yaw = o["rot"]["y"] if "rot" in o else None
        attrs = o.get("attributes", {}) or {}
        ori_from = attrs.get("orientation")
        if ori_from is None and yaw is not None:
            ori_from = yaw_to_orientation(yaw)
        old_pos[key] = {
            "x": o["pos"]["x"],
            "z": o["pos"]["z"],
            "yaw": yaw,
            "orientation": ori_from,
            "name": o.get("name"),
            "object_id": o.get("object_id"),
        }
    # Only update objects that were changed; others keep original pos/rot
    for obj in meta.get("objects", []):
        meta_key = str(obj.get("object_id")) if obj.get("object_id") is not None else obj.get("name")
        if meta_key not in changed_names:
            continue
        state = name_to_state.get(meta_key)
        if not state:
            continue
        # update position (keep y) using offset to map back to world coords
        off_x, off_z = meta.get("offset", [0, 0])
        # row = x + off_x, col = z + off_z  => x = row - off_x, z = col - off_z
        obj["pos"]["x"] = float(state["pos"]["row"] - off_x)
        obj["pos"]["z"] = float(state["pos"]["col"] - off_z)
        # update rotation/orientation only if this change includes orientation
        has_ori = obj.get("attributes", {}).get("has_orientation", False)
        change_entry = change_map.get(meta_key)
        if has_ori and change_entry and getattr(change_entry, "ori", False):
            yaw_offset = vec_to_angle(state["ori_vec"])
            # 写回基础朝向（不叠加 default_rotation）；实际放置时会叠加
            obj["rot"]["y"] = int(yaw_offset)
            obj["attributes"]["orientation"] = yaw_to_orientation(yaw_offset)

    # Add detailed change records with from/to positions (x,z) and orientation
    detailed_changes = []
    obj_by_name = {str(o.get("object_id")) if o.get("object_id") is not None else o.get("name"): o for o in meta.get("objects", [])}
    for c in changes:
        before = old_pos.get(c.name)
        after_obj = obj_by_name.get(c.name)
        entry = c.to_dict()
        # Preserve human-readable name/object_id
        if before:
            entry["name"] = before.get("name")
            entry["object_id"] = before.get("object_id")
        if before:
            entry["pos_from"] = {"x": before["x"], "z": before["z"]}
            if before.get("yaw") is not None:
                entry["yaw_from"] = before["yaw"]
                entry["orientation_from"] = before.get("orientation")
        if after_obj:
            entry["pos_to"] = {
                "x": after_obj["pos"]["x"],
                "z": after_obj["pos"]["z"],
            }
            if "rot" in after_obj and getattr(change_map.get(c.name), "ori", False):
                state = name_to_state.get(c.name, {})
                # yaw_to 仅在旋转改动时记录基础朝向（不含 default_rotation）
                entry["yaw_to"] = state.get("yaw_base")
                if entry["yaw_to"] is None and after_obj.get("attributes", {}).get("has_orientation"):
                    entry["yaw_to"] = vec_to_angle(state.get("ori_vec", np.array([0, 1])))
                if entry.get("yaw_to") is not None:
                    yaw_to = entry["yaw_to"] % 360
                    entry["orientation_to"] = yaw_to_orientation(yaw_to)
        detailed_changes.append(entry)

    meta["_fb_changes"] = detailed_changes
    return meta


def main():
    ap = argparse.ArgumentParser(description="Apply ObjectModifier-style changes to an existing ToS meta_data.json")
    ap.add_argument("--input", required=True, help="Path to meta_data.json")
    ap.add_argument("--output", required=True, help="Path to save modified meta_data.json")
    ap.add_argument("--seed", type=int, default=0, help="Seed for ObjectModifier")
    ap.add_argument("--n-changes", type=int, default=None,
                    help="[Ignored] Kept for compatibility. VAGEN behavior: always sample 1-3 via seed.")
    ap.add_argument("--mod-type", type=str, default="auto", choices=["auto", "move", "rotate"],
                    help="Force move/rotate or let modifier choose")
    args = ap.parse_args()

    meta_path = Path(args.input).expanduser().resolve()
    out_path = Path(args.output).expanduser().resolve()
    # If user passes a directory as output, write into that dir with same filename
    if out_path.is_dir():
        out_path = out_path / meta_path.name

    meta = load_tos_meta(meta_path)
    room, agent = to_vagen_room(meta)

    modifier = ObjectModifier(
        seed=args.seed,
        agent_pos=getattr(agent, "pos", None),
    )
    modified_room, changes = modifier.modify(room)

    meta_out = apply_changes_to_meta(meta, modified_room, changes)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(meta_out, indent=2))

    print(f"[INFO] Done. Changes: {[c.to_dict() for c in changes]}")
    print(f"[INFO] Saved modified meta to: {out_path}")


if __name__ == "__main__":
    main()

