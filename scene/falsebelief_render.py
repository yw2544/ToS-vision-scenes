#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Render false-belief scenes from existing falsebelief_exp.json files.

Workflow per run:
1) Load runXX/falsebelief_exp.json (includes mask/objects/agent).
2) Use EnhancedMask2Scene TDW rendering, but objects are taken directly from meta, with no random generation/validation/viewpoint generation.
3) Write render output to a temp directory, then move all pngs back to runXX with the original relative paths and a suffix (default _fbexp) to avoid overwriting.

Usage:
python generate_pipeline.py \
  --config config.yaml \
  --falsebelief-exp \
  --fb-runs-root ./tos_dataset_dir \
  --fb-runs 0-24 \
  --fb-meta-name falsebelief_exp.json \
  --fb-mod-type auto \
  --fb-render \
  --fb-suffix _fbexp \
  --fb-port 1074 \
  --skip-validation
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List
import tempfile
import numpy as np

from dataclasses import dataclass

from scene.mask2scene_enhanced import EnhancedMask2Scene
from multi_room_generator.object_generator import PlacedObject

# default_rotation lookup (shared with false-belief flow)
_DEFAULT_YAW_MAP: Dict[str, float] = {}

def _load_default_yaw_map() -> None:
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


@dataclass
class Agent:
    object_id: int
    name: str
    pos: Dict[str, float]
    rot: Dict[str, float]
    size: tuple
    scale: float
    color: Dict[str, float] | None
    room_id: int


def _load_meta(meta_path: Path) -> Dict:
    return json.loads(meta_path.read_text())


def _write_mask_from_meta(meta: Dict) -> Path:
    if "mask" not in meta:
        raise ValueError("meta missing 'mask' field")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w", encoding="utf-8") as tmp:
        json.dump(meta["mask"], tmp)
        path = Path(tmp.name)
    return path


def _load_record_mapping(paths: List[str], base_dir: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for path in paths:
        if not path:
            continue
        p = Path(path)
        if not p.is_absolute():
            p = (base_dir / p).resolve()
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        if isinstance(data, dict):
            items = data.items()
        elif isinstance(data, list):
            items = [(d.get("name"), d) for d in data if isinstance(d, dict)]
        else:
            items = []
        for k, v in items:
            if not isinstance(v, dict):
                continue
            name = v.get("name", k)
            model_name = v.get("model_name", name)
            record = v.get("record")
            if not record:
                continue
            rp = Path(record)
            if not rp.is_absolute():
                rp = (p.parent / rp).resolve()
            if not rp.exists():
                continue
            # Map both model and name keys; prefer model to match meta "model"
            if model_name:
                mapping[model_name] = str(rp)
            if name:
                mapping[name] = str(rp)
    return mapping


class FBExpMask2Scene(EnhancedMask2Scene):
    """Preset-object mode; skip random generation/validation/viewpoints/metadata updates."""

    def __init__(self, meta_override: Dict, fb_suffix: str = "_fbexp", custom_records: Dict[str, str] = None, **kwargs):
        self.meta_override = meta_override
        self.fb_suffix = fb_suffix
        # False-belief rendering does not need agent segmentation debug images
        self.save_agent_seg_debug = False
        # total_objects only satisfies base-class argument checks
        total_objects = len(meta_override.get("objects", [])) or 1
        if "total_objects" in kwargs:
            kwargs.pop("total_objects")
        self._custom_records = custom_records or {}
        super().__init__(total_objects=total_objects, **kwargs)

    # Override: use meta objects directly; skip generation/validation
    def _generate_and_validate_objects_with_retry(self) -> bool:
        try:
            objs: List[PlacedObject] = []
            for o in self.meta_override.get("objects", []):
                name_lower = str(o.get("name", "")).lower()
                model_lower = str(o.get("model", "")).lower()
                # Skip doors: doors are generated from the mask in mask2scene_enhanced; do not add twice
                if "door" in name_lower or "door" in model_lower:
                    continue
                pos = o.get("pos", {})
                rot = o.get("rot", {})
                attrs = o.get("attributes", {})
                # Add default_rotation for TDW (keep base rotation in meta)
                rot_y = rot.get("y", 0.0)
                rot_y = (rot_y + _get_default_yaw_for_model(o.get("model", ""))) % 360
                obj_color = o.get("color")
                if obj_color is None:
                    obj_color = attrs.get("color")
                if isinstance(obj_color, dict):
                    obj_color = obj_color.get("name")
                obj = PlacedObject(
                    object_id=int(o.get("object_id", len(objs) + 1)),
                    model=o.get("model", ""),
                    name=o.get("name", ""),
                    pos={"x": pos.get("x", 0.0), "y": pos.get("y", 0.0), "z": pos.get("z", 0.0)},
                    rot={"x": rot.get("x", 0.0), "y": rot_y, "z": rot.get("z", 0.0)},
                    size=tuple(o.get("size", [1.0, 1.0])),
                    scale=float(attrs.get("scale", 1.0)),
                    color=obj_color,
                    room_id=int(attrs.get("room_id", 1)),
                    has_orientation=bool(attrs.get("has_orientation", False)),
                    orientation=attrs.get("orientation"),
                    is_custom_model=False,
                    custom_config=None,
                    model_config=o.get("model_config"),
                )
                # If a record exists in custom_models, load as custom model
                rec = self._custom_records.get(obj.model)
                if rec:
                    obj.is_custom_model = True
                    obj.custom_config = {"record": rec}
                objs.append(obj)

            agent_obj = None
            for cam in self.meta_override.get("cameras", []):
                if cam.get("id") == "agent":
                    p = cam.get("position", {})
                    r = cam.get("rotation", {})
                    agent_obj = Agent(
                        object_id=-1,
                        name="agent",
                        pos={"x": p.get("x", 0.0), "y": p.get("y", 0.0), "z": p.get("z", 0.0)},
                        rot={"x": 0.0, "y": r.get("y", 0.0), "z": 0.0},
                        size=(0.5, 1.7),
                        scale=1.0,
                        color=None,
                        room_id=1
                    )
                    break

            class _StubGen:
                def __init__(self, all_objects, agent_obj):
                    self.all_objects = all_objects
                    self._agent = agent_obj
                    self.min_distance = 1.0

                def get_all_objects(self):
                    return self.all_objects

                def export_summary(self):
                    return {
                        "total_objects": len(self.all_objects),
                        "rooms_with_objects": len(set(o.room_id for o in self.all_objects)),
                        "used_categories": []
                    }

                def get_agent(self):
                    return self._agent

            self.object_generator = _StubGen(objs, agent_obj)
            return True
        except Exception as e:
            print(f"[ERROR] preload objects failed: {e}")
            return False

    # Override: no viewpoint generation and no image updates
    def _generate_task_viewpoints(self) -> bool:
        return True

    def _capture_task_viewpoint_images(self) -> bool:
        return True

    def _update_metadata_with_images(self) -> bool:
        return True

    def _generate_orientation_instruction(self) -> bool:
        return True


def _rename_and_move_pngs(src_root: Path, dst_root: Path, suffix: str):
    norm_suffix = suffix if suffix.startswith("_") else f"_{suffix}" if suffix else ""
    for p in src_root.rglob("*.png"):
        rel = p.relative_to(src_root)
        stem = p.stem
        # Do not keep debug images
        if "agent_seg_debug" in stem:
            continue
        new_name = stem + norm_suffix + p.suffix
        dst_path = dst_root / rel.parent / new_name
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(p), str(dst_path))


def render_one_run(run_dir: Path, fb_meta_name: str, fb_suffix: str,
                   port: int, config: Dict,
                   builtin_models_path: str = None,
                   custom_models_path: str = None,
                   door_record_path: str = None) -> bool:
    base_dir = run_dir.parent.parent  # multi_room_gen root
    record_map = _load_record_mapping([custom_models_path, door_record_path], base_dir)
    fb_meta_path = run_dir / fb_meta_name
    if not fb_meta_path.exists():
        print(f"[WARN] skip {run_dir.name}: {fb_meta_name} not found")
        return False
    meta = _load_meta(fb_meta_path)
    mask_tmp = _write_mask_from_meta(meta)

    temp_out = run_dir / ".fbexp_tmp"
    if temp_out.exists():
        shutil.rmtree(temp_out)
    temp_out.mkdir(parents=True, exist_ok=True)

    seed = meta.get("run_seed", meta.get("seed", 0))
    scene_cfg = (config or {}).get("scene_generation", {})
    cell_size = scene_cfg.get("cell_size", 1.0)
    wall_thickness = scene_cfg.get("wall_thickness", 0.01)
    wall_height = scene_cfg.get("wall_height", 2.0)
    door_width = scene_cfg.get("door_width", 0.6)
    overall_scale = scene_cfg.get("overall_scale", 1.0)
    enable_gravity_fix = scene_cfg.get("enable_gravity_fix", True)
    physics_settle_time = scene_cfg.get("physics_settle_time", 0.2)
    with_ray = scene_cfg.get("with_ray", False)

    gen = FBExpMask2Scene(
        meta_override=meta,
        fb_suffix=fb_suffix,
        custom_records=record_map,
        mask_path=str(mask_tmp),
        output_dir=str(temp_out),
        cell_size=cell_size,
        wall_thickness=wall_thickness,
        wall_height=wall_height,
        door_width=door_width,
        seed=seed,
        total_objects=1,  # unused in preload
        port=port,
        x_offset=0.0,
        z_offset=0.0,
        with_ray=with_ray,
        overall_scale=overall_scale,
        enable_gravity_fix=enable_gravity_fix,
        physics_settle_time=physics_settle_time,
        fix_object_n=None,
        object_mode="total",
        proportional_to_area=False,
        builtin_models_path=builtin_models_path,
        custom_models_path=custom_models_path,
        config={},
        run_seed=seed,
        custom_models_config=None,
        use_custom_models=True
    )

    ok = gen.generate_complete_scene()
    if ok:
        _rename_and_move_pngs(temp_out, run_dir, fb_suffix)
    shutil.rmtree(temp_out, ignore_errors=True)
    try:
        mask_tmp.unlink()
    except Exception:
        pass
    return ok


def render_fb_runs(root: Path, run_ids: List[int], fb_meta_name: str,
                   fb_suffix: str, port: int, config: Dict,
                   builtin_models_path: str = None,
                   custom_models_path: str = None,
                   door_record_path: str = None):
    for rid in run_ids:
        run_dir = root / f"run{rid:02d}"
        if not run_dir.exists():
            print(f"[WARN] skip run{rid:02d}: not found")
            continue
        print(f"[INFO] Rendering {run_dir.name} with {fb_meta_name} ...")
        ok = render_one_run(run_dir, fb_meta_name, fb_suffix, port, config,
                            builtin_models_path, custom_models_path, door_record_path)
        if ok:
            print(f"[INFO] ✅ run{rid:02d} done")
        else:
            print(f"[WARN] run{rid:02d} failed or skipped")


def main():
    ap = argparse.ArgumentParser(description="Render false-belief scenes from falsebelief_exp.json")
    ap.add_argument("--runs-root", required=True, help="Root dir containing runXX folders")
    ap.add_argument("--runs", type=str, default="0-24", help="Range 'start-end'")
    ap.add_argument("--fb-meta-name", type=str, default="falsebelief_exp.json")
    ap.add_argument("--fb-suffix", type=str, default="_fbexp")
    ap.add_argument("--port", type=int, default=1071)
    ap.add_argument("--builtin-models-path", type=str, default=None)
    ap.add_argument("--custom-models-path", type=str, default=None)
    args = ap.parse_args()

    try:
        s, e = [int(x) for x in args.runs.split("-")]
    except Exception:
        print(f"[ERROR] bad --runs '{args.runs}', expected start-end")
        return 1
    run_ids = list(range(s, e + 1))

    render_fb_runs(
        root=Path(args.runs_root).expanduser().resolve(),
        run_ids=run_ids,
        fb_meta_name=args.fb_meta_name,
        fb_suffix=args.fb_suffix,
        port=args.port,
        builtin_models_path=args.builtin_models_path,
        custom_models_path=args.custom_models_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

