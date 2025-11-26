import argparse
import json
import math
import os
import sys
import yaml
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np

# TDW imports
try:
    from tdw.controller import Controller
    from tdw.tdw_utils import TDWUtils
    from tdw.add_ons.third_person_camera import ThirdPersonCamera
    from tdw.add_ons.image_capture import ImageCapture
    from tdw.librarian import ModelLibrarian
    from tdw.librarian import HDRISkyboxLibrarian
except ImportError:
    print("Warning: TDW not installed or found.")
    Controller = None

# Local imports
from .room_analyzer import RoomAnalyzer
from .object_generator import ObjectGenerator, COLORS
from .door_handler import DoorHandler
from .camera_capture import CameraCapture
from .metadata_manager import MetadataManager
from .collinear_validator import CollinearValidator
from . import mask2scene
from .orientation_instruction_generator import OrientationInstructionGenerator
from .task_viewpoint_generator import generate_task_viewpoints, viewpoints_to_camera_metadata
from ..validation.ragen import PreRenderValidator

DEFAULT_CELL_SIZE = 1.0
DEFAULT_WALL_HEIGHT = 2.0
DEFAULT_WALL_THICKNESS = 0.01
DEFAULT_DOOR_WIDTH = 0.6
DEFAULT_MIN_OBJECT_DISTANCE = 1.42
DEFAULT_COLLINEAR_TOLERANCE = 0.8
DEFAULT_ENABLE_GRAVITY_FIX = True
DEFAULT_PHYSICS_SETTLE_TIME = 0.1

class SceneGenerator:
    LEGACY_VALIDATED_TASKS = [
        "rot",
        "rot_dual",
        "dir",
        "pov",
        "e2a",
        "fwd_loc",
        "bwd_loc",
        "false_belief",
        "fwd_fov",
        "bwd_nav",
        "bwd_pov",
    ]
    def __init__(self, config: Dict, mask_path: Path, output_dir: Path, seed: int):
        self.config = config
        self.mask_path = mask_path
        self.output_dir = output_dir
        self.seed = seed
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.scene_config = config.get('scene_generation', {})
        door_record_path = self.scene_config.get('door_record_path')
        if door_record_path:
            project_root = Path(__file__).resolve().parents[1]
            resolved_path = Path(door_record_path)
            if not resolved_path.is_absolute():
                resolved_path = project_root / resolved_path
            if hasattr(mask2scene, "set_door_record_path"):
                mask2scene.set_door_record_path(resolved_path)
            else:
                mask2scene.DOOR_RECORD_JSON = resolved_path
        self.obj_config = config.get('object_generation', {})
        debug_config = config.get('debug', {})
        self.object_debug_logs = debug_config.get('object_generator_logs', False)
        
        self.cell_size = DEFAULT_CELL_SIZE
        self.wall_height = DEFAULT_WALL_HEIGHT
        self.wall_thickness = DEFAULT_WALL_THICKNESS
        self.door_width = DEFAULT_DOOR_WIDTH
        self.overall_scale = float(self.scene_config.get('overall_scale', 1.0))
        self.enable_gravity_fix = DEFAULT_ENABLE_GRAVITY_FIX
        self.physics_settle_time = DEFAULT_PHYSICS_SETTLE_TIME
        self.scene_params = {
            "seed": self.seed,
            "cell_size": self.cell_size,
            "wall_height": self.wall_height,
            "wall_thickness": self.wall_thickness,
            "door_width": self.door_width,
            "overall_scale": self.overall_scale
        }
        self.door_handler: Optional[DoorHandler] = None
        self.metadata_path: Optional[Path] = None
        self.door_id_mapping: Dict[int, int] = {}
        self.door_positions: Dict[int, Dict[str, float]] = {}
        self.filler_positions: Dict[int, Dict[str, float]] = {}
        self.door_object_ids: List[int] = []
        self.filler_object_ids: List[int] = []
        self.scene_bounds: Optional[Tuple[float, float, float, float]] = None
        self.scene_width: Optional[float] = None
        self.scene_depth: Optional[float] = None
        self.captured_images: List[Dict] = []
        self.task_viewpoints: List[Dict] = []
        self.run_seed = self.seed
        
        self.np_random = np.random.default_rng(self.seed)
        
    def generate(self):
        if Controller is None:
            raise ImportError("TDW is required for scene generation.")

        print(f"[INFO] Starting scene generation for seed {self.seed}")
        
        # 1. Load Mask
        mask_data = mask2scene.load_mask(self.mask_path)
        self.mask_array = np.array(mask_data)
        
        # 2. Analyze Rooms
        self.room_analyzer = RoomAnalyzer(self.mask_array, self.cell_size)
        print(f"[INFO] Found {len(self.room_analyzer.rooms)} rooms and {len(self.room_analyzer.doors)} doors")
        self.door_handler = DoorHandler(self.room_analyzer, seed=self.seed)
        self.door_handler.validate_door_positions()
        print(self.door_handler.get_door_summary())
        
        # 3. Generate Objects (Virtual Placement)
        self.object_generator = ObjectGenerator(
            self.room_analyzer, 
            self.seed,
            min_distance=DEFAULT_MIN_OBJECT_DISTANCE,
            builtin_models_path=self.scene_config.get('builtin_models_path'),
            custom_models_path=self.scene_config.get('custom_models_path'),
            tolerance_width=DEFAULT_COLLINEAR_TOLERANCE,
            debug_enabled=self.object_debug_logs
        )
        print(f"[INFO] Object min distance set to {DEFAULT_MIN_OBJECT_DISTANCE}")
        print(f"[INFO] Collinear tolerance set to {DEFAULT_COLLINEAR_TOLERANCE}")
        
        # Initialize libraries
        self.model_lib = ModelLibrarian()
        self.object_generator.set_model_library(self.model_lib)
        
        # Generate
        mode = self.obj_config.get('mode', 'total')
        total_objects = self.obj_config.get('total_objects', 10)
        
        success = False
        if mode == 'fixed':
            fix_object_n = self.obj_config.get('fix_object_n')
            success = self.object_generator.generate_all_rooms(fix_object_n=fix_object_n)
        else:
            success = self.object_generator.generate_all_rooms(total_objects=total_objects)
            
        if not success:
            print("[WARN] Object generation had issues, proceeding anyway...")

        # 4. Pre-render Validation
        self._validation_metadata: Dict[str, Dict] = {}
        validator = PreRenderValidator(self.config)
        if getattr(validator, "is_enabled", None):
            validator_enabled = validator.is_enabled()
        else:
            validator_enabled = getattr(validator, "enabled", False)
        if validator_enabled:
            # Extract data for validation
            mask, objects_data, agent_data = self._extract_validation_data()
            is_valid, summary = validator.validate(mask, objects_data, agent_data, self.np_random)
            if not is_valid:
                print("[WARN] Pre-render validation failed. Scene might be invalid for tasks.")
            else:
                self._validation_metadata = summary or {}
        
        # 5. Generate legacy-compatible metadata before TDW init
        self._generate_base_metadata()
        self._generate_task_viewpoints()
        
        # 6. Initialize TDW Controller
        self.c = Controller(
            port=self.scene_config.get('port', 1071),
            launch_build=True
        )
        self.c.communicate({"$type": "set_render_quality", "render_quality": 5})
        
        # 7. Build Physical Scene (Walls, Doors, Floor)
        self._build_scene_structure()
        
        # 8. Place Objects
        self._place_objects()
        
        # 9. Add Skybox and Lighting
        self._setup_environment()
        
        # 10. Capture Data
        self._capture_scene_data()
        
        # 11. Capture task viewpoints if available
        if self.task_viewpoints:
            self._capture_task_viewpoint_images()
        
        # 12. Update metadata and export auxiliary files
        self._update_metadata_with_images()
        self._generate_orientation_instruction()
        self._export_additional_metadata()
        
        self.c.communicate({"$type": "terminate"})
        print(f"[INFO] Scene generation completed: {self.output_dir}")

    def _build_scene_structure(self):
        print("[INFO] Building scene structure...")
        
        mask_list = self.mask_array.tolist()
        if mask_list:
            rows = len(mask_list)
            cols = len(mask_list[0])
        else:
            rows = 0
            cols = 0
        scene_width = cols * self.cell_size
        scene_depth = rows * self.cell_size
        large_room_w = max(scene_width + 10, 40)
        large_room_d = max(scene_depth + 10, 40)
        self.scene_width = scene_width
        self.scene_depth = scene_depth
        offset_x = -(cols - 1) / 2.0 * self.cell_size
        offset_z = -(rows - 1) / 2.0 * self.cell_size
        min_x = offset_x
        max_x = offset_x + (cols - 1) * self.cell_size
        min_z = offset_z
        max_z = offset_z + (rows - 1) * self.cell_size
        self.scene_bounds = (min_x, max_x, min_z, max_z)
        
        self.c.communicate([
            TDWUtils.create_empty_room(large_room_w, large_room_d),
            {"$type": "set_screen_size", "width": 1024, "height": 1024}
        ])
        
        self._mask_offsets = (-(cols - 1) / 2.0 * self.cell_size, -(rows - 1) / 2.0 * self.cell_size)
        self._door_position_colors: Dict[Tuple[int, int], Dict[str, float]] = {}
        if self.door_handler:
            for processed_door in self.door_handler.processed_doors.values():
                mask_rc = self._world_to_mask_indices(processed_door.door_info.center)
                if mask_rc:
                    self._door_position_colors[mask_rc] = processed_door.color
        
        original_add_door_visual = getattr(mask2scene, "_add_door_visual", None)
        
        def position_aware_add_door_visual(ctrl, *, x, z, wall_h, wall_t, door_w, horizontal, color=None):
            mask_rc = self._world_to_mask_indices((x, z))
            if mask_rc and mask_rc in self._door_position_colors:
                color = self._door_position_colors[mask_rc]
            return original_add_door_visual(ctrl, x=x, z=z, wall_h=wall_h, wall_t=wall_t, door_w=door_w, horizontal=horizontal, color=color)
        
        if original_add_door_visual:
            mask2scene._add_door_visual = position_aware_add_door_visual
        
        try:
            door_ids, filler_ids, door_pos, filler_pos = mask2scene.build_scene(
                self.c,
                mask_list,
                cell=self.cell_size,
                wall_h=self.wall_height,
                wall_t=self.wall_thickness,
                door_w=self.door_width
            )
        finally:
            if original_add_door_visual:
                mask2scene._add_door_visual = original_add_door_visual
        
        self.door_positions = door_pos or {}
        self.filler_positions = filler_pos or {}
        self.door_object_ids = door_ids or []
        self.filler_object_ids = filler_ids or []
        
        if self.door_handler and hasattr(self.door_handler, 'update_with_physical_info'):
             self.door_handler.update_with_physical_info(door_ids, filler_ids, door_pos, filler_pos)

    def _world_to_mask_indices(self, world_pos: Tuple[float, float]) -> Optional[Tuple[int, int]]:
        if not hasattr(self, "_mask_offsets"):
            mask_list = self.mask_array.tolist()
            rows = len(mask_list)
            cols = len(mask_list[0]) if rows else 0
            self._mask_offsets = (-(cols - 1) / 2.0 * self.cell_size, -(rows - 1) / 2.0 * self.cell_size)
        offset_x, offset_z = self._mask_offsets
        try:
            x, z = world_pos
        except (TypeError, ValueError):
            return None
        col = round((x - offset_x) / self.cell_size)
        row = round((z - offset_z) / self.cell_size)
        return (row, col)

    def _ensure_door_id_mapping(self):
        if not self.door_handler:
            return
        if self.door_id_mapping:
            return
        for door_id, processed_door in self.door_handler.processed_doors.items():
            door_center = processed_door.door_info.center
            door_object_id = self._generate_door_id_from_coordinates(door_center, door_id)
            self.door_id_mapping[door_id] = door_object_id

    def _generate_door_id_from_coordinates(self, door_center: Tuple[float, float], door_id: int) -> int:
        position_hash = abs(hash((round(door_center[0], 2), round(door_center[1], 2)))) % 100
        door_object_id = 20000 + (door_id % 100) * 100 + position_hash
        if not hasattr(self, "_used_door_ids"):
            self._used_door_ids: set = set()
        while door_object_id in self._used_door_ids:
            door_object_id += 1
        self._used_door_ids.add(door_object_id)
        return door_object_id

    def _place_objects(self):
        print("[INFO] Adding generated objects to TDW scene...")
        all_objects = self.object_generator.get_all_objects()
        print(f"[INFO] Total objects to add: {len(all_objects)}")
        
        for idx, obj in enumerate(all_objects, 1):
            final_pos = obj.get_final_position()
            final_rot = obj.get_final_rotation()
            print(f"[INFO] Adding object {idx}/{len(all_objects)}: {obj.name} at ({final_pos['x']:.1f}, {final_pos['y']:.1f}, {final_pos['z']:.1f})")
            
            if obj.is_custom_model and obj.custom_config:
                try:
                    from tdw.librarian import ModelRecord
                    record_file = obj.custom_config.get("record")
                    if not record_file:
                        print(f"[ERROR] Custom model {obj.name} missing record file path")
                        continue
                    with open(Path(record_file), 'r', encoding='utf-8') as f:
                        record_data = json.load(f)
                    custom_record = ModelRecord(record_data)
                    model_name = custom_record.name
                    model_url = custom_record.get_url()
                    if not model_url:
                        print(f"[ERROR] Custom model {obj.name} has no valid URL")
                        continue
                    self.c.communicate([{
                        "$type": "add_object",
                        "name": model_name,
                        "url": model_url,
                        "scale_factor": obj.scale * self.overall_scale,
                        "position": {"x": final_pos["x"], "y": final_pos["y"] + 1.0, "z": final_pos["z"]},
                        "rotation": {"x": final_rot["x"], "y": final_rot["y"] + 1.0, "z": final_rot["z"]},
                        "category": "custom",
                        "id": obj.object_id
                    }, {
                        "$type": "set_kinematic_state",
                        "id": obj.object_id,
                        "is_kinematic": True,
                        "use_gravity": False
                    }])
                except Exception as e:
                    print(f"[ERROR] Failed to load custom model {obj.name}: {e}")
                    continue
            else:
                self.c.communicate([
                    self.c.get_add_object(
                model_name=obj.model,
                position=final_pos,
                        rotation=final_rot,
                        object_id=obj.object_id,
                        library="models_core.json"
                    )
                ])
                final_scale = obj.scale * self.overall_scale
                if final_scale != 1.0:
                    self.c.communicate([{
                        "$type": "scale_object",
                        "id": obj.object_id,
                        "scale_factor": {
                            "x": final_scale,
                            "y": final_scale,
                            "z": final_scale
                        }
                    }])
            
            if obj.color:
                if obj.is_custom_model and obj.custom_config and obj.custom_config.get("color"):
                    color_values = obj.custom_config.get("color")
                else:
                    color_values = COLORS.get(obj.color, {"r": 1.0, "g": 1.0, "b": 1.0, "a": 1.0})
                self.c.communicate([{
                    "$type": "set_color",
                    "id": obj.object_id,
                    "color": color_values
                }])
            
            self.c.communicate([{
                "$type": "set_mass",
                "id": obj.object_id,
                "mass": 1.0
            }])
            self.c.communicate([{
                "$type": "set_kinematic_state",
                "id": obj.object_id,
                "is_kinematic": True,
                "use_gravity": False
            }])
        
        if self.enable_gravity_fix:
            print("[INFO] Waiting for all objects to settle with gravity...")
            time.sleep(self.physics_settle_time)
        
        kinematic_commands = []
        for obj in all_objects:
            kinematic_commands.append({
                "$type": "set_kinematic_state",
                "id": obj.object_id,
                "is_kinematic": True,
                "use_gravity": False
            })
        if kinematic_commands:
            self.c.communicate(kinematic_commands)
            print(f"[INFO] Set {len(all_objects)} objects to kinematic state")

    def _setup_environment(self):
        skybox_lib = HDRISkyboxLibrarian()
        try:
            skybox_record = skybox_lib.records[77]
            self.c.communicate({
                "$type": "add_hdri_skybox",
                "name": skybox_record.name,
                "url": skybox_record.get_url(),
                "exposure": 0,
                "initial_skybox_rotation": 180,
                "sun_elevation": 60,
                "sun_initial_angle": 60,
                "sun_intensity": 1
            })
        except Exception:
            fallback_record = skybox_lib.get_record("sky_white")
            if fallback_record:
                TDWUtils.add_hdri_skybox(self.c, fallback_record)

    def _capture_scene_data(self):
        print("[INFO] Setting up cameras and capturing data...")
        self._ensure_door_id_mapping()
        
        # Initialize CameraCapture with legacy-compatible parameters
        self.camera_capture = CameraCapture(
            room_analyzer=self.room_analyzer,
            object_generator=self.object_generator,
            door_handler=self.door_handler,
            controller=self.c,
            output_dir=self.output_dir,
            door_id_mapping=self.door_id_mapping,
            door_positions=self.door_positions,
            filler_positions=self.filler_positions,
            door_object_ids=self.door_object_ids,
            filler_object_ids=self.filler_object_ids,
            scene_bounds=self.scene_bounds,
            physics_settle_time=self.physics_settle_time,
            enable_gravity_fix=self.enable_gravity_fix,
            overall_scale=self.overall_scale,
            model_librarian=self.model_lib
        )
        
        # Setup cameras in TDW (match legacy positions)
        main_cam_height = 0.8
        self.main_cam = ThirdPersonCamera(
            position={"x": 0, "y": main_cam_height, "z": 0},
            field_of_view=95,
            avatar_id="main_cam"
        )
        
        top_height = max(self.scene_width or 20, self.scene_depth or 20) * 1
        if top_height < 15:
            top_height = 15
        self.top_cam = ThirdPersonCamera(
            position={"x": 0, "y": top_height, "z": 0},
            look_at={"x": 0, "y": 0, "z": 0},
            field_of_view=60,
            avatar_id="top_down"
        )
        self.image_capture = ImageCapture(
            path=str(self.output_dir),
            avatar_ids=["main_cam", "top_down"],
            pass_masks=["_img", "_id"],
            png=True
        )
        
        self.c.add_ons.extend([self.main_cam, self.top_cam, self.image_capture])
        self.image_capture.set(frequency="never")
        
        self.camera_capture.top_camera_height = 20.0
        self.camera_capture.setup_cameras(self.main_cam, self.top_cam, None, self.image_capture)
        self.captured_images = self.camera_capture.capture_all()

    def _generate_base_metadata(self):
        print("[INFO] Generating legacy-compatible meta_data.json...")
        mask_list = self.mask_array.tolist()
        rows = len(mask_list)
        cols = len(mask_list[0]) if rows else 0
        scene_width = cols * self.cell_size
        scene_depth = rows * self.cell_size
        offset = self.room_analyzer.get_offset() if self.room_analyzer else (0, 0)
        metadata = {
            "room_size": [scene_width, scene_depth],
            "screen_size": [768, 768],
            "seed": self.seed,
            "run_seed": self.run_seed,
            "min_distance": self.object_generator.min_distance if self.object_generator else 0.0,
            "objects": [],
            "cameras": [],
            "images": [],
            "rooms": {},
            "doors": {},
            "room_object_assignments": {},
            "mask": mask_list,
            "offset": (-offset[0], -offset[1])
        }
        if self.object_generator:
            all_objects = self.object_generator.get_all_objects()
            for idx, obj in enumerate(all_objects, 1):
                metadata["objects"].append({
                    "object_id": obj.object_id,
                    "model": obj.model,
                    "name": obj.name,
                    "label": str(idx),
                    "pos": obj.get_final_position(),
                    "rot": obj.rot,
                    "size": list(obj.size),
                    "attributes": obj.attributes
                })
        self._ensure_door_id_mapping()
        if self.door_handler and self.door_id_mapping:
            for door_id, processed in self.door_handler.processed_doors.items():
                door_obj_id = self.door_id_mapping[door_id]
                door_center = processed.door_info.center
                metadata["objects"].append({
                    "object_id": door_obj_id,
                    "model": f"{processed.color_name}_door",
                    "name": f"{processed.color_name} door",
                    "pos": {"x": door_center[0], "y": 0, "z": door_center[1]},
                    "rot": {"x": 0, "y": 0, "z": 0},
                    "size": [processed.door_info.width, 2.0],
                    "attributes": {
                        "color": processed.color_name,
                        "orientation": processed.door_info.orientation,
                        "connected_rooms": processed.door_info.connected_rooms
                    }
                })
        agent = self.object_generator.get_agent() if self.object_generator else None
        if agent:
            metadata["cameras"].append({
                "id": "agent",
                "label": "A",
                "position": agent.pos,
                "rotation": {"y": 0.0}
            })
        if self.object_generator:
            for idx, obj in enumerate(self.object_generator.get_all_objects(), 1):
                final_pos = obj.get_final_position()
                yaw = math.degrees(math.atan2(-final_pos["x"], -final_pos["z"]))
                metadata["cameras"].append({
                    "id": str(obj.object_id),
                    "label": str(idx),
                    "position": {"x": final_pos["x"], "y": 0.8, "z": final_pos["z"]},
                    "rotation": {"y": yaw}
                })
        if self.door_handler and self.door_id_mapping:
            for door_id, processed in self.door_handler.processed_doors.items():
                door_obj_id = self.door_id_mapping[door_id]
                door_center = processed.door_info.center
                metadata["cameras"].append({
                    "id": str(door_obj_id),
                    "label": f"D{door_obj_id}",
                    "position": {"x": door_center[0], "y": 0.8, "z": door_center[1]},
                    "rotation": {"y": 0.0}
                })
        if self.room_analyzer:
            for room_id, room in self.room_analyzer.rooms.items():
                room_key = int(room_id)
                bounds = self.room_analyzer.get_room_bounds_world(room_id)
                metadata["rooms"][room_key] = {
                    "center": [room.center[0], room.center[1]],
                    "bounds": bounds,
                    "area": room.area,
                    "connected_rooms": list(self.room_analyzer.room_connections.get(room_id, []))
                }
                metadata["room_object_assignments"][room_key] = []
        if self.object_generator:
            for obj in self.object_generator.get_all_objects():
                room_key = int(obj.room_id)
                if room_key in metadata["room_object_assignments"]:
                    metadata["room_object_assignments"][room_key].append({
                        "object_id": obj.object_id,
                        "name": obj.name,
                        "label": next((cam["label"] for cam in metadata["cameras"] if cam["id"] == str(obj.object_id)), str(obj.object_id))
                    })
        if self.door_handler:
            for door_id, processed in self.door_handler.processed_doors.items():
                door_key = int(door_id)
                metadata["doors"][door_key] = {
                    "center": [
                        processed.door_info.center[0],
                        processed.door_info.center[1]
                    ],
                    "color": processed.color_name,
                    "connected_rooms": processed.door_info.connected_rooms,
                    "orientation": processed.door_info.orientation,
                    "width": processed.door_info.width
                }
        if getattr(self, "_validation_metadata", None):
            validated = self._validation_metadata.get("validated_tasks")
            if validated:
                metadata["validated_tasks"] = validated
        if "validated_tasks" not in metadata:
            metadata["validated_tasks"] = self.LEGACY_VALIDATED_TASKS.copy()
        self.metadata_path = self.output_dir / "meta_data.json"
        safe_metadata = self._make_json_safe(metadata)
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(safe_metadata, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Base metadata saved to {self.metadata_path}")

    def _update_metadata_with_images(self):
        if not self.metadata_path or not self.metadata_path.exists():
            print("[WARN] Metadata path not set, cannot update images.")
            return
        try:
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            metadata["images"] = self.captured_images
            safe_metadata = self._make_json_safe(metadata)
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(safe_metadata, f, indent=2, ensure_ascii=False)
            print(f"[INFO] Updated metadata with {len(self.captured_images)} images")
        except Exception as e:
            print(f"[WARN] Failed to update metadata images: {e}")

    def _generate_task_viewpoints(self):
        if not self.metadata_path or not self.metadata_path.exists():
            print("[WARN] Metadata not ready, skip task viewpoint generation.")
            return
        try:
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            metadata.setdefault("run_seed", self.run_seed)
            metadata.setdefault("seed", self.seed)
            self.task_viewpoints = generate_task_viewpoints(
                metadata=metadata,
                run_seed=self.run_seed,
                num_questions_per_task=3
            )
            if not self.task_viewpoints:
                print("[WARN] No task viewpoints generated.")
                return
            task_cameras = viewpoints_to_camera_metadata(self.task_viewpoints)
            metadata.setdefault("cameras", []).extend(task_cameras)
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            print(f"[INFO] Added {len(task_cameras)} task cameras to metadata")
        except Exception as e:
            print(f"[WARN] Failed to generate task viewpoints: {e}")
            self.task_viewpoints = []

    def _capture_task_viewpoint_images(self):
        if not self.task_viewpoints:
            return
        print(f"[INFO] Capturing {len(self.task_viewpoints)} task viewpoints...")
        for vp in self.task_viewpoints:
            task_group = vp.get("task_group", vp.get("task_type"))
            if task_group == "false_belief":
                self._capture_false_belief_image(vp)
            else:
                self._capture_regular_task_image(vp)

    def _capture_regular_task_image(self, vp: Dict):
        pos = vp.get("pos_tdw")
        yaw = vp.get("yaw", 0.0)
        if pos is None:
            return
        camera_pos = {"x": float(pos[0]), "y": 0.8, "z": float(pos[1])}
        rad = math.radians(yaw)
        look_at = {
            "x": camera_pos["x"] + math.sin(rad) * 3.0,
            "y": camera_pos["y"],
            "z": camera_pos["z"] + math.cos(rad) * 3.0
        }
        self.main_cam.teleport(camera_pos)
        self.main_cam.look_at(look_at)
        hidden_items = self._temporarily_hide_items_for_camera(camera_pos)
        try:
            self.image_capture.set(frequency="once", avatar_ids=["main_cam"], pass_masks=["_img"], save=False)
            self.c.communicate([])
            images = self.image_capture.get_pil_images()
            if "main_cam" in images and "_img" in images["main_cam"]:
                filename = f"{vp['image_name']}.png"
                images["main_cam"]["_img"].save(self.output_dir / filename)
                self._record_task_capture(filename, camera_pos, yaw, vp)
        finally:
            self._restore_hidden_items(hidden_items)

    def _capture_false_belief_image(self, vp: Dict):
        print("[WARN] False belief task capture not yet implemented; skipping image.")

    def _record_task_capture(self, filename: str, camera_pos: Dict[str, float], yaw: float, vp: Dict):
        task_group = vp.get("task_group", vp.get("task_type"))
        task_type = vp.get("task_type")
        self.captured_images.append({
            "file": filename,
            "cam_id": f"task_{vp.get('image_name')}",
            "pos": camera_pos,
            "rotation": {"y": yaw},
            "direction": f"task_{task_group}" if task_group else "",
            "task_type": task_type,
            "task_group": task_group,
            "object_ratios": []
        })

    def _find_doors_at_position(self, camera_pos: Dict[str, float], tolerance: float = 0.1):
        doors_at = []
        fillers_at = []
        cam_x, cam_z = camera_pos["x"], camera_pos["z"]
        for door_id, pos in self.door_positions.items():
            distance = ((cam_x - pos["x"]) ** 2 + (cam_z - pos["z"]) ** 2) ** 0.5
            if distance <= tolerance:
                doors_at.append(door_id)
        for filler_id, pos in self.filler_positions.items():
            distance = ((cam_x - pos["x"]) ** 2 + (cam_z - pos["z"]) ** 2) ** 0.5
            if distance <= tolerance:
                fillers_at.append(filler_id)
        return doors_at, fillers_at

    def _temporarily_hide_items_for_camera(self, camera_pos: Dict[str, float], tolerance: float = 0.5):
        hidden_items = []
        if self.object_generator:
            for obj in self.object_generator.get_all_objects():
                final_pos = obj.get_final_position()
                if abs(final_pos["x"] - camera_pos["x"]) < 1e-4 and abs(final_pos["z"] - camera_pos["z"]) < 1e-4:
                    self.c.communicate([{
                        "$type": "teleport_object",
                        "id": obj.object_id,
                        "position": {"x": 999, "y": -999, "z": 999}
                    }])
                    hidden_items.append(("object", obj))
                    break
        doors_at, fillers_at = self._find_doors_at_position(camera_pos, tolerance)
        for door_id in doors_at:
            self.c.communicate([{
                "$type": "teleport_object",
                "id": door_id,
                "position": {"x": 999, "y": -999, "z": 999}
            }])
            hidden_items.append(("door", door_id))
        for filler_id in fillers_at:
            self.c.communicate([{
                "$type": "teleport_object",
                "id": filler_id,
                "position": {"x": 999, "y": -999, "z": 999}
            }])
            hidden_items.append(("filler", filler_id))
        return hidden_items

    def _restore_hidden_items(self, hidden_items: List[Tuple[str, any]]):
        if not hidden_items:
            return
        time.sleep(0.05)
        for item_type, data in hidden_items:
            if item_type == "object":
                obj = data
                self.c.communicate([{
                    "$type": "teleport_object",
                    "id": obj.object_id,
                    "position": obj.pos
                }])
                if self.enable_gravity_fix:
                    self.c.communicate([{
                        "$type": "set_kinematic_state",
                        "id": obj.object_id,
                        "is_kinematic": True,
                        "use_gravity": False
                    }])
            elif item_type == "door" and data in self.door_positions:
                self.c.communicate([{
                    "$type": "teleport_object",
                    "id": data,
                    "position": self.door_positions[data]
                }])
            elif item_type == "filler" and data in self.filler_positions:
                self.c.communicate([{
                    "$type": "teleport_object",
                    "id": data,
                    "position": self.filler_positions[data]
                }])
        time.sleep(self.physics_settle_time)

    def _make_json_safe(self, value):
        if isinstance(value, dict):
            return {k: self._make_json_safe(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._make_json_safe(v) for v in value]
        if isinstance(value, tuple):
            return [self._make_json_safe(v) for v in value]
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        if isinstance(value, np.ndarray):
            return self._make_json_safe(value.tolist())
        return value

    def _generate_orientation_instruction(self):
        if not self.metadata_path:
            return
        base_models_dir = Path(__file__).resolve().parents[1] / "models"
        default_instruction = base_models_dir / "new_models"
        default_dual = base_models_dir / "new_models_dual"
        instruction_dir = Path(self.scene_config.get('orientation_instruction_dir', default_instruction))
        dual_dir = Path(self.scene_config.get('orientation_instruction_dual_dir', default_dual))
        side_dir_config = self.scene_config.get('orientation_instruction_side_dir', None)
        generator = OrientationInstructionGenerator(
            instruction_images_dir=str(instruction_dir),
            output_dir=self.output_dir,
            dual_images_dir=str(dual_dir) if dual_dir else None,
            side_images_dir=side_dir_config
        )
        result = generator.generate_instruction_image(self.metadata_path, output_filename="orientation_instruction.png")
        if not result:
            print("[WARN] Orientation instruction image was not generated")

    def _export_additional_metadata(self):
        if not all([self.room_analyzer, self.object_generator, self.door_handler, self.camera_capture]):
            return
        try:
            metadata_manager = MetadataManager(
            self.room_analyzer,
            self.object_generator,
            self.door_handler,
            self.camera_capture,
                self.output_dir,
                self.scene_params
        )
            metadata_manager.export_room_settings()
            metadata_manager.export_summary_report()
        except Exception as e:
            print(f"[WARN] Failed to export auxiliary metadata: {e}")

    def _extract_validation_data(self):
        # Helper to extract data for validation
        mask = self.room_analyzer.mask
        objects_data = []
        for obj in self.object_generator.get_all_objects():
             final_pos = obj.get_final_position()
             objects_data.append({
                 'name': obj.name,
                 'position': (final_pos['x'], final_pos['z']),
                 'room_id': obj.room_id,
                 'is_placed': True
             })
        
        agent = self.object_generator.get_agent()
        agent_data = {}
        if agent:
             agent_data = {
                 'name': 'agent',
                 'position': (agent.pos['x'], agent.pos['z']),
                 'rotation': agent.rotation.get('y', 0),
                 'room_id': agent.room_id
             }
        
        return mask, objects_data, agent_data

