"""
Camera Capture Module
====================

Handles all camera operations and image capture:
- Room photography (agent + objects)
- Door photography from 4 directions
- Top-down and oblique views
- Image annotation and processing
"""

import math
import time
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from tdw.librarian import ModelLibrarian  # type: ignore
except ImportError:  # pragma: no cover - TDW optional during linting
    ModelLibrarian = None

from .room_analyzer import RoomAnalyzer
from .object_generator import ObjectGenerator, PlacedObject, AgentInfo
from .door_handler import DoorHandler, DoorCameraSpec


def _compute_object_id_array(id_img: np.ndarray) -> np.ndarray:
    """Convert TDW _id pass RGB image to integer segmentation IDs."""
    if id_img.ndim == 3 and id_img.shape[2] >= 3:
        return (
            id_img[..., 0].astype("int32")
            + (id_img[..., 1].astype("int32") << 8)
            + (id_img[..., 2].astype("int32") << 16)
        )
    return id_img.astype("int32")


def build_seg_to_object_map_topdown(controller, image_capture, objects: List[PlacedObject], agent_marker_id: Optional[int] = None) -> Dict[int, int]:
    """
    Build segmentation ID -> object ID mapping by hiding/showing objects individually.
    """
    mapping: Dict[int, int] = {}
    if not objects:
        return mapping

    original_pos = {o.object_id: dict(o.pos) for o in objects}

    def hide_cmd(obj_id):
        return {"$type": "teleport_object", "id": obj_id, "position": {"x": 999, "y": -999, "z": 999}}

    def show_cmd(obj_id, pos):
        return {"$type": "teleport_object", "id": obj_id, "position": pos}

    try:
        controller.communicate([hide_cmd(o.object_id) for o in objects])
        controller.communicate([])

        image_capture.set(frequency="once", avatar_ids=["top_down"], pass_masks=["_id"], save=False)
        controller.communicate([])
        bg_img = image_capture.get_pil_images()["top_down"]["_id"]
        bg_arr = _compute_object_id_array(np.array(bg_img))
        bg_ids = set(np.unique(bg_arr).tolist())
        bg_ids.discard(0)

        for obj in objects:
            controller.communicate([show_cmd(obj.object_id, original_pos[obj.object_id])])
            controller.communicate([])
            image_capture.set(frequency="once", avatar_ids=["top_down"], pass_masks=["_id"], save=False)
            controller.communicate([])
            cur_img = image_capture.get_pil_images()["top_down"]["_id"]
            cur_arr = _compute_object_id_array(np.array(cur_img))
            cur_ids = set(np.unique(cur_arr).tolist())
            cur_ids.discard(0)

            new_ids = [sid for sid in cur_ids if sid not in bg_ids]
            if new_ids:
                best_seg = max(new_ids, key=lambda sid: int((cur_arr == sid).sum()))
                mapping[best_seg] = obj.object_id

            controller.communicate([hide_cmd(obj.object_id)])
            controller.communicate([])

    finally:
        controller.communicate([show_cmd(oid, pos) for oid, pos in original_pos.items()])
        controller.communicate([])

    return mapping


def annotate_topdown_with_segmentation(rgbimg: Image.Image,
                                       id_img: Image.Image,
                                       objects: List[PlacedObject],
                                       doors_data: Dict[int, Dict],
                                       agent_data: Dict[str, float],
                                       save_path: Path,
                                       scene_bounds: Optional[Tuple[float, float, float, float]] = None,
                                       seg_to_obj_map: Optional[Dict[int, int]] = None,
                                       agent_pixel_position: Optional[Tuple[int, int]] = None):
    """Annotate top-down image using segmentation IDs for accurate labels."""
    rgb = rgbimg.convert("RGBA")
    id_arr = _compute_object_id_array(np.array(id_img))
    draw = ImageDraw.Draw(rgb)

    font = None
    font_paths = [
        "arialbd.ttf",
        "/System/Library/Fonts/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Windows/Fonts/arialbd.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for font_path in font_paths:
        try:
            font = ImageFont.truetype(font_path, 15)
            break
        except Exception:
            continue
    if font is None:
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

    object_to_label = {}
    for i, obj in enumerate(objects, 1):
        object_to_label[obj.object_id] = str(i)

    matched_objects = []
    if seg_to_obj_map:
        for obj in objects:
            seg_id = next((sid for sid, oid in seg_to_obj_map.items() if oid == obj.object_id), None)
            if seg_id is None:
                continue
            mask = (id_arr == seg_id)
            if not mask.any():
                continue
            ys, xs = np.nonzero(mask)
            cy, cx = int(ys.mean()), int(xs.mean())
            matched_objects.append((obj, (cx, cy)))
    else:
        h, w = id_arr.shape[:2]
        if scene_bounds:
            min_x, max_x, min_z, max_z = scene_bounds
            scene_width = max_x - min_x
            scene_depth = max_z - min_z
        else:
            min_x = max_x = min_z = max_z = 0
            scene_width = scene_depth = 1
        for obj in objects:
            obj_pos = obj.get_final_position()
            norm_x = (obj_pos["x"] - min_x) / scene_width if scene_width else 0.5
            norm_z = (obj_pos["z"] - min_z) / scene_depth if scene_depth else 0.5
            img_x = int(norm_x * w)
            img_y = int((1.0 - norm_z) * h)
            best_match = None
            best_distance = float("inf")
            unique_ids = np.unique(id_arr)
            for seg_id in unique_ids:
                if seg_id == 0:
                    continue
                mask = (id_arr == seg_id)
                if not mask.any():
                    continue
                ys, xs = np.nonzero(mask)
                cy, cx = int(ys.mean()), int(xs.mean())
                distance = ((img_x - cx) ** 2 + (img_y - cy) ** 2) ** 0.5
                if distance < best_distance:
                    best_distance = distance
                    best_match = (cx, cy)
            if best_match and best_distance < 200:
                matched_objects.append((obj, best_match))

    for obj, (cx, cy) in matched_objects:
        label_text = object_to_label.get(obj.object_id, "?")
        r = 5
        label_x = cx + r + 8
        label_y = cy + r + 8
        if font:
            bbox = draw.textbbox((label_x, label_y), label_text, font=font)
            padding = 2
            expanded_bbox = (
                bbox[0] - padding,
                bbox[1] - padding,
                bbox[2] + padding,
                bbox[3] + padding,
            )
            draw.rectangle(expanded_bbox, fill=(255, 0, 0, 200))
            draw.text((label_x, label_y), label_text, fill=(255, 255, 255, 255), font=font)
        else:
            draw.rectangle((label_x, label_y, label_x + 15, label_y + 10), fill=(255, 0, 0, 255))

    if agent_pixel_position:
        agent_x, agent_y = agent_pixel_position
        agent_radius = 8
        arrow_length = 30
        arrow_end_x = agent_x
        arrow_end_y = agent_y - arrow_length
        draw.line([(agent_x, agent_y), (arrow_end_x, arrow_end_y)], fill=(255, 0, 0, 255), width=3)
        head_size = 8
        arrow_head = [
            (arrow_end_x, arrow_end_y),
            (arrow_end_x - head_size // 2, arrow_end_y + head_size),
            (arrow_end_x + head_size // 2, arrow_end_y + head_size)
        ]
        draw.polygon(arrow_head, fill=(255, 0, 0, 255))
        draw.ellipse(
            [
                (agent_x - agent_radius, agent_y - agent_radius),
                (agent_x + agent_radius, agent_y + agent_radius)
            ],
            fill=(0, 0, 255, 255)
        )

    for door_id, door_info in doors_data.items():
        door_center = door_info.get("center", (0, 0))
        if scene_bounds:
            min_x, max_x, min_z, max_z = scene_bounds
            width = max_x - min_x
            depth = max_z - min_z
            px = int(((door_center[0] - min_x) / width) * rgb.width) if width else rgb.width // 2
            py = int((1 - (door_center[1] - min_z) / depth) * rgb.height) if depth else rgb.height // 2
        else:
            px = rgb.width // 2
            py = rgb.height // 2
        color_name = door_info.get("color", "door")
        draw.rectangle([px - 6, py - 6, px + 6, py + 6], fill=(255, 165, 0, 255))
        label = f"{color_name}"
        label_x = px + 8
        label_y = py - 8
        if font:
            bbox = draw.textbbox((label_x, label_y), label, font=font)
            padding = 2
            expanded_bbox = (
                bbox[0] - padding,
                bbox[1] - padding,
                bbox[2] + padding,
                bbox[3] + padding,
            )
            draw.rectangle(expanded_bbox, fill=(255, 165, 0, 255))
            draw.text((label_x, label_y), label, fill=(255, 255, 255, 255), font=font)
        else:
            draw.rectangle((label_x, label_y, label_x + 40, label_y + 15), fill=(255, 165, 0, 255))
            draw.text((label_x + 2, label_y + 2), label, fill=(255, 255, 255, 255))

    rgb.save(save_path)

class CameraCapture:
    """Handles all camera capture operations for multi-room environments"""
    
    def __init__(self, room_analyzer: RoomAnalyzer, object_generator: ObjectGenerator, 
                 door_handler: DoorHandler, controller, output_dir: Path,
                 door_id_mapping: Optional[Dict[int, int]] = None,
                 door_positions: Optional[Dict[int, Dict[str, float]]] = None,
                 filler_positions: Optional[Dict[int, Dict[str, float]]] = None,
                 door_object_ids: Optional[List[int]] = None,
                 filler_object_ids: Optional[List[int]] = None,
                 scene_bounds: Optional[Tuple[float, float, float, float]] = None,
                 physics_settle_time: float = 0.2,
                 enable_gravity_fix: bool = True,
                 overall_scale: float = 1.0,
                 model_librarian: Optional["ModelLibrarian"] = None):
        """
        Initialize camera capture system
        
        Args:
            room_analyzer: Room layout analysis
            object_generator: Object placement results
            door_handler: Door processing results
            controller: TDW controller instance
            output_dir: Directory for saving images
        """
        self.room_analyzer = room_analyzer
        self.object_generator = object_generator
        self.door_handler = door_handler
        self.controller = controller
        self.output_dir = Path(output_dir)
        self.door_id_mapping = door_id_mapping or {}
        self.door_positions = door_positions or {}
        self.filler_positions = filler_positions or {}
        self.door_object_ids = door_object_ids or []
        self.filler_object_ids = filler_object_ids or []
        self.scene_bounds = scene_bounds
        self.physics_settle_time = physics_settle_time
        self.enable_gravity_fix = enable_gravity_fix
        self.overall_scale = overall_scale
        self.model_librarian = model_librarian
        
        # Camera components (will be initialized)
        self.main_cam = None
        self.top_cam = None
        self.oblique_cam = None
        self.image_capture = None
        
        # Captured data
        self.room_shots: Dict[int, List[Dict]] = {}
        self.door_shots: List[Dict] = []
        self.global_shots: List[Dict] = []
        self.captured_images: List[Dict] = []
        self.seg_to_obj_map: Dict[int, int] = {}
        self.agent_pixel_position: Optional[Tuple[int, int]] = None
        self.top_camera_height: float = 20.0
        
        self.output_dir.mkdir(exist_ok=True, parents=True)
    
    def setup_cameras(self, main_cam, top_cam, oblique_cam, image_capture):
        """Set camera components from main script"""
        self.main_cam = main_cam
        self.top_cam = top_cam  
        self.oblique_cam = oblique_cam
        self.image_capture = image_capture
    
    def capture_all(self) -> List[Dict]:
        """Capture all required images following the legacy pipeline."""
        self.global_shots.clear()
        self.room_shots.clear()
        self.door_shots.clear()
        self.captured_images = []
        print("[INFO] Starting comprehensive image capture...")
        self._capture_topdown_views()
        self._capture_perspective_views()
        print(f"[INFO] Captured {len(self.captured_images)} images total")
        return self.captured_images
    
    def _capture_topdown_views(self):
        """Capture legacy top-down RGB + annotated views with segmentation."""
        print("[INFO] Capturing top-down views...")
        agent = self.object_generator.get_agent()
        agent_marker_id = None
        if agent:
            agent_marker_id = self.controller.get_unique_id()
            self.controller.communicate([
                {
                    "$type": "load_primitive_from_resources",
                    "primitive_type": "Cube",
                    "id": agent_marker_id,
                    "position": {"x": agent.pos["x"], "y": 0.1, "z": agent.pos["z"]}
                },
                {
                    "$type": "scale_object",
                    "id": agent_marker_id,
                    "scale_factor": {"x": 0.1, "y": 0.1, "z": 0.1}
                },
                {
                    "$type": "set_color",
                    "id": agent_marker_id,
                    "color": {"r": 1.0, "g": 0.0, "b": 0.0, "a": 1.0}
                }
            ])
        
        self.image_capture.set(frequency="once", avatar_ids=["top_down"], pass_masks=["_img", "_id"], save=False)
        self.controller.communicate([])
        top_imgs = self.image_capture.get_pil_images()
        rgb_img = top_imgs["top_down"]["_img"]
        id_img = top_imgs["top_down"]["_id"]
        rgb_path = self.output_dir / "top_down.png"
        rgb_img.save(rgb_path)

        self.global_shots.append({
            "type": "top_down",
            "file": "top_down.png",
            "camera": "top_down",
            "description": "Complete scene overview"
        })
        self.captured_images.append({
            "file": "top_down.png",
            "cam_id": "top_down",
            "pos": {"x": 0, "y": int(self.top_camera_height), "z": 0},
            "direction": "down",
            "object_ratios": []
        })

        # Build segmentation mapping
        all_objects = self.object_generator.get_all_objects()
        self.seg_to_obj_map = build_seg_to_object_map_topdown(self.controller, self.image_capture, all_objects, agent_marker_id)

        self.agent_pixel_position = None
        if agent_marker_id:
            self.controller.communicate([{
                "$type": "teleport_object",
                "id": agent_marker_id,
                "position": {"x": 999, "y": -999, "z": 999}
            }])
            self.controller.communicate([])
            self.image_capture.set(frequency="once", avatar_ids=["top_down"], pass_masks=["_id"], save=False)
            self.controller.communicate([])
            without_agent = self.image_capture.get_pil_images()["top_down"]["_id"]
            without_arr = _compute_object_id_array(np.array(without_agent))
            with_arr = _compute_object_id_array(np.array(id_img))
            with_ids = set(np.unique(with_arr).tolist())
            without_ids = set(np.unique(without_arr).tolist())
            diff_ids = with_ids - without_ids
            if diff_ids:
                seg_id = list(diff_ids)[0]
                mask = (with_arr == seg_id)
                if mask.any():
                    ys, xs = np.nonzero(mask)
                    cy, cx = int(ys.mean()), int(xs.mean())
                    self.agent_pixel_position = (cx, cy)
            self.controller.communicate([{"$type": "destroy_object", "id": agent_marker_id}])
        
        doors_meta = {}
        if self.door_handler:
            for door_id, processed in self.door_handler.processed_doors.items():
                doors_meta[door_id] = {
                    "center": processed.door_info.center,
                    "color": processed.color_name
                }

        agent_data = {"x": agent.pos["x"], "z": agent.pos["z"]} if agent else {"x": 0.0, "z": 0.0}
        annotated_path = self.output_dir / "top_down_annotated.png"
        annotate_topdown_with_segmentation(
            rgb_img,
            id_img,
            all_objects,
            doors_meta,
            agent_data,
            annotated_path,
            scene_bounds=self.scene_bounds,
            seg_to_obj_map=self.seg_to_obj_map,
            agent_pixel_position=self.agent_pixel_position
        )
        self.global_shots.append({
            "type": "top_down_annotated",
            "file": "top_down_annotated.png",
            "camera": "top_down",
            "description": "Annotated scene overview"
        })
        self.captured_images.append({
            "file": "top_down_annotated.png",
            "cam_id": "top_down_annotated",
            "pos": {"x": 0, "y": int(self.top_camera_height), "z": 0},
            "direction": "down",
            "object_ratios": []
        })
    
    def _capture_perspective_views(self):
        """Capture agent/object/door perspective views."""
        print("[INFO] Capturing perspective views...")
        all_positions: List[Dict] = []
        agent = self.object_generator.get_agent()
        if agent:
            all_positions.append({
                "id": "agent",
                "label": "A",
                "position": {"x": agent.pos["x"], "y": 0.8, "z": agent.pos["z"]},
                "type": "agent",
                "room": agent.room_id
            })

        all_objects = self.object_generator.get_all_objects()
        for i, obj in enumerate(all_objects, 1):
            pos = obj.get_final_position()
            all_positions.append({
                "id": str(obj.object_id),
                "label": str(i),
                "position": {"x": pos["x"], "y": 0.8, "z": pos["z"]},
                "type": "object",
                "room": obj.room_id
            })

        if self.door_handler and self.door_id_mapping:
            for door_id, processed in self.door_handler.processed_doors.items():
                door_obj_id = self.door_id_mapping.get(door_id)
                if not door_obj_id:
                    continue
                center = processed.door_info.center
                for room_id in processed.door_info.connected_rooms:
                    all_positions.append({
                        "id": str(door_obj_id),
                        "label": f"D{door_obj_id}",
                        "position": {"x": center[0], "y": 0.8, "z": center[1]},
                        "type": "door",
                        "room": room_id,
                        "original_door_id": door_id
                    })

        total_expected = len(all_positions) * 4
        captured = 0
        for cam_spec in all_positions:
            pos = cam_spec["position"]
            cam_type = cam_spec["type"]
            for direction in [0, 90, 180, 270]:
                try:
                    self._capture_view(
                        pos=pos.copy(),
                        deg=direction,
                        tag=cam_spec["id"],
                        label=cam_spec["label"],
                        cam_type=cam_type,
                        room=cam_spec.get("room"),
                        original_door_id=cam_spec.get("original_door_id")
                    )
                    captured += 1
                except Exception as exc:
                    print(f"[WARN] Failed to capture {cam_type} {cam_spec['id']} facing {direction}: {exc}")
                    continue

        print(f"[INFO] Successfully captured {captured}/{total_expected} perspective images")
    
    def _find_doors_at_position(self, pos: Dict[str, float], tolerance: float = 0.5) -> Tuple[List[int], List[int]]:
        doors = []
        fillers = []
        for door_id in self.door_object_ids:
            orig = self.door_positions.get(door_id)
            if not orig:
                continue
            dist = math.sqrt((orig["x"] - pos["x"]) ** 2 + (orig["z"] - pos["z"]) ** 2)
            if dist <= tolerance:
                doors.append(door_id)
        for filler_id in self.filler_object_ids:
            orig = self.filler_positions.get(filler_id)
            if not orig:
                continue
            dist = math.sqrt((orig["x"] - pos["x"]) ** 2 + (orig["z"] - pos["z"]) ** 2)
            if dist <= tolerance:
                fillers.append(filler_id)
        return doors, fillers

    def _capture_view(self, pos: Dict[str, float], deg: int, tag: str, label: str,
                      cam_type: str, room: Optional[int] = None,
                      original_door_id: Optional[int] = None):
        rad = math.radians(deg)
        look_at = {
            "x": pos["x"] + math.sin(rad) * 3,
            "y": pos["y"],
            "z": pos["z"] + math.cos(rad) * 3
        }
        self.main_cam.teleport(pos)
        self.main_cam.look_at(look_at)

        hidden_object = None
        if cam_type == "object" and tag.isdigit():
            object_id = int(tag)
            for obj in self.object_generator.get_all_objects():
                if obj.object_id == object_id:
                    final_pos = obj.get_final_position()
                    if abs(final_pos["x"] - pos["x"]) < 1e-3 and abs(final_pos["z"] - pos["z"]) < 1e-3:
                        hidden_object = obj
                        break
            if hidden_object:
                self.controller.communicate([{
                    "$type": "teleport_object",
                    "id": hidden_object.object_id,
                    "position": {"x": 999, "y": -999, "z": 999}
                }])

        hidden_doors = []
        if cam_type == "door":
            doors_at_pos, fillers_at_pos = self._find_doors_at_position(pos)
            for door_id in doors_at_pos:
                self.controller.communicate([{
                    "$type": "teleport_object",
                    "id": door_id,
                    "position": {"x": 999, "y": -999, "z": 999}
                }])
                hidden_doors.append(("door", door_id))
            for filler_id in fillers_at_pos:
                self.controller.communicate([{
                    "$type": "teleport_object",
                    "id": filler_id,
                    "position": {"x": 999, "y": -999, "z": 999}
                }])
                hidden_doors.append(("filler", filler_id))

        self.image_capture.set(frequency="once", avatar_ids=["main_cam"], pass_masks=["_img"], save=False)
        self.controller.communicate([])
        color_images = self.image_capture.get_pil_images()["main_cam"]

        if hidden_object is not None:
            self.controller.communicate([{
                "$type": "teleport_object",
                "id": hidden_object.object_id,
                "position": hidden_object.pos
            }])
            time.sleep(self.physics_settle_time)
            if self.enable_gravity_fix:
                self.controller.communicate([{
                    "$type": "set_kinematic_state",
                    "id": hidden_object.object_id,
                    "is_kinematic": True,
                    "use_gravity": False
                }])

        if hidden_doors:
            for obj_type, obj_id in hidden_doors:
                if obj_type == "door" and obj_id in self.door_positions:
                    self.controller.communicate([{
                        "$type": "teleport_object",
                        "id": obj_id,
                        "position": self.door_positions[obj_id]
                    }])
                if obj_type == "filler" and obj_id in self.filler_positions:
                    self.controller.communicate([{
                        "$type": "teleport_object",
                        "id": obj_id,
                        "position": self.filler_positions[obj_id]
                    }])
            time.sleep(0.05)

        dir_name = {0: "north", 90: "east", 180: "south", 270: "west"}[deg]
        filename = f"{tag}_facing_{dir_name}.png"
        if "_img" in color_images:
            color_images["_img"].copy().save(self.output_dir / filename)
            object_ratios = self._calculate_object_ratios(
                camera_pos=pos,
                camera_deg=deg,
                cam_id=tag,
                room_filter=room
            )
            image_info = {
                "file": filename,
                "cam_id": tag,
                "pos": {"x": pos["x"], "y": pos["y"], "z": pos["z"]},
                "direction": dir_name,
                "object_ratios": object_ratios
            }
            self.captured_images.append(image_info)
            if cam_type in {"agent", "object"}:
                if room is not None:
                    room_entry = dict(image_info)
                    room_entry.update({"type": cam_type, "room": room})
                    self.room_shots.setdefault(room, []).append(room_entry)
            if cam_type == "door":
                door_entry = dict(image_info)
                door_entry.update({
                    "type": cam_type,
                    "room": room,
                    "original_door_id": original_door_id
                })
                self.door_shots.append(door_entry)

    def _get_object_bounds(self, obj: PlacedObject) -> Tuple[float, float, float]:
        """Return approximate object bounds, applying overall scale and rotation."""
        rotation = obj.get_final_rotation() if hasattr(obj, "get_final_rotation") else obj.rot
        scale = getattr(obj, "scale", 1.0) * self.overall_scale
        if self.model_librarian is not None:
            try:
                record = self.model_librarian.get_record(obj.model)
                bounds = record.bounds
                width = (bounds["right"]["x"] - bounds["left"]["x"]) * scale
                depth = (bounds["front"]["z"] - bounds["back"]["z"]) * scale
                height = (bounds["top"]["y"] - bounds["bottom"]["y"]) * scale
                y_rot = rotation.get("y", 0) % 360
                if y_rot in (90, 270):
                    width, depth = depth, width
                elif y_rot not in (0, 180):
                    angle = math.radians(y_rot)
                    new_width = abs(width * math.cos(angle)) + abs(depth * math.sin(angle))
                    new_depth = abs(width * math.sin(angle)) + abs(depth * math.cos(angle))
                    width, depth = new_width, new_depth
                return width, depth, height
            except Exception:
                pass
        width = obj.size[0] if obj.size else scale
        depth = obj.size[1] if len(obj.size) > 1 else scale
        height = max(width, depth)
        return width, depth, height

    def _is_object_bbox_in_view(self, obj: PlacedObject, cam_pos: Dict[str, float],
                                look_direction: Tuple[float, float], fov_deg: float = 90) -> bool:
        """Check if the object's bounding box intersects the camera FOV."""
        try:
            from shapely.geometry import Polygon  # type: ignore
        except ImportError:  # pragma: no cover
            Polygon = None
        obj_pos = obj.get_final_position()
        width, depth, _ = self._get_object_bounds(obj)
        hx, hz = width / 2, depth / 2
        x0, z0 = obj_pos["x"], obj_pos["z"]
        obj_corners = [
            (x0 - hx, z0 - hz), (x0 + hx, z0 - hz),
            (x0 + hx, z0 + hz), (x0 - hx, z0 + hz)
        ]
        cx, cz = cam_pos["x"], cam_pos["z"]
        look_dir = np.array(look_direction, dtype=float)
        look_dir = look_dir / np.linalg.norm(look_dir)
        half_fov = math.radians(fov_deg / 2)

        for corner_x, corner_z in obj_corners:
            to_corner = np.array([corner_x - cx, corner_z - cz])
            if np.linalg.norm(to_corner) < 1e-6:
                return True
            to_corner_norm = to_corner / np.linalg.norm(to_corner)
            cos_angle = np.clip(np.dot(look_dir, to_corner_norm), -1.0, 1.0)
            angle = math.acos(cos_angle)
            if angle <= half_fov:
                return True

        if Polygon is None:
            return False

        radius = max(20, np.linalg.norm([x0 - cx, z0 - cz]) * 2)
        obj_polygon = Polygon(obj_corners)
        sector_points = [(cx, cz)]
        for i in range(32 + 1):
            angle = -half_fov + i * (2 * half_fov / 32)
            rot_x = look_dir[0] * math.cos(angle) - look_dir[1] * math.sin(angle)
            rot_z = look_dir[0] * math.sin(angle) + look_dir[1] * math.cos(angle)
            sector_points.append((cx + rot_x * radius, cz + rot_z * radius))
        fov_polygon = Polygon(sector_points)
        return obj_polygon.intersects(fov_polygon)

    def _compute_visibility_ratio(self, obj: PlacedObject, cam_pos: Dict[str, float],
                                  look_direction: Tuple[float, float], fov_deg: float = 90) -> float:
        """Compute how much of the object's bbox lies inside the FOV sector."""
        try:
            from shapely.geometry import Polygon  # type: ignore
        except ImportError:  # pragma: no cover
            return 0.0
        obj_pos = obj.get_final_position()
        width, depth, _ = self._get_object_bounds(obj)
        hx, hz = width / 2, depth / 2
        x0, z0 = obj_pos["x"], obj_pos["z"]
        bbox = Polygon([
            (x0 - hx, z0 - hz), (x0 + hx, z0 - hz),
            (x0 + hx, z0 + hz), (x0 - hx, z0 + hz)
        ])
        if bbox.area <= 0:
            return 0.0
        cx, cz = cam_pos["x"], cam_pos["z"]
        look_dir = np.array(look_direction, dtype=float)
        look_dir = look_dir / np.linalg.norm(look_dir)
        half_fov = math.radians(fov_deg / 2)
        radius = max(20, np.linalg.norm([x0 - cx, z0 - cz]) * 2)
        sector_points = [(cx, cz)]
        for i in range(64 + 1):
            angle = -half_fov + i * (2 * half_fov / 64)
            rot_x = look_dir[0] * math.cos(angle) - look_dir[1] * math.sin(angle)
            rot_z = look_dir[0] * math.sin(angle) + look_dir[1] * math.cos(angle)
            sector_points.append((cx + rot_x * radius, cz + rot_z * radius))
        sector = Polygon(sector_points)
        intersection = bbox.intersection(sector)
        return float(intersection.area / bbox.area)

    def _calculate_object_ratios(self, camera_pos: Dict[str, float], camera_deg: int,
                                 cam_id: str, room_filter: Optional[int] = None) -> List[Dict]:
        """Legacy-compatible visibility metadata for agent/object cameras."""
        object_ratios: List[Dict] = []
        if not self.object_generator:
            return object_ratios

        camera_room_id: Optional[int] = None
        if room_filter is not None:
            camera_room_id = room_filter
        elif cam_id == "agent":
            agent = self.object_generator.get_agent()
            if agent:
                camera_room_id = agent.room_id
        else:
            try:
                obj_id = int(cam_id)
            except ValueError:
                obj_id = None
            if obj_id is not None:
                for obj in self.object_generator.get_all_objects():
                    if obj.object_id == obj_id:
                        camera_room_id = obj.room_id
                        break

        if camera_room_id is None:
            return object_ratios

        room_objects = [
            obj for obj in self.object_generator.get_all_objects()
            if obj.room_id == camera_room_id and str(obj.object_id) != cam_id
        ]

        rad = math.radians(camera_deg)
        look_direction = (math.sin(rad), math.cos(rad))

        for obj in room_objects:
            try:
                bbox_in_view = self._is_object_bbox_in_view(obj, camera_pos, look_direction)
                visibility_ratio = self._compute_visibility_ratio(obj, camera_pos, look_direction)
                if bbox_in_view and visibility_ratio > 0.05:
                    obj_pos = obj.get_final_position()
                    distance = math.hypot(obj_pos["x"] - camera_pos["x"], obj_pos["z"] - camera_pos["z"])
                    is_occluded = distance > 8.0 or visibility_ratio < 0.3
                    object_ratios.append({
                        "object_id": obj.object_id,
                        "model": obj.model,
                        "visibility_ratio": round(float(visibility_ratio), 3),
                        "occlusion_ratio": 1.0 if is_occluded else 0.0
                    })
            except Exception:
                continue
        return object_ratios
    
    def _capture_all_doors(self):
        """Capture photography for all doors"""
        print("[INFO] Capturing door photography...")
        
        door_camera_specs = self.door_handler.get_all_camera_specs()
        
        for spec in door_camera_specs:
            shot_data = self._capture_door_view(spec)
            if shot_data:
                self.door_shots.append(shot_data)
        
        print(f"[INFO] Captured {len(self.door_shots)} door shots")
    
    def _capture_door_view(self, spec: DoorCameraSpec) -> Optional[Dict]:
        """Capture a single door view"""
        try:
            # Calculate look-at position (look towards door center)
            door_info = self.door_handler.get_door_info(spec.door_id)
            if not door_info:
                return None
            
            door_center = door_info.door_info.center
            look_at = {
                "x": door_center[0],
                "y": spec.position["y"],
                "z": door_center[1]
            }
            
            # Position camera
            self.main_cam.teleport(spec.position)
            self.main_cam.look_at(look_at)
            
            # Capture image
            self.image_capture.set(frequency="once", avatar_ids=["main_cam"], pass_masks=["_img"], save=False)
            self.controller.communicate([])
            images = self.image_capture.get_pil_images()
            
            if "main_cam" in images and "_img" in images["main_cam"]:
                # Save image
                filename = f"{spec.label}.png"
                filepath = self.output_dir / filename
                
                img = images["main_cam"]["_img"]
                img.save(filepath)
                
                return {
                    "file": filename,
                    "door_id": spec.door_id,
                    "camera_label": spec.label,
                    "position": spec.position.copy(),
                    "rotation": spec.rotation.copy(),
                    "direction": spec.direction,
                    "door_center": door_center,
                    "door_color": door_info.color_name,
                    "connected_rooms": door_info.door_info.connected_rooms
                }
        
        except Exception as e:
            print(f"[ERROR] Failed to capture door view {spec.label}: {e}")
        
        return None
    
    def create_annotated_topdown(self):
        """Create annotated top-down view showing all rooms, objects, and doors"""
        try:
            print("[INFO] Creating annotated top-down view...")
            
            # Load the top-down image
            top_down_path = self.output_dir / "top_down_global.png"
            if not top_down_path.exists():
                print("[WARN] Top-down image not found, skipping annotation")
                return
            
            img = Image.open(top_down_path).convert("RGBA")
            draw = ImageDraw.Draw(img)
            w, h = img.size
            
            # Try to load font
            font = None
            try:
                font = ImageFont.truetype("arialbd.ttf", 15)
            except:
                try:
                    font = ImageFont.load_default()
                except:
                    pass
            
            # Calculate coordinate conversion
            # Estimate scene size from room bounds
            all_rooms = list(self.room_analyzer.rooms.values())
            if all_rooms:
                min_x = min(room.center[0] for room in all_rooms) - 5
                max_x = max(room.center[0] for room in all_rooms) + 5
                min_z = min(room.center[1] for room in all_rooms) - 5
                max_z = max(room.center[1] for room in all_rooms) + 5
                
                scene_width = max_x - min_x
                scene_height = max_z - min_z
                
                def world_to_pixel(world_x, world_z):
                    pixel_x = w * (world_x - min_x) / scene_width
                    pixel_y = h * (1 - (world_z - min_z) / scene_height)  # Flip Y
                    return (pixel_x, pixel_y)
                
                # Draw agent (blue dot + red arrow)
                agent = self.object_generator.get_agent()
                if agent:
                    px, py = world_to_pixel(agent.pos["x"], agent.pos["z"])
                    
                    # Blue dot
                    r = 12
                    draw.ellipse([(px-r, py-r), (px+r, py+r)], fill=(0, 0, 255, 255))
                    
                    # Red arrow pointing north
                    arrow_length = 30
                    arrow_end_x = px
                    arrow_end_y = py - arrow_length
                    draw.line([(px, py), (arrow_end_x, arrow_end_y)], fill=(255, 0, 0, 255), width=3)
                    
                    # Arrow head
                    head_size = 8
                    arrow_head = [
                        (arrow_end_x, arrow_end_y),
                        (arrow_end_x - head_size//2, arrow_end_y + head_size),
                        (arrow_end_x + head_size//2, arrow_end_y + head_size)
                    ]
                    draw.polygon(arrow_head, fill=(255, 0, 0, 255))
                
                # Draw objects with labels
                object_counter = 1
                for room_id, room_objects in self.object_generator.rooms_objects.items():
                    for obj in room_objects:
                        obj_final_pos = obj.get_final_position()
                        px, py = world_to_pixel(obj_final_pos["x"], obj_final_pos["z"])
                        
                        # Object label
                        label_text = str(object_counter)
                        object_counter += 1
                        
                        # Position label
                        label_x = px + 8
                        label_y = py + 8
                        
                        # Draw label background and text
                        if font:
                            bbox = draw.textbbox((label_x, label_y), label_text, font=font)
                            padding = 2
                            expanded_bbox = (bbox[0] - padding, bbox[1] - padding,
                                           bbox[2] + padding, bbox[3] + padding)
                            draw.rectangle(expanded_bbox, fill=(255, 0, 0, 255))
                            draw.text((label_x, label_y), label_text, fill=(255, 255, 255, 255), font=font)
                        else:
                            draw.rectangle((label_x, label_y, label_x+15, label_y+15), fill=(255, 0, 0, 255))
                            draw.text((label_x+2, label_y+2), label_text, fill=(255, 255, 255, 255))
                
                # Draw doors with their colors
                for door_id, processed_door in self.door_handler.processed_doors.items():
                    door_center = processed_door.door_info.center
                    px, py = world_to_pixel(door_center[0], door_center[1])
                    
                    # Draw door marker in its assigned color
                    color = processed_door.color
                    door_color = (int(color["r"]*255), int(color["g"]*255), int(color["b"]*255), 255)
                    
                    # Door rectangle
                    door_size = 6
                    draw.rectangle([px-door_size, py-door_size, px+door_size, py+door_size], 
                                 fill=door_color, outline=(0, 0, 0, 255), width=2)
                    
                    # Door label using color name instead of ID
                    if font:
                        door_label = processed_door.color_name  # Use color name directly
                        draw.text((px+8, py-8), door_label, fill=(0, 0, 0, 255), font=font)
            
            # Save annotated image
            annotated_path = self.output_dir / "top_down_annotated.png"
            img.save(annotated_path)
            
            self.global_shots.append({
                "type": "top_down_annotated",
                "file": "top_down_annotated.png",
                "camera": "top_down",
                "description": "Annotated top-down view with labels"
            })
            
            print(f"[INFO] Created annotated top-down view: {annotated_path}")
            
        except Exception as e:
            print(f"[ERROR] Failed to create annotated top-down view: {e}")
            import traceback
            traceback.print_exc()
    
    def get_capture_summary(self) -> Dict:
        """Get summary of all captured images"""
        return {
            "global_shots": len(self.global_shots),
            "room_shots": {room_id: len(shots) for room_id, shots in self.room_shots.items()},
            "door_shots": len(self.door_shots),
            "total_images": len(self.global_shots) + sum(len(shots) for shots in self.room_shots.values()) + len(self.door_shots),
            "output_directory": str(self.output_dir),
            "files": {
                "global": [shot["file"] for shot in self.global_shots],
                "rooms": {room_id: [shot["file"] for shot in shots] for room_id, shots in self.room_shots.items()},
                "doors": [shot["file"] for shot in self.door_shots]
            }
        } 
