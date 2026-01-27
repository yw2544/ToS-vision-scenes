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
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .room_analyzer import RoomAnalyzer
from .object_generator import ObjectGenerator, PlacedObject, AgentInfo
from .door_handler import DoorHandler, DoorCameraSpec

class CameraCapture:
    """Handles all camera capture operations for multi-room environments"""
    
    def __init__(self, room_analyzer: RoomAnalyzer, object_generator: ObjectGenerator, 
                 door_handler: DoorHandler, controller, output_dir: Path):
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
        
        # Camera components (will be initialized)
        self.main_cam = None
        self.top_cam = None
        self.oblique_cam = None
        self.image_capture = None
        
        # Captured data
        self.room_shots: Dict[int, List[Dict]] = {}  # room_id -> list of shots
        self.door_shots: List[Dict] = []  # door shots
        self.global_shots: List[Dict] = []  # top-down and oblique shots
        
        self.output_dir.mkdir(exist_ok=True, parents=True)
    
    def setup_cameras(self, main_cam, top_cam, oblique_cam, image_capture):
        """Set camera components from main script"""
        self.main_cam = main_cam
        self.top_cam = top_cam  
        self.oblique_cam = oblique_cam
        self.image_capture = image_capture
    
    def capture_all(self) -> bool:
        """
        Capture all required images:
        1. Top-down and oblique views
        2. Room photography (agent + objects)
        3. Door photography
        
        Returns:
            True if capture successful, False otherwise
        """
        try:
            print("[INFO] Starting comprehensive image capture...")
            
            # 1. Capture global views first
            self._capture_global_views()
            
            # 2. Capture each room with objects and agent
            self._capture_all_rooms()
            
            # 3. Capture all doors
            self._capture_all_doors()
            
            print(f"[INFO] Capture complete: {len(self.global_shots)} global, "
                  f"{sum(len(shots) for shots in self.room_shots.values())} room, "
                  f"{len(self.door_shots)} door shots")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Camera capture failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _capture_global_views(self):
        """Capture top-down and oblique views of the entire scene"""
        print("[INFO] Capturing global views...")
        
        # Top-down view
        self.image_capture.set(frequency="once", avatar_ids=["top_down"], pass_masks=["_img"], save=False)
        self.controller.communicate([])
        top_images = self.image_capture.get_pil_images()
        
        if "top_down" in top_images and "_img" in top_images["top_down"]:
            top_img = top_images["top_down"]["_img"]
            top_path = self.output_dir / "top_down_global.png"
            top_img.save(top_path)
            
            self.global_shots.append({
                "type": "top_down",
                "file": "top_down_global.png",
                "camera": "top_down",
                "description": "Complete scene overview"
            })
            
            print(f"[INFO] Saved top-down view: {top_path}")
        
        # Oblique view with segmentation
        self._capture_oblique_with_annotation()
    
    def _capture_oblique_with_annotation(self):
        """Capture oblique view with object annotation"""
        print("[INFO] Capturing oblique view with annotation...")
        
        # Add agent marker if agent exists
        agent_marker_id = None
        agent = self.object_generator.get_agent()
        
        if agent:
            agent_marker_id = self.controller.get_unique_id()
            print(f"[INFO] Adding agent marker at {agent.pos}")
            
            agent_cube_commands = [
                {
                    "$type": "load_primitive_from_resources",
                    "primitive_type": "Cube",
                    "id": agent_marker_id,
                    "position": {"x": agent.pos["x"], "y": 0.05, "z": agent.pos["z"]},
                    "rotation": {"x": 0, "y": 0, "z": 0}
                },
                {
                    "$type": "scale_object",
                    "id": agent_marker_id,
                    "scale_factor": {"x": 0.02, "y": 0.02, "z": 0.02}
                },
                {
                    "$type": "set_color",
                    "id": agent_marker_id,
                    "color": {"r": 0.0, "g": 0.0, "b": 1.0, "a": 1.0}
                }
            ]
            self.controller.communicate(agent_cube_commands)
        
        # Capture oblique view
        self.image_capture.set(frequency="once", avatar_ids=["oblique_cam"], pass_masks=["_img", "_id"], save=False)
        self.controller.communicate([])
        oblique_images = self.image_capture.get_pil_images()
        
        # Remove agent marker
        if agent_marker_id:
            self.controller.communicate([{"$type": "destroy_object", "id": agent_marker_id}])
        
        if "oblique_cam" in oblique_images:
            # Save raw oblique view
            if "_img" in oblique_images["oblique_cam"]:
                oblique_img = oblique_images["oblique_cam"]["_img"]
                oblique_path = self.output_dir / "oblique_global.png"
                oblique_img.save(oblique_path)
                
                self.global_shots.append({
                    "type": "oblique",
                    "file": "oblique_global.png", 
                    "camera": "oblique_cam",
                    "description": "Oblique scene view"
                })
            
            # Create annotated version if segmentation available
            if "_id" in oblique_images["oblique_cam"] and "_img" in oblique_images["oblique_cam"]:
                self._create_annotated_oblique(
                    oblique_images["oblique_cam"]["_img"],
                    oblique_images["oblique_cam"]["_id"],
                    agent_marker_id
                )
    
    def _create_annotated_oblique(self, rgb_img: Image.Image, id_img: Image.Image, agent_marker_id: Optional[int]):
        """Create annotated oblique view with object labels"""
        try:
            from ..pipeline_partial_label_only import annotate_oblique_view
            
            # Get camera specs for annotation
            all_objects = self.object_generator.get_all_objects()
            agent = self.object_generator.get_agent()
            
            # Create camera specs
            cam_specs = []
            if agent:
                cam_specs.append({
                    "id": "agent",
                    "label": "A",
                    "position": agent.pos
                })
            
            for i, obj in enumerate(all_objects, 1):
                cam_specs.append({
                    "id": str(obj.object_id),
                    "label": str(i),
                    "position": {"x": obj.get_final_position()["x"], "y": 0.8, "z": obj.get_final_position()["z"]}
                })
            
            # Create a simple data constructor-like object for projection
            class ProjectionHelper:
                def __init__(self, room_analyzer):
                    self.w = max(room.width for room in room_analyzer.rooms.values()) if room_analyzer.rooms else 10
                    self.d = max(room.height for room in room_analyzer.rooms.values()) if room_analyzer.rooms else 10
                
                def _improved_oblique_projection(self, world_pos, cam_pos, look_at, screen_w, screen_h):
                    # Simplified projection for annotation
                    wx, wy, wz = world_pos["x"], world_pos["y"], world_pos["z"]
                    cx, cy, cz = cam_pos["x"], cam_pos["y"], cam_pos["z"]
                    
                    # Simple perspective projection
                    depth = abs(wz - cz)
                    if depth < 0.1:
                        return None
                    
                    scale_factor = screen_w / (self.w + 4)
                    screen_x = screen_w / 2 + (wx - cx) * scale_factor
                    screen_y = screen_h * 0.8 - (wy + (wz - cz) * 0.3) * scale_factor
                    
                    return (screen_x, screen_y)
            
            projection_helper = ProjectionHelper(self.room_analyzer)
            
            # Annotate the oblique view
            annotated_path = self.output_dir / "oblique_annotated.png"
            annotate_oblique_view(
                rgb_img, id_img, all_objects, str(annotated_path),
                projection_helper, agent_marker_id, cam_specs
            )
            
            self.global_shots.append({
                "type": "oblique_annotated",
                "file": "oblique_annotated.png",
                "camera": "oblique_cam",
                "description": "Annotated oblique view with object labels"
            })
            
            print(f"[INFO] Created annotated oblique view: {annotated_path}")
            
        except Exception as e:
            print(f"[WARN] Failed to create annotated oblique view: {e}")
    
    def _capture_all_rooms(self):
        """Capture photography for all rooms with objects"""
        print("[INFO] Capturing room photography...")
        
        agent = self.object_generator.get_agent()
        
        for room_id, room_objects in self.object_generator.rooms_objects.items():
            print(f"[INFO] Capturing room {room_id} with {len(room_objects)} objects")
            
            room_shots = []
            
            # Agent photography (if agent is in this room)
            if agent and agent.room_id == room_id:
                agent_shots = self._capture_room_agent(agent, room_objects)
                room_shots.extend(agent_shots)
            
            # Object photography
            for obj in room_objects:
                obj_shots = self._capture_room_object(obj, room_objects, agent if agent and agent.room_id == room_id else None)
                room_shots.extend(obj_shots)
            
            self.room_shots[room_id] = room_shots
            print(f"[INFO] Room {room_id}: captured {len(room_shots)} shots")
    
    def _capture_room_agent(self, agent: AgentInfo, room_objects: List[PlacedObject]) -> List[Dict]:
        """Capture agent views in a room"""
        print(f"[INFO] Capturing agent views in room {agent.room_id}")
        
        shots = []
        directions = [0, 90, 180, 270]  # North, East, South, West
        direction_names = {0: "north", 90: "east", 180: "south", 270: "west"}
        
        for direction in directions:
            shot_data = self._capture_perspective_view(
                agent.pos, direction, f"agent_room_{agent.room_id}", "A",
                room_objects, is_agent=True
            )
            if shot_data:
                shot_data.update({
                    "room_id": agent.room_id,
                    "camera_type": "agent",
                    "direction": direction_names[direction]
                })
                shots.append(shot_data)
        
        return shots
    
    def _capture_room_object(self, obj: PlacedObject, room_objects: List[PlacedObject], 
                           agent: Optional[AgentInfo]) -> List[Dict]:
        """Capture object views in a room"""
        print(f"[INFO] Capturing views for {obj.name} in room {obj.room_id}")
        
        shots = []
        directions = [0, 90, 180, 270]
        direction_names = {0: "north", 90: "east", 180: "south", 270: "west"}
        
        # Hide the object being photographed
        self.controller.communicate({
            "$type": "teleport_object",
            "id": obj.object_id,
            "position": {"x": 999, "y": -999, "z": 999}
        })
        
        # Capture from object position
        obj_final_pos = obj.get_final_position()
        obj_camera_pos = {"x": obj_final_pos["x"], "y": 0.8, "z": obj_final_pos["z"]}
        
        for direction in directions:
            shot_data = self._capture_perspective_view(
                obj_camera_pos, direction, f"obj_{obj.object_id}_room_{obj.room_id}", str(obj.object_id),
                [o for o in room_objects if o.object_id != obj.object_id], is_agent=False
            )
            if shot_data:
                shot_data.update({
                    "room_id": obj.room_id,
                    "camera_type": "object",
                    "object_id": obj.object_id,
                    "object_name": obj.name,
                    "direction": direction_names[direction]
                })
                shots.append(shot_data)
        
        # Restore the object
        self.controller.communicate({
            "$type": "teleport_object",
            "id": obj.object_id,
            "position": obj.pos
        })
        
        return shots
    
    def _capture_perspective_view(self, pos: Dict[str, float], direction: int, tag: str, label: str,
                                visible_objects: List[PlacedObject], is_agent: bool = False) -> Optional[Dict]:
        """Capture a single perspective view with direction ray"""
        try:
            rad = math.radians(direction)
            look_at = {
                "x": pos["x"] + math.sin(rad),
                "y": pos["y"],
                "z": pos["z"] + math.cos(rad)
            }
            
            # Position camera
            self.main_cam.teleport(pos)
            self.main_cam.look_at(look_at)
            
            # Create direction ray
            ray_ids = self._create_direction_ray(pos, direction)
            
            # Capture image
            self.image_capture.set(frequency="once", avatar_ids=["main_cam"], pass_masks=["_img"], save=False)
            self.controller.communicate([])
            images = self.image_capture.get_pil_images()
            
            # Remove direction ray
            for ray_id in ray_ids:
                self.controller.communicate([{"$type": "destroy_object", "id": ray_id}])
            
            if "main_cam" in images and "_img" in images["main_cam"]:
                # Save image
                direction_name = {0: "north", 90: "east", 180: "south", 270: "west"}[direction]
                filename = f"{tag}_facing_{direction_name}.png"
                filepath = self.output_dir / filename
                
                img = images["main_cam"]["_img"]
                img.save(filepath)
                
                return {
                    "file": filename,
                    "camera_id": tag,
                    "camera_label": label,
                    "position": pos.copy(),
                    "direction": direction,
                    "is_agent": is_agent,
                    "visible_objects": len(visible_objects)
                }
        
        except Exception as e:
            print(f"[ERROR] Failed to capture perspective view {tag}_{direction}: {e}")
        
        return None
    
    def _create_direction_ray(self, pos: Dict[str, float], direction: int) -> List[int]:
        """Create direction ray on ground (copied from pipeline_partial_label_only.py)"""
        try:
            ray_length = 10  # Shorter rays for rooms
            dash_len = 0.1
            gap_len = 0.05
            
            rad = math.radians(direction)
            dir_x, dir_z = math.sin(rad), math.cos(rad)
            
            num_segments = int(ray_length // (dash_len + gap_len))
            segment_ids = []
            commands = []
            
            for i in range(num_segments):
                t = (gap_len/2) + i * (dash_len + gap_len) + (dash_len/2)
                cx = pos["x"] + dir_x * t
                cz = pos["z"] + dir_z * t
                
                seg_id = self.controller.get_unique_id()
                segment_ids.append(seg_id)
                
                commands.extend([
                    {
                        "$type": "load_primitive_from_resources",
                        "primitive_type": "Cube",
                        "id": seg_id,
                        "position": {"x": cx, "y": 0.02, "z": cz}
                    },
                    {
                        "$type": "scale_object",
                        "id": seg_id,
                        "scale_factor": {"x": dash_len if direction in (90, 270) else 0.02,
                                       "y": 0.005,
                                       "z": dash_len if direction in (0, 180) else 0.02}
                    },
                    {
                        "$type": "set_color",
                        "id": seg_id,
                        "color": {"r": 1.0, "g": 0.0, "b": 0.0, "a": 1.0}
                    }
                ])
            
            if commands:
                self.controller.communicate(commands)
            
            return segment_ids
            
        except Exception as e:
            print(f"[WARN] Failed to create direction ray: {e}")
            return []
    
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