"""
Metadata Manager Module
======================

Manages and exports comprehensive metadata for multi-room environments:
- Room information and object placement
- Door specifications and connections
- Camera positions and shot metadata
- Complete scene description for analysis
"""

import json
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
from dataclasses import asdict
from datetime import datetime

from .room_analyzer import RoomAnalyzer
from .object_generator import ObjectGenerator, PlacedObject, AgentInfo
from .door_handler import DoorHandler
from .camera_capture import CameraCapture

class MetadataManager:
    """Manages and exports comprehensive metadata for multi-room scenes"""
    
    def __init__(self, room_analyzer: RoomAnalyzer, object_generator: ObjectGenerator,
                 door_handler: DoorHandler, camera_capture: CameraCapture,
                 output_dir: Path, scene_params: Dict):
        """
        Initialize metadata manager
        
        Args:
            room_analyzer: Room layout analysis
            object_generator: Object placement results
            door_handler: Door processing results
            camera_capture: Image capture results
            output_dir: Output directory for metadata files
            scene_params: Scene generation parameters (seed, cell_size, etc.)
        """
        self.room_analyzer = room_analyzer
        self.object_generator = object_generator
        self.door_handler = door_handler
        self.camera_capture = camera_capture
        self.output_dir = Path(output_dir)
        self.scene_params = scene_params
        
        # Metadata structure
        self.metadata = {}
        
        # Build complete metadata
        self._build_metadata()
    
    def _build_metadata(self):
        """Build comprehensive metadata structure"""
        print("[INFO] Building comprehensive metadata...")
        
        # Basic scene information
        self.metadata["scene_info"] = {
            "generation_time": datetime.now().isoformat(),
            "mask_size": [self.room_analyzer.rows, self.room_analyzer.cols],
            "cell_size": self.room_analyzer.cell_size,
            "total_rooms": len(self.room_analyzer.rooms),
            "total_doors": len(self.room_analyzer.doors),
            "total_objects": len(self.object_generator.get_all_objects()),
            **self.scene_params
        }
        
        # Room metadata
        self.metadata["rooms"] = self._build_room_metadata()
        
        # Door metadata  
        self.metadata["doors"] = self._build_door_metadata()
        
        # Object metadata
        self.metadata["objects"] = self._build_object_metadata()
        
        # Agent metadata
        self.metadata["agent"] = self._build_agent_metadata()
        
        # Camera and image metadata
        self.metadata["cameras"] = self._build_camera_metadata()
        self.metadata["images"] = self._build_image_metadata()
        
        # Spatial relationships
        self.metadata["spatial_relationships"] = self._build_spatial_relationships()
        
        # Generation summary
        self.metadata["generation_summary"] = self._build_generation_summary()
        
        print("[INFO] Metadata structure complete")
    
    def _build_room_metadata(self) -> Dict:
        """Build room-specific metadata"""
        rooms_data = {}
        
        for room_id, room_info in self.room_analyzer.rooms.items():
            # Get objects in this room
            room_objects = self.object_generator.get_room_objects(room_id)
            
            # Get agent info if in this room
            agent = self.object_generator.get_agent()
            has_agent = agent and agent.room_id == room_id
            
            rooms_data[str(room_id)] = {
                "room_id": room_id,
                "name": room_info.name,
                "area_cells": room_info.area,
                "usable_area": room_info.usable_area,
                "dimensions": {
                    "width_cells": room_info.width,
                    "height_cells": room_info.height,
                    "width_world": room_info.width * self.room_analyzer.cell_size,
                    "height_world": room_info.height * self.room_analyzer.cell_size
                },
                "center_world": {
                    "x": room_info.center[0],
                    "z": room_info.center[1]
                },
                "bounds_cells": {
                    "min_row": room_info.bounds[0],
                    "max_row": room_info.bounds[1], 
                    "min_col": room_info.bounds[2],
                    "max_col": room_info.bounds[3]
                },
                "bounds_world": self.room_analyzer.get_room_bounds_world(room_id),
                "objects": [
                    {
                        "object_id": obj.object_id,
                        "name": obj.name,
                        "model": obj.model,
                        "position": obj.get_final_position(),
                        "rotation": obj.rot,  # 存储时使用base rotation，不加default_rotation
                        "attributes": obj.attributes
                    }
                    for obj in room_objects
                ],
                "object_count": len(room_objects),
                "has_agent": has_agent,
                "connected_rooms": list(self.room_analyzer.room_connections.get(room_id, set())),
                "max_objects_capacity": self.room_analyzer.get_max_objects_for_room(room_id)
            }
        
        return rooms_data
    
    def _build_door_metadata(self) -> Dict:
        """Build door-specific metadata"""
        doors_data = {}
        
        for door_id, processed_door in self.door_handler.processed_doors.items():
            door_info = processed_door.door_info
            
            doors_data[str(door_id)] = {
                "door_id": door_id,
                "name": door_info.name,
                "center_world": {
                    "x": door_info.center[0],
                    "z": door_info.center[1]
                },
                "connected_rooms": door_info.connected_rooms,
                "room_pairs": [
                    {"room1": room1, "room2": room2}
                    for i, room1 in enumerate(door_info.connected_rooms)
                    for room2 in door_info.connected_rooms[i+1:]
                ] if len(door_info.connected_rooms) >= 2 else [],
                "orientation": door_info.orientation,
                "width_cells": door_info.width,
                "width_world": door_info.width * self.room_analyzer.cell_size,
                "color": {
                    "name": processed_door.color_name,
                    "rgba": processed_door.color
                },
                "camera_positions": [
                    {
                        "direction": spec.direction,
                        "position": spec.position,
                        "rotation": spec.rotation,
                        "label": spec.label
                    }
                    for spec in processed_door.camera_specs
                ]
            }
        
        return doors_data
    
    def _build_object_metadata(self) -> Dict:
        """Build object-specific metadata"""
        all_objects = self.object_generator.get_all_objects()
        
        objects_data = {
            "total_count": len(all_objects),
            "by_room": {
                str(room_id): len(objs) 
                for room_id, objs in self.object_generator.rooms_objects.items()
            },
            "used_types": list(self.object_generator.used_categories),
            "objects": {}
        }
        
        for obj in all_objects:
            objects_data["objects"][str(obj.object_id)] = {
                "object_id": obj.object_id,
                "name": obj.name,
                "model": obj.model,
                "room_id": obj.room_id,
                "position": obj.get_final_position(),
                "rotation": obj.rot,  # 存储时使用base rotation，不加default_rotation
                "size": {
                    "width": obj.size[0],
                    "depth": obj.size[1]
                },
                "scale": obj.scale,
                "color": obj.color,
                "has_orientation": obj.has_orientation,
                "orientation": obj.orientation,
                "attributes": obj.attributes
            }
        
        return objects_data
    
    def _build_agent_metadata(self) -> Optional[Dict]:
        """Build agent-specific metadata"""
        agent = self.object_generator.get_agent()
        if not agent:
            return None
        
        return {
            "position": agent.pos,
            "rotation": agent.rotation,
            "room_id": agent.room_id,
            "room_name": f"room_{agent.room_id}",
            "facing_direction": self._rotation_to_direction(agent.rotation["y"])
        }
    
    def _build_camera_metadata(self) -> Dict:
        """Build camera position metadata"""
        cameras_data = {
            "total_positions": 0,
            "agent_cameras": [],
            "object_cameras": [],
            "door_cameras": []
        }
        
        # Agent camera
        agent = self.object_generator.get_agent()
        if agent:
            agent_camera = {
                "id": "agent",
                "label": "A", 
                "type": "agent",
                "position": agent.pos,
                "rotation": agent.rotation,
                "room_id": agent.room_id
            }
            cameras_data["agent_cameras"].append(agent_camera)
            cameras_data["total_positions"] += 1
        
        # Object cameras
        for obj in self.object_generator.get_all_objects():
            obj_camera = {
                "id": str(obj.object_id),
                "label": str(obj.object_id),
                "type": "object",
                "object_name": obj.name,
                "position": {"x": obj.get_final_position()["x"], "y": 0.8, "z": obj.get_final_position()["z"]},
                "rotation": {"x": 0, "y": 0, "z": 0},  # Objects look towards room center
                "room_id": obj.room_id
            }
            cameras_data["object_cameras"].append(obj_camera)
            cameras_data["total_positions"] += 1
        
        # Door cameras
        for spec in self.door_handler.get_all_camera_specs():
            door_camera = {
                "id": spec.label,
                "label": spec.label,
                "type": "door",
                "door_id": spec.door_id,
                "position": spec.position,
                "rotation": spec.rotation,
                "direction": spec.direction
            }
            cameras_data["door_cameras"].append(door_camera)
            cameras_data["total_positions"] += 1
        
        return cameras_data
    
    def _build_image_metadata(self) -> Dict:
        """Build image capture metadata"""
        capture_summary = self.camera_capture.get_capture_summary()
        
        images_data = {
            "total_images": capture_summary["total_images"],
            "global_views": {
                "count": capture_summary["global_shots"],
                "images": self.camera_capture.global_shots
            },
            "room_views": {
                "rooms_count": len(capture_summary["room_shots"]),
                "total_shots": sum(capture_summary["room_shots"].values()),
                "by_room": {}
            },
            "door_views": {
                "count": capture_summary["door_shots"],
                "images": self.camera_capture.door_shots
            }
        }
        
        # Room view details
        for room_id, shots in self.camera_capture.room_shots.items():
            images_data["room_views"]["by_room"][str(room_id)] = {
                "room_id": room_id,
                "shot_count": len(shots),
                "images": shots
            }
        
        return images_data
    
    def _build_spatial_relationships(self) -> Dict:
        """Build spatial relationship metadata"""
        relationships = {
            "room_connections": {},
            "door_room_connections": {},
            "object_room_assignments": {},
            "agent_visibility": {}
        }
        
        # Room connections through doors
        for room_id, connected_rooms in self.room_analyzer.room_connections.items():
            relationships["room_connections"][str(room_id)] = list(connected_rooms)
        
        # Door-room connections with details
        for door_id, processed_door in self.door_handler.processed_doors.items():
            connected_rooms = processed_door.door_info.connected_rooms
            relationships["door_room_connections"][str(door_id)] = {
                "door_id": door_id,
                "connected_rooms": connected_rooms,
                "connection_pairs": [
                    {"room1": room1, "room2": room2}
                    for i, room1 in enumerate(connected_rooms)
                    for room2 in connected_rooms[i+1:]
                ] if len(connected_rooms) >= 2 else []
            }
        
        # Object-room assignments
        for room_id, objects in self.object_generator.rooms_objects.items():
            relationships["object_room_assignments"][str(room_id)] = [
                {
                    "object_id": obj.object_id,
                    "name": obj.name,
                    "position": obj.pos
                }
                for obj in objects
            ]
        
        # Agent visibility (simplified - could be expanded with actual visibility calculations)
        agent = self.object_generator.get_agent()
        if agent:
            agent_room_objects = self.object_generator.get_room_objects(agent.room_id)
            relationships["agent_visibility"] = {
                "agent_room": agent.room_id,
                "same_room_objects": [
                    {
                        "object_id": obj.object_id,
                        "name": obj.name,
                        "distance": self._calculate_distance(agent.pos, obj.pos)
                    }
                    for obj in agent_room_objects
                ],
                "accessible_rooms": list(self.room_analyzer.room_connections.get(agent.room_id, set()))
            }
        
        return relationships
    
    def _build_generation_summary(self) -> Dict:
        """Build generation process summary"""
        return {
            "room_analysis": self.room_analyzer.export_summary(),
            "object_generation": self.object_generator.export_summary(),
            "door_processing": self.door_handler.export_metadata(),
            "image_capture": self.camera_capture.get_capture_summary(),
            "success_metrics": {
                "rooms_with_objects": len([
                    room_id for room_id, objs in self.object_generator.rooms_objects.items() 
                    if objs
                ]),
                "rooms_without_objects": len([
                    room_id for room_id, objs in self.object_generator.rooms_objects.items() 
                    if not objs
                ]),
                "total_camera_positions": len(self.object_generator.get_all_objects()) + 
                                        (1 if self.object_generator.get_agent() else 0) +
                                        len(self.door_handler.get_all_camera_specs()),
                "unique_object_types": len(self.object_generator.used_categories),
                "door_color_assignments": len(set(self.door_handler.used_colors))
            }
        }
    
    def _rotation_to_direction(self, rotation_y: float) -> str:
        """Convert rotation angle to direction string"""
        angle = rotation_y % 360
        if angle < 45 or angle >= 315:
            return "north"
        elif angle < 135:
            return "east"
        elif angle < 225:
            return "south"
        else:
            return "west"
    
    def _calculate_distance(self, pos1: Dict[str, float], pos2: Dict[str, float]) -> float:
        """Calculate distance between two positions"""
        return ((pos1["x"] - pos2["x"])**2 + (pos1["z"] - pos2["z"])**2)**0.5
    
    def export_metadata(self, filename: str = "multi_room_metadata.json") -> Path:
        """Export complete metadata to JSON file"""
        output_path = self.output_dir / filename
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
            
            print(f"[INFO] Exported complete metadata to: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"[ERROR] Failed to export metadata: {e}")
            raise
    
    def export_room_settings(self, filename: str = "room_settings.json") -> Path:
        """Export room settings in format compatible with existing tools"""
        output_path = self.output_dir / filename
        
        try:
            # Convert to room_setting format
            room_settings = {
                "name": "multi_room_scene",
                "rooms": {},
                "doors": {},
                "agent": None,
                "generation_info": {
                    "total_rooms": len(self.room_analyzer.rooms),
                    "total_doors": len(self.room_analyzer.doors),
                    "total_objects": len(self.object_generator.get_all_objects()),
                    "generation_time": self.metadata["scene_info"]["generation_time"]
                }
            }
            
            # Add room information
            for room_id, room_data in self.metadata["rooms"].items():
                room_settings["rooms"][room_id] = {
                    "name": room_data["name"],
                    "center": [room_data["center_world"]["x"], room_data["center_world"]["z"]],
                    "objects": [
                        {
                            "name": obj["name"],
                            "pos": [obj["position"]["x"], obj["position"]["z"]],
                            "ori": self._rotation_to_orientation_vector(obj["rotation"]["y"])
                        }
                        for obj in room_data["objects"]
                    ]
                }
            
            # Add door information
            for door_id, door_data in self.metadata["doors"].items():
                room_settings["doors"][door_id] = {
                    "name": door_data["name"],
                    "center": [door_data["center_world"]["x"], door_data["center_world"]["z"]],
                    "connected_rooms": door_data["connected_rooms"],
                    "color": door_data["color"]["name"],
                    "orientation": door_data["orientation"]
                }
            
            # Add agent information
            if self.metadata["agent"]:
                agent_data = self.metadata["agent"]
                room_settings["agent"] = {
                    "name": "agent",
                    "pos": [agent_data["position"]["x"], agent_data["position"]["z"]],
                    "ori": self._rotation_to_orientation_vector(agent_data["rotation"]["y"]),
                    "room_id": agent_data["room_id"]
                }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(room_settings, f, indent=2, ensure_ascii=False)
            
            print(f"[INFO] Exported room settings to: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"[ERROR] Failed to export room settings: {e}")
            raise
    
    def _rotation_to_orientation_vector(self, rotation_y: float) -> List[int]:
        """Convert rotation to orientation vector [x, z]"""
        angle = rotation_y % 360
        if angle < 45 or angle >= 315:
            return [0, 1]   # North
        elif angle < 135:
            return [1, 0]   # East
        elif angle < 225:
            return [0, -1]  # South
        else:
            return [-1, 0]  # West
    
    def export_summary_report(self, filename: str = "generation_report.txt") -> Path:
        """Export human-readable summary report"""
        output_path = self.output_dir / filename
        
        try:
            report_lines = []
            report_lines.append("Multi-Room Scene Generation Report")
            report_lines.append("=" * 50)
            report_lines.append(f"Generated: {self.metadata['scene_info']['generation_time']}")
            report_lines.append("")
            
            # Scene overview
            scene_info = self.metadata["scene_info"]
            report_lines.append("Scene Overview:")
            report_lines.append(f"  - Mask size: {scene_info['mask_size'][0]}x{scene_info['mask_size'][1]} cells")
            report_lines.append(f"  - Cell size: {scene_info['cell_size']} units")
            report_lines.append(f"  - Total rooms: {scene_info['total_rooms']}")
            report_lines.append(f"  - Total doors: {scene_info['total_doors']}")
            report_lines.append(f"  - Total objects: {scene_info['total_objects']}")
            report_lines.append("")
            
            # Room details
            report_lines.append("Room Details:")
            for room_id, room_data in self.metadata["rooms"].items():
                report_lines.append(f"  Room {room_id}:")
                report_lines.append(f"    - Objects: {room_data['object_count']}")
                report_lines.append(f"    - Has agent: {room_data['has_agent']}")
                report_lines.append(f"    - Connected to: {room_data['connected_rooms']}")
                if room_data["objects"]:
                    report_lines.append(f"    - Object types: {[obj['name'] for obj in room_data['objects']]}")
            report_lines.append("")
            
            # Door details
            report_lines.append("Door Details:")
            for door_id, door_data in self.metadata["doors"].items():
                report_lines.append(f"  Door {door_id}:")
                report_lines.append(f"    - Color: {door_data['color']['name']}")
                report_lines.append(f"    - Orientation: {door_data['orientation']}")
                report_lines.append(f"    - Connects: {door_data['connected_rooms']}")
            report_lines.append("")
            
            # Agent details
            if self.metadata["agent"]:
                agent_data = self.metadata["agent"]
                report_lines.append("Agent Details:")
                report_lines.append(f"  - Room: {agent_data['room_id']}")
                report_lines.append(f"  - Position: ({agent_data['position']['x']:.2f}, {agent_data['position']['z']:.2f})")
                report_lines.append(f"  - Facing: {agent_data['facing_direction']}")
                report_lines.append("")
            
            # Image capture summary
            image_data = self.metadata["images"]
            report_lines.append("Image Capture Summary:")
            report_lines.append(f"  - Total images: {image_data['total_images']}")
            report_lines.append(f"  - Global views: {image_data['global_views']['count']}")
            report_lines.append(f"  - Room views: {image_data['room_views']['total_shots']}")
            report_lines.append(f"  - Door views: {image_data['door_views']['count']}")
            report_lines.append("")
            
            # Generation metrics
            metrics = self.metadata["generation_summary"]["success_metrics"]
            report_lines.append("Success Metrics:")
            report_lines.append(f"  - Rooms with objects: {metrics['rooms_with_objects']}")
            report_lines.append(f"  - Rooms without objects: {metrics['rooms_without_objects']}")
            report_lines.append(f"  - Camera positions: {metrics['total_camera_positions']}")
            report_lines.append(f"  - Unique object types: {metrics['unique_object_types']}")
            report_lines.append(f"  - Door colors used: {metrics['door_color_assignments']}")
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report_lines))
            
            print(f"[INFO] Exported summary report to: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"[ERROR] Failed to export summary report: {e}")
            raise
    
    def get_metadata(self) -> Dict:
        """Get the complete metadata dictionary"""
        return self.metadata.copy() 