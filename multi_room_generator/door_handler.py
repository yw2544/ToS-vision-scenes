"""
Door Handler Module
==================

Handles door-related operations:
- Assigns special colors to doors
- Manages door photography from 4 directions
- Tracks door metadata (position, orientation, connected rooms)
"""

import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import math

from .room_analyzer import RoomAnalyzer, DoorInfo


DOOR_COLORS = {
  "yellow": {"r": 0.988, "g": 0.776, "b": 0.114, "a": 1.0},
  "blue": {"r": 0.149, "g": 0.300, "b": 0.498, "a": 1.0},
  "green": {"r": 0.541, "g": 0.651, "b": 0.141, "a": 1.0},
  "red": {"r": 0.549, "g": 0.063, "b": 0.027, "a": 1.0},
  "brown": {"r": 0.65, "g": 0.58, "b": 0.50, "a": 1.0}
}
 


@dataclass 
class DoorCameraSpec:
    """Camera specification for door photography"""
    door_id: int
    position: Dict[str, float]
    rotation: Dict[str, float]
    direction: str  # "north", "east", "south", "west"
    label: str  # Camera label for identification

@dataclass
class ProcessedDoor:
    """A door with assigned color and camera specifications"""
    door_info: DoorInfo
    color: Dict[str, float]
    color_name: str
    camera_specs: List[DoorCameraSpec]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for metadata storage"""
        return {
            "door_id": self.door_info.door_id,
            "name": self.door_info.name,
            "center": self.door_info.center,
            "connected_rooms": self.door_info.connected_rooms,
            "orientation": self.door_info.orientation,
            "width": self.door_info.width,
            "color": {
                "name": self.color_name,
                **self.color
            },
            "camera_positions": [
                {
                    "direction": spec.direction,
                    "position": spec.position,
                    "rotation": spec.rotation,
                    "label": spec.label
                }
                for spec in self.camera_specs
            ]
        }

class DoorHandler:
    """Handles door processing: colors, camera positioning, and metadata"""
    
    def __init__(self, room_analyzer: RoomAnalyzer, seed: int = 42):
        """
        Initialize door handler
        
        Args:
            room_analyzer: Analyzed room layout with door information
            seed: Random seed for color assignment
        """
        self.room_analyzer = room_analyzer
        self.rng = random.Random(seed)
        
        # Processing results
        self.processed_doors: Dict[int, ProcessedDoor] = {}
        self.used_colors: List[str] = []
        
        # Process all doors
        self._process_all_doors()
    
    def _process_all_doors(self):
        """Process all doors: assign colors and create camera specs"""
        print("[INFO] Processing doors...")
        
        doors = list(self.room_analyzer.doors.values())
        if not doors:
            print("[INFO] No doors found to process")
            return
        
        # Shuffle available colors
        available_colors = list(DOOR_COLORS.keys())
        self.rng.shuffle(available_colors)
        
        for i, door in enumerate(doors):
            # Assign color (cycle through colors if we have more doors than colors)
            # With 6 colors available: 红橙黄蓝绿紫
            color_name = available_colors[i % len(available_colors)]
            color = DOOR_COLORS[color_name]
            
            # Create camera specifications for this door
            camera_specs = self._create_door_camera_specs(door)
            
            # Create processed door
            processed_door = ProcessedDoor(
                door_info=door,
                color=color,
                color_name=color_name,
                camera_specs=camera_specs
            )
            
            self.processed_doors[door.door_id] = processed_door
            self.used_colors.append(color_name)
            
            print(f"[INFO] Door {door.door_id}: color={color_name}, cameras={len(camera_specs)}")
        
        print(f"[INFO] Processed {len(self.processed_doors)} doors")
    
    def _create_door_camera_specs(self, door: DoorInfo) -> List[DoorCameraSpec]:
        """Create camera specifications for photographing a door from 4 directions"""
        camera_specs = []
        
        # Calculate camera positions around the door
        door_x, door_z = door.center
        camera_distance = 2.0  # Distance from door for photography
        camera_height = 1.6    # Camera height for door photography
        
        # Define 4 directions and their offsets
        directions = [
            ("north", 0, -camera_distance, 0),    # North: look south towards door
            ("east", camera_distance, 0, 270),    # East: look west towards door  
            ("south", 0, camera_distance, 180),   # South: look north towards door
            ("west", -camera_distance, 0, 90)     # West: look east towards door
        ]
        
        for direction_name, offset_x, offset_z, rotation_y in directions:
            # Calculate camera position
            cam_x = door_x + offset_x
            cam_z = door_z + offset_z
            
            # Create camera specification
            camera_spec = DoorCameraSpec(
                door_id=door.door_id,
                position={"x": cam_x, "y": camera_height, "z": cam_z},
                rotation={"x": 0, "y": rotation_y, "z": 0},
                direction=direction_name,
                label=f"door_{door.door_id}_{direction_name}"
            )
            
            camera_specs.append(camera_spec)
        
        return camera_specs
    
    def get_door_colors_for_tdw(self) -> List[Dict]:
        """Get door color commands for TDW scene building"""
        commands = []
        
        for door_id, processed_door in self.processed_doors.items():
            # TDW command to set door color
            color_command = {
                "$type": "set_color",
                "id": door_id,  # This should match the door object ID in TDW
                "color": processed_door.color
            }
            commands.append(color_command)
        
        return commands
    
    def get_all_camera_specs(self) -> List[DoorCameraSpec]:
        """Get all door camera specifications for photography"""
        all_specs = []
        for processed_door in self.processed_doors.values():
            all_specs.extend(processed_door.camera_specs)
        return all_specs
    
    def get_door_info(self, door_id: int) -> Optional[ProcessedDoor]:
        """Get processed door information"""
        return self.processed_doors.get(door_id)
    
    def get_doors_between_rooms(self, room1_id: int, room2_id: int) -> List[ProcessedDoor]:
        """Get doors that connect two specific rooms"""
        connecting_doors = []
        
        for processed_door in self.processed_doors.values():
            connected_rooms = processed_door.door_info.connected_rooms
            if (room1_id in connected_rooms and room2_id in connected_rooms and 
                len(connected_rooms) == 2):
                connecting_doors.append(processed_door)
        
        return connecting_doors
    
    def validate_door_positions(self) -> bool:
        """Validate that all door camera positions are reasonable"""
        print("[INFO] Validating door camera positions...")
        
        valid = True
        for door_id, processed_door in self.processed_doors.items():
            door_center = processed_door.door_info.center
            
            for camera_spec in processed_door.camera_specs:
                cam_pos = camera_spec.position
                
                # Check camera is not too far from door
                distance = math.sqrt(
                    (cam_pos["x"] - door_center[0])**2 + 
                    (cam_pos["z"] - door_center[1])**2
                )
                
                if distance > 5.0:  # Max reasonable distance
                    print(f"[WARN] Door {door_id} camera {camera_spec.direction} too far from door: {distance:.2f}m")
                    valid = False
                
                # Check camera height is reasonable
                if cam_pos["y"] < 0.5 or cam_pos["y"] > 3.0:
                    print(f"[WARN] Door {door_id} camera {camera_spec.direction} unreasonable height: {cam_pos['y']:.2f}m")
                    valid = False
        
        if valid:
            print("[INFO] All door camera positions are valid")
        else:
            print("[WARN] Some door camera positions may be problematic")
        
        return valid
    
    def export_metadata(self) -> Dict:
        """Export door metadata for storage"""
        return {
            "total_doors": len(self.processed_doors),
            "used_colors": self.used_colors,
            "doors": {
                door_id: processed_door.to_dict()
                for door_id, processed_door in self.processed_doors.items()
            },
            "door_connections": {
                door_id: {
                    "connects_rooms": processed_door.door_info.connected_rooms,
                    "room_pairs": [
                        tuple(sorted([room1, room2])) 
                        for i, room1 in enumerate(processed_door.door_info.connected_rooms)
                        for room2 in processed_door.door_info.connected_rooms[i+1:]
                    ] if len(processed_door.door_info.connected_rooms) >= 2 else []
                }
                for door_id, processed_door in self.processed_doors.items()
            }
        }
    
    def get_door_summary(self) -> str:
        """Get a human-readable summary of door processing"""
        if not self.processed_doors:
            return "No doors processed"
        
        summary_lines = [f"Processed {len(self.processed_doors)} doors:"]
        
        for door_id, processed_door in self.processed_doors.items():
            door_info = processed_door.door_info
            connected_rooms = door_info.connected_rooms
            
            if len(connected_rooms) == 2:
                room_connection = f"connects rooms {connected_rooms[0]} and {connected_rooms[1]}"
            elif len(connected_rooms) == 1:
                room_connection = f"connects to room {connected_rooms[0]}"
            else:
                room_connection = f"connects {len(connected_rooms)} rooms"
            
            summary_lines.append(
                f"  Door {door_id}: {processed_door.color_name}, "
                f"{door_info.orientation}, {room_connection}"
            )
        
        return "\n".join(summary_lines) 