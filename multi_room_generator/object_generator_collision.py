"""
Collision and helper utilities for ObjectGenerator.
"""

import math
from typing import Tuple

from .object_generator_types import PlacedObject


def _overlaps_with_existing(self, x: float, z: float, w: float, d: float, room_id: int) -> bool:
    """Enhanced collision detection with bounding box overlap checking"""
    # First check if the grid position is already occupied
    if self.use_grid_occupancy:
        check_row, check_col = self.room_analyzer._world_to_cell(x, z)
        if (int(check_row), int(check_col)) in self.occupied_rc:
            return True

    # Use 8-neighbor grid check instead of distance calculation
    int_x, int_z = int(round(x)), int(round(z))
    return self._check_8_neighbor_conflicts(int_x, int_z, room_id, check_doors=True, check_agent=True, check_objects=True)


def _overlaps_with_existing_with_default(self, x: float, z: float, w: float, d: float, room_id: int, obj_config: dict) -> bool:
    """Enhanced collision detection using 8-neighbor grid check"""
    # Calculate candidate's final center = base(x,z) + default_position offset
    default_position = obj_config.get("default_position", {})
    dx = default_position.get("x", 0.0)
    dz = default_position.get("z", 0.0)
    fx, fz = x + float(dx), z + float(dz)

    # Convert to integer grid coordinates
    int_fx, int_fz = int(round(fx)), int(round(fz))

    # Check the 8-neighbor grid around the final position
    return self._check_8_neighbor_conflicts(int_fx, int_fz, room_id, check_doors=True, check_agent=True, check_objects=True)


def _check_8_neighbor_conflicts(self, center_x: int, center_z: int, room_id: int,
                               check_doors: bool = True, check_agent: bool = True,
                               check_objects: bool = True) -> bool:
    """
    Check whether a candidate position is within the 8-neighborhood
    of any object/agent/door.

    Args:
        center_x, center_z: Candidate integer coordinates
        room_id: Room ID
        check_doors: Whether to check door conflicts
        check_agent: Whether to check agent conflicts
        check_objects: Whether to check object conflicts

    Returns:
        True if the candidate is within any 8-neighborhood (conflict),
        False if safe
    """
    # Door conflicts: candidate must not be within any door's 8-neighborhood
    if check_doors and hasattr(self.room_analyzer, 'doors'):
        for door_id, door_info in self.room_analyzer.doors.items():
            connected_rooms = getattr(door_info, "connected_rooms", [])

            # Check all doors, regardless of connected rooms
            door_x, door_z = door_info.center
            int_door_x, int_door_z = int(round(door_x)), int(round(door_z))

            # Strict 8-neighborhood check for doors
            if abs(center_x - int_door_x) <= 1 and abs(center_z - int_door_z) <= 1:
                return True

    # Agent conflicts: candidate in agent's 8-neighborhood
    if check_agent and self.agent and self.agent.room_id == room_id:
        agent_x, agent_z = int(round(self.agent.pos["x"])), int(round(self.agent.pos["z"]))
        if abs(center_x - agent_x) <= 1 and abs(center_z - agent_z) <= 1:
            return True

    # Object conflicts: candidate in any object's 8-neighborhood
    if check_objects:
        room_objects = self.rooms_objects.get(room_id, [])
        for obj in room_objects:
            obj_final_pos = obj.get_final_position()
            obj_x, obj_z = int(round(obj_final_pos["x"])), int(round(obj_final_pos["z"]))
            if abs(center_x - obj_x) <= 1 and abs(center_z - obj_z) <= 1:
                return True
            else:
                continue

    return False

def _check_bounding_box_overlap(self, x1: float, z1: float, w1: float, d1: float,
                               x2: float, z2: float, w2: float, d2: float,
                               safety_margin: float = 0.3) -> bool:
    """
    Check if two axis-aligned bounding boxes overlap with safety margin

    Args:
        x1, z1, w1, d1: First object center position and dimensions
        x2, z2, w2, d2: Second object center position and dimensions
        safety_margin: Additional safety distance between objects

    Returns:
        True if bounding boxes overlap (including safety margin)
    """
    # Calculate half-dimensions with safety margin
    half_w1 = w1 / 2 + safety_margin
    half_d1 = d1 / 2 + safety_margin
    half_w2 = w2 / 2 + safety_margin
    half_d2 = d2 / 2 + safety_margin

    # Check for overlap in both X and Z axes
    x_overlap = abs(x1 - x2) < (half_w1 + half_w2)
    z_overlap = abs(z1 - z2) < (half_d1 + half_d2)

    return x_overlap and z_overlap


def _check_center_distance(self, x1: float, z1: float, x2: float, z2: float,
                          min_distance: float) -> bool:
    """
    Check whether the distance between two centers meets the minimum requirement.

    Args:
        x1, z1: First object center
        x2, z2: Second object center
        min_distance: Minimum allowed distance

    Returns:
        True if distance < min_distance (too close, not allowed)
        False if distance >= min_distance (safe)
    """
    distance = math.sqrt((x1 - x2)**2 + (z1 - z2)**2)
    # Strict check: distance must be >= min_distance to allow placement
    result = distance < min_distance
    if result:
        print(f"[DEBUG] Distance check failed: {distance:.3f} < min_distance {min_distance}")
    return result


def _final_center_from_candidate(self, x: float, z: float, obj_cfg_or_obj) -> Tuple[float, float]:
    """
    Convert candidate base position to final center coordinates
    (including default_position offset).

    Args:
        x, z: Candidate base position
        obj_cfg_or_obj: Object config dict or PlacedObject

    Returns:
        Final center coordinates (fx, fz)
    """
    # Support dict (obj_config) or PlacedObject
    if isinstance(obj_cfg_or_obj, dict):
        dp = obj_cfg_or_obj.get("default_position", {}) or {}
    else:
        # PlacedObject
        cfg = obj_cfg_or_obj.model_config or {}
        dp = cfg.get("default_position", {}) or {}

    return x + float(dp.get("x", 0.0)), z + float(dp.get("z", 0.0))


def _mark_object_occupied(self, placed_obj: PlacedObject):
    """
    Mark both base and final grid positions as occupied.
    """
    # Mark base position
    base_r, base_c = self.room_analyzer._world_to_cell(placed_obj.pos["x"], placed_obj.pos["z"])
    self.occupied_rc.add((int(base_r), int(base_c)))

    # Mark final position (if different from base)
    final_pos = placed_obj.get_final_position()
    final_r, final_c = self.room_analyzer._world_to_cell(final_pos["x"], final_pos["z"])
    if (int(final_r), int(final_c)) != (int(base_r), int(base_c)):
        self.occupied_rc.add((int(final_r), int(final_c)))


def _unmark_object_occupied(self, placed_obj: PlacedObject):
    """
    Clear occupied state for both base and final grid positions.
    """
    # Clear base position
    base_r, base_c = self.room_analyzer._world_to_cell(placed_obj.pos["x"], placed_obj.pos["z"])
    self.occupied_rc.discard((int(base_r), int(base_c)))

    # Clear final position (if different from base)
    final_pos = placed_obj.get_final_position()
    final_r, final_c = self.room_analyzer._world_to_cell(final_pos["x"], final_pos["z"])
    if (int(final_r), int(final_c)) != (int(base_r), int(base_c)):
        self.occupied_rc.discard((int(final_r), int(final_c)))


def _get_object_bounds(self, model_name: str) -> Tuple[float, float]:
    """Get object 2D bounds from model library or custom models"""
    # Check custom models from models_by_category first
    for category, models in self.models_by_category.items():
        for obj_config in models:
            if obj_config["model"] == model_name and obj_config.get("is_custom_model", False):
                # Custom model: try to get bounds from record file
                record_path = obj_config.get("record")
                if record_path:
                    try:
                        import json
                        from tdw.librarian import ModelRecord
                        from pathlib import Path

                        record_path = Path(record_path)
                        if record_path.exists():
                            record_data_text = record_path.read_text(encoding='utf-8')
                            record = ModelRecord(json.loads(record_data_text))
                            bounds = record.bounds
                            width = abs(bounds['right']['x'] - bounds['left']['x'])
                            depth = abs(bounds['front']['z'] - bounds['back']['z'])
                            return (width, depth)
                    except Exception as e:
                        print(f"[WARN] Failed to read record bounds for custom model {model_name}: {e}")

                # Fallback to default bounds if record is unavailable
                print(f"[WARN] Custom model {model_name} using default bounds")
                return (1.0, 1.0)

    # Standard library models
    if not self.model_lib:
        return (1.0, 1.0)  # Default

    try:
        record = self.model_lib.get_record(model_name)
        if record is None:
            print(f"[WARN] Model {model_name} not found in library, using default bounds")
            return (1.0, 1.0)
        bounds = record.bounds
        width = abs(bounds['right']['x'] - bounds['left']['x'])
        depth = abs(bounds['front']['z'] - bounds['back']['z'])
        return (width, depth)
    except Exception as e:
        print(f"[WARN] Failed to get bounds for {model_name}: {e}")
        return (1.0, 1.0)


def _rotation_to_orientation(self, rotation_y: int) -> str:
    """Convert rotation angle to orientation string"""
    angle = rotation_y % 360
    if angle == 0:
        return "north"
    elif angle == 90:
        return "east"
    elif angle == 180:
        return "south"
    elif angle == 270:
        return "west"
    else:
        # Round to nearest 90 degrees
        rounded = round(angle / 90) * 90 % 360
        return self._rotation_to_orientation(rounded)
