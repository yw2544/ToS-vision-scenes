#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Collinear Position Validator
===========================

Check whether object positions in a room are collinear, avoiding three or more
objects (including the agent) on the same line. Use a 0.8-width tolerance band
to judge collinearity.
"""

import math
from typing import List, Dict, Tuple, Optional, Any
import numpy as np


def point_to_line_distance(px: float, py: float,
                          x1: float, y1: float,
                          x2: float, y2: float) -> float:
    """
    Compute the distance from a point to a line.
    
    Args:
        px, py: point coordinates
        x1, y1, x2, y2: two points on the line
    
    Returns:
        The shortest distance from the point to the line.
    """
    # If the two points coincide, return the point-to-point distance.
    if abs(x2 - x1) < 1e-6 and abs(y2 - y1) < 1e-6:
        return math.sqrt((px - x1)**2 + (py - y1)**2)
    
    # Point-to-line distance: |ax + by + c| / sqrt(a^2 + b^2)
    # Line equation: (y2-y1)x - (x2-x1)y + (x2-x1)y1 - (y2-y1)x1 = 0
    a = y2 - y1
    b = -(x2 - x1)
    c = (x2 - x1) * y1 - (y2 - y1) * x1
    
    distance = abs(a * px + b * py + c) / math.sqrt(a**2 + b**2)
    return distance


def are_three_points_collinear(p1: Tuple[float, float],
                              p2: Tuple[float, float],
                              p3: Tuple[float, float],
                              tolerance: float = 0.5) -> bool:
    """
    Check whether three points are collinear (within tolerance band).
    
    Args:
        p1, p2, p3: three point coordinates (x, z)
        tolerance: tolerance distance (half of a 0.8-width band)
    
    Returns:
        True if the three points are collinear, False otherwise.
    """
    x1, z1 = p1
    x2, z2 = p2  
    x3, z3 = p3
    
    # Check distance from each point to the line through the other two points.
    distances = [
        point_to_line_distance(x1, z1, x2, z2, x3, z3),  # p1 to line(p2, p3)
        point_to_line_distance(x2, z2, x1, z1, x3, z3),  # p2 to line(p1, p3)
        point_to_line_distance(x3, z3, x1, z1, x2, z2)   # p3 to line(p1, p2)
    ]
    
    # If any point is within tolerance of the other two-point line, treat as collinear.
    min_distance = min(distances)
    is_collinear = min_distance <= tolerance
    
    return is_collinear


def check_room_collinearity(positions: List[Tuple[float, float]],
                           names: List[str],
                           tolerance: float = 0.5) -> List[Tuple[int, int, int]]:
    """
    Check whether there are collinear positions in a room.
    
    Args:
        positions: position list [(x, z), ...], includes objects and agent
        names: corresponding names for debugging
        tolerance: collinearity tolerance (half of a 0.8-width band)
    
    Returns:
        Index triplets [(i, j, k), ...] where positions[i], positions[j], positions[k] are collinear.
    """
    if len(positions) < 3:
        return []
    
    collinear_groups = []
    n = len(positions)
    
    # Check all triplets.
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if are_three_points_collinear(positions[i], positions[j], positions[k], tolerance):
                    collinear_groups.append((i, j, k))
    return collinear_groups


class CollinearValidator:
    """Collinearity validator for room object positions."""
    
    def __init__(self, tolerance_width: float = 0.8):
        """
        Args:
            tolerance_width: width of the collinearity tolerance band
        """
        self.tolerance = tolerance_width / 2.0  # Use half as point-to-line threshold
        
    def validate_room_positions(self, 
                               room_objects: List[Any],
                               agent_pos: Optional[Dict[str, float]] = None,
                               room_id: int = None,
                               room_analyzer: Any = None) -> Tuple[bool, List[Tuple[int, int, int]]]:
        """
        Validate whether room positions are collinear.
        
        Args:
            room_objects: room objects list (PlacedObject)
            agent_pos: agent position dict {"x": x, "z": z}
            room_id: room id for logging
            room_analyzer: room analyzer for door positions
        
        Returns:
            (is_valid, collinear_groups)
            is_valid: True means no collinearity issues
            collinear_groups: index triplets that are collinear
        """
        # Collect positions and names.
        positions = []
        names = []
        
        # Add object positions.
        for obj in room_objects:
            # Use final position (includes default_position offset).
            final_pos = obj.get_final_position()
            positions.append((final_pos["x"], final_pos["z"]))
            names.append(obj.name)
        
        # Add agent position (if present and in the same room).
        if agent_pos:
            # Handle different agent_pos formats.
            if hasattr(agent_pos, 'room_id') and hasattr(agent_pos, 'pos'):
                # AgentInfo object.
                if room_id is None or agent_pos.room_id == room_id:
                    positions.append((agent_pos.pos["x"], agent_pos.pos["z"]))
                    names.append("agent")
            elif isinstance(agent_pos, dict):
                # Dict format.
                if room_id is None or agent_pos.get('room_id') == room_id:
                    agent_x = agent_pos.get("x", 0.0)
                    agent_z = agent_pos.get("z", 0.0)
                    positions.append((agent_x, agent_z))
                    names.append("agent")
        
        # Add door positions for the room.
        if room_analyzer and room_id is not None:
            door_positions_in_room = self._get_doors_for_room(room_analyzer, room_id)
            for door_name, door_pos in door_positions_in_room:
                positions.append(door_pos)
                names.append(door_name)
        if len(positions) < 3:
            return True, []  # Fewer than 3 positions cannot be collinear.

        # Check collinearity.
        collinear_groups = check_room_collinearity(positions, names, self.tolerance)
        
        is_valid = len(collinear_groups) == 0
        
        status = "pass" if is_valid else "fail"
        print(f"[INFO] Collinearity check room {room_id}: {status} ({len(collinear_groups)} groups)")
        
        return is_valid, collinear_groups
    
    def _get_doors_for_room(self, room_analyzer: Any, room_id: int) -> List[Tuple[str, Tuple[float, float]]]:
        """
        Get door positions related to a room.
        
        Args:
            room_analyzer: room analyzer
            room_id: room id
        
        Returns:
            [(door_name, (x, z)), ...] list of door names and positions
        """
        door_positions = []
        
        if not hasattr(room_analyzer, 'doors'):
            return door_positions
        
        for door_id, door_info in room_analyzer.doors.items():
            # Check whether the door connects to the room.
            if room_id in door_info.connected_rooms:
                door_name = f"door_{door_id}"
                door_x, door_z = door_info.center
                door_positions.append((door_name, (door_x, door_z)))
        
        return door_positions
    
    def get_repositioning_suggestions(self, 
                                    room_objects: List[Any],
                                    collinear_groups: List[Tuple[int, int, int]],
                                    room_cells: List[Tuple[int, int]],
                                    room_analyzer: Any,
                                    agent_pos: Any = None) -> List[Dict[str, Any]]:
        """
        Provide relocation suggestions for collinear objects.

        Args:
            room_objects: room objects list
            collinear_groups: collinear triplets
            room_cells: available room cells
            room_analyzer: room analyzer

        Returns:
            list of relocation suggestions
        """
        suggestions = []
        
        for group_idx, (i, j, k) in enumerate(collinear_groups):
            # Find object indices in the triplet (exclude doors/agent).
            obj_indices = []
            for idx in [i, j, k]:
                if idx < len(room_objects): 
                    obj_indices.append(idx)
            
            # Skip if no movable objects.
            if not obj_indices:
                continue
            
            # Move the last object in the triplet.
            obj_to_move_idx = obj_indices[-1]
            obj_to_move = room_objects[obj_to_move_idx]
            
            # Find alternative positions.
            alternative_positions = self._find_alternative_positions(
                obj_to_move, room_objects, room_cells, room_analyzer, agent_pos
            )
            
            suggestion = {
                "group_index": group_idx,
                "object_index": obj_to_move_idx,
                "object_name": obj_to_move.name,
                "current_position": obj_to_move.get_final_position(),
                "alternative_positions": alternative_positions[:3]  # Up to 3 suggestions
            }
            suggestions.append(suggestion)
        
        return suggestions
    
    def _find_alternative_positions(self, 
                                  target_obj: Any,
                                  room_objects: List[Any], 
                                  room_cells: List[Tuple[int, int]],
                                  room_analyzer: Any,
                                  agent_pos: Any = None) -> List[Dict[str, float]]:
        """Find alternative positions."""
        alternatives = []
        
        # Convert room_cells to world coordinates.
        for row, col in room_cells:
            world_x, world_z = room_analyzer._cell_to_world(row, col)
            int_x, int_z = int(round(world_x)), int(round(world_z))
            
            # Check if position is valid and avoids new conflicts.
            if self._is_position_valid_for_relocation(
                float(int_x), float(int_z), target_obj, room_objects, room_analyzer, agent_pos
            ):
                alternatives.append({"x": float(int_x), "y": 0.05, "z": float(int_z)})
                
                if len(alternatives) >= 5:  # Limit candidate count
                    break
        
        return alternatives
    
    def _is_position_valid_for_relocation(self, 
                                        x: float, z: float,
                                        target_obj: Any,
                                        room_objects: List[Any],
                                        room_analyzer: Any,
                                        agent_pos: Any = None) -> bool:
        """Check whether relocation is valid."""
        # Check if inside the room.
        if not room_analyzer.is_position_valid_for_placement(x, z, target_obj.room_id):
            return False
        
        # Check distance to other objects (basic collision check).
        min_distance = 1.1  # Minimum distance
        for obj in room_objects:
            if obj.object_id != target_obj.object_id:
                final_pos = obj.get_final_position()
                obj_x, obj_z = final_pos["x"], final_pos["z"]
                distance = ((x - obj_x) ** 2 + (z - obj_z) ** 2) ** 0.5
                if distance < min_distance:
                    return False
        
        # Check exact overlap with other objects.
        for obj in room_objects:
            if obj.object_id != target_obj.object_id:
                final_pos = obj.get_final_position()
                if abs(x - final_pos["x"]) < 0.1 and abs(z - final_pos["z"]) < 0.1:
                    return False
        
        # Check distance to agent.
        if agent_pos:
            agent_x, agent_z = None, None
            if hasattr(agent_pos, 'room_id') and hasattr(agent_pos, 'pos'):
                if agent_pos.room_id == target_obj.room_id:
                    agent_x, agent_z = agent_pos.pos["x"], agent_pos.pos["z"]
            elif isinstance(agent_pos, dict):
                if agent_pos.get('room_id') == target_obj.room_id:
                    agent_x, agent_z = agent_pos.get("x", 0.0), agent_pos.get("z", 0.0)
            
            if agent_x is not None and agent_z is not None:
                distance = ((x - agent_x) ** 2 + (z - agent_z) ** 2) ** 0.5
                if distance < min_distance:
                    return False
        
        # Check distance to doors.
        if room_analyzer:
            door_positions_in_room = self._get_doors_for_room(room_analyzer, target_obj.room_id)
            for door_name, door_pos in door_positions_in_room:
                door_x, door_z = door_pos
                distance = ((x - door_x) ** 2 + (z - door_z) ** 2) ** 0.5
                if distance < min_distance:
                    return False
        
        # Do basic checks only; allow re-validation after applying fixes.
        return True


def apply_collinear_fix(room_objects: List[Any],
                       suggestions: List[Dict[str, Any]],
                       room_analyzer: Any) -> bool:
    """
    Apply collinearity fix suggestions.

    Args:
        room_objects: room objects list
        suggestions: relocation suggestions
        room_analyzer: room analyzer

    Returns:
        True if fixes were applied
    """
    fixed_count = 0
    used_positions = set()  # Track used positions
    
    for suggestion in suggestions:
        obj_idx = suggestion["object_index"]
        alternatives = suggestion["alternative_positions"]
        
        if obj_idx < len(room_objects) and alternatives:
            obj = room_objects[obj_idx]
            
            # Pick the first unused position.
            selected_pos = None
            for alt_pos in alternatives:
                pos_key = (alt_pos["x"], alt_pos["z"])
                if pos_key not in used_positions:
                    selected_pos = alt_pos
                    used_positions.add(pos_key)
                    break
            
            if selected_pos is None:
                print(f"[WARN] No available non-overlapping position for {obj.name}, skip")
                continue
            
            # Capture old position (handle different object types).
            if hasattr(obj, 'pos'):
                # Real PlacedObject
                old_pos = obj.pos.copy()
                obj.pos.update(selected_pos)
            else:
                # Mock or other type
                old_pos = {"x": obj.x, "z": obj.z}
                obj.x = selected_pos["x"]
                obj.z = selected_pos["z"]
            
            fixed_count += 1
    
    return fixed_count > 0
