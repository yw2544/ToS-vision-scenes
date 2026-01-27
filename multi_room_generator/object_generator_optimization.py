"""
Object placement optimization and collinearity checks.
"""

from typing import List, Dict, Optional

from .room_analyzer import RoomInfo
from .object_generator_types import PlacedObject
from .collinear_validator import are_three_points_collinear


def _is_position_safe_from_collinearity(self, x: float, z: float, room_id: int) -> bool:
    """
    Incremental collinearity safety check for a candidate placement.

    Args:
        x, z: Candidate position
        room_id: Room ID

    Returns:
        True if position is safe (no collinearity)
        False if position would cause collinearity
    """
    # Get objects already placed in the room
    room_objects = self.rooms_objects.get(room_id, [])

    # If fewer than 2 objects, no collinearity risk
    if len(room_objects) < 2:
        return True

    # Build current positions list (including doors and agent)
    positions = []
    names = []

    # Add placed objects
    for obj in room_objects:
        final_pos = obj.get_final_position()
        positions.append((final_pos["x"], final_pos["z"]))
        names.append(obj.name)

    # Add door positions
    if hasattr(self.room_analyzer, 'doors'):
        for door_id, door_info in self.room_analyzer.doors.items():
            if room_id in door_info.connected_rooms:
                door_x, door_z = door_info.center
                positions.append((door_x, door_z))
                names.append(f"door_{door_id}")

    # Add agent position (if in the same room)
    if self.agent and self.agent.room_id == room_id:
        positions.append((self.agent.pos["x"], self.agent.pos["z"]))
        names.append("agent")

    # Add candidate position last (x,z already in final coordinates)
    cand_idx = len(positions)  # Index of candidate point
    positions.append((x, z))
    names.append("new_object")

    # Check collinearity
    tolerance = self.collinear_validator.tolerance  # 0.5 (from tolerance_width=1.0)

    # Check all triplets that include the candidate (including doors/agent)
    for i in range(cand_idx):
        for j in range(i + 1, cand_idx):
            if are_three_points_collinear(positions[i], positions[j], positions[cand_idx], tolerance):
                print(f"[DEBUG] Position ({x:.1f}, {z:.1f}) would be collinear: {names[i]} - {names[j]} - {names[cand_idx]}")
                return False

    return True


def _optimize_room_positions(self, room: RoomInfo, room_objects: List[PlacedObject],
                            agent_pos: Optional[Dict[str, float]] = None) -> bool:
    """
    Optimize object positions within a room using simplified visibility constraints

    This is a simplified version of the optimization from pipeline_partial_label_only.py
    adapted for multi-room scenarios where we don't need full visibility optimization.
    """
    if not room_objects:
        return True

    print(f"[INFO] Optimizing positions for {len(room_objects)} objects in room {room.room_id}")

    # Simple optimization: just ensure no overlaps and reasonable spacing
    max_iterations = 5
    improved = True

    for iteration in range(max_iterations):
        if not improved:
            break

        improved = False

        for obj in room_objects:
            original_pos = obj.pos.copy()

            # Try small position adjustments (integer offsets)
            for attempt in range(10):
                offset_x = float(self.rng.randint(-2, 3))  # -2, -1, 0, 1, 2
                offset_z = float(self.rng.randint(-2, 3))  # -2, -1, 0, 1, 2
                new_x = original_pos["x"] + offset_x
                new_z = original_pos["z"] + offset_z

                # Check if new position is valid (not on walls/doors, not overlapping, and safe from collinearity)
                # Optimization phase must also respect 8-neighborhood constraints
                position_valid = self.room_analyzer.is_position_valid_for_placement(new_x, new_z, room.room_id)
                no_overlap = not self._would_overlap_at_position(obj, new_x, new_z, room.room_id, agent_pos, room_objects)
                # Use final coordinates for collinearity during optimization
                safe_collinear = self._is_position_safe_from_collinearity_during_optimization(new_x, new_z, room.room_id, obj)

                # Enforce 8-neighborhood check during optimization
                no_8neighbor_conflict = not self._check_8_neighbor_conflicts(int(round(new_x)), int(round(new_z)), room.room_id)

                print(f"[OPTIMIZE] Position ({new_x:.0f}, {new_z:.0f}): room_valid={position_valid}, no_overlap={no_overlap}, no_collinear={safe_collinear}, no_8neighbor_conflict={no_8neighbor_conflict}")

                if position_valid and no_overlap and safe_collinear and no_8neighbor_conflict:

                    old_r, old_c = self.room_analyzer._world_to_cell(original_pos["x"], original_pos["z"])
                    new_r, new_c = self.room_analyzer._world_to_cell(new_x, new_z)

                    # Clear old occupancy and set new position
                    self._unmark_object_occupied(obj)
                    obj.pos["x"] = new_x
                    obj.pos["z"] = new_z
                    self._mark_object_occupied(obj)
                    improved = True
                    break

    print(f"[INFO] Room {room.room_id} optimization completed")
    return True


def _is_position_safe_from_collinearity_during_optimization(self, x: float, z: float, room_id: int, moving_obj: PlacedObject) -> bool:
    """
    Collinearity safety check during optimization for moving an object.

    Args:
        x, z: Candidate position
        room_id: Room ID
        moving_obj: Object being moved

    Returns:
        True if position is safe (no collinearity)
        False if position would cause collinearity
    """
    # Get other objects in the room (exclude moving object)
    room_objects = self.rooms_objects.get(room_id, [])
    other_objects = [obj for obj in room_objects if obj.object_id != moving_obj.object_id]

    # If fewer than 2 other objects, no collinearity risk
    if len(other_objects) < 2:
        return True

    # Build position list
    positions = []
    names = []

    # Add other objects
    for obj in other_objects:
        final_pos = obj.get_final_position()
        positions.append((final_pos["x"], final_pos["z"]))
        names.append(obj.name)

    # Add door positions
    if hasattr(self.room_analyzer, 'doors'):
        for door_id, door_info in self.room_analyzer.doors.items():
            if room_id in door_info.connected_rooms:
                door_x, door_z = door_info.center
                positions.append((door_x, door_z))
                names.append(f"door_{door_id}")

    # Add agent position (if in the same room)
    if self.agent and self.agent.room_id == room_id:
        positions.append((self.agent.pos["x"], self.agent.pos["z"]))
        names.append("agent")

    # Add moving object's new position last (convert to final coords)
    cand_idx = len(positions)  # Index of moving object
    cand_fx, cand_fz = self._final_center_from_candidate(x, z, moving_obj)
    positions.append((cand_fx, cand_fz))
    names.append(moving_obj.name)

    # Check collinearity
    tolerance = self.collinear_validator.tolerance

    # Check all triplets including the moving object (including doors/agent)
    for i in range(cand_idx):
        for j in range(i + 1, cand_idx):
            if are_three_points_collinear(positions[i], positions[j], positions[cand_idx], tolerance):
                return False

    return True


def _would_overlap_at_position(self, test_obj: PlacedObject, x: float, z: float,
                              room_id: int, agent_pos: Optional[Dict[str, float]] = None,
                              room_objects_override: Optional[List[PlacedObject]] = None) -> bool:
    """Check if object would overlap at given position"""
    w, d = test_obj.size

    # Check overlap with agent
    if self.agent and self.agent.room_id == room_id:
        if self._check_center_distance(x, z, self.agent.pos["x"], self.agent.pos["z"], self.min_distance):
            return True

    # Check overlap with other objects in the room
    room_objects = room_objects_override if room_objects_override is not None else self.rooms_objects.get(room_id, [])
    for obj in room_objects:
        if obj.object_id == test_obj.object_id:
            continue  # Skip self

        obj_final_pos = obj.get_final_position()

        # Compute test object's final position at the new location
        test_base_pos = {"x": x, "y": test_obj.pos["y"], "z": z}
        test_final_pos = test_obj.get_final_position(test_base_pos)

        # Use final position to compute distance
        if self._check_center_distance(test_final_pos["x"], test_final_pos["z"], obj_final_pos["x"], obj_final_pos["z"], self.min_distance):
            return True

    return False
