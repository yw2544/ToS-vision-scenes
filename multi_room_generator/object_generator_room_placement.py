"""
Room-level placement helpers for ObjectGenerator.
"""

import math
from typing import List, Dict, Optional

import numpy as np

from .room_analyzer import RoomInfo
from .object_generator_types import PlacedObject, AgentInfo


def _generate_room_objects(self, room: RoomInfo, num_objects: int, is_main_room: bool) -> List[PlacedObject]:
    """Generate objects for a specific room"""
    if num_objects == 0:
        return []

    room_objects = []
    # Ensure checks can see objects already placed in this room
    self.rooms_objects[room.room_id] = room_objects

    # If this is the main room, place agent first
    if is_main_room:
        agent_success = self._place_agent_in_room(room)
        if not agent_success:
            print(f"[WARN] Failed to place agent in main room {room.room_id}")

    # Select object types for this room (ensuring no global duplicates)
    current_pool = self.get_current_object_pool()
    available_objects = sorted([
        obj_key for obj_key in current_pool.keys()
        if current_pool[obj_key]['category'] not in self.used_categories
    ])

    if len(available_objects) < num_objects:
        print(f"[WARN] Only {len(available_objects)} unique objects available, reducing from {num_objects}")
        num_objects = len(available_objects)

    if num_objects == 0:
        return []

    # Use the new category-first selection
    preferred_categories = []
    # Remove chair priority to keep selection fully random

    # Oriented object requirements by room type
    # Main room needs 3; other rooms need at least 2 (for bwd_pov tasks)
    min_oriented = 2 if is_main_room else 2
    selected_objects = self.select_objects_by_categories(num_objects, preferred_categories, min_oriented_objects=min_oriented)

    if len(selected_objects) < num_objects:
        print(f"[WARN] Room {room.room_id}: only selected {len(selected_objects)} objects instead of {num_objects}")

    print(f"[INFO] Room {room.room_id} selected objects: {[obj['name'] for obj in selected_objects]}")

    # Place each selected object using new method
    for obj_config in selected_objects:
        placed_obj = self._place_object_in_room_new(room, obj_config)
        if placed_obj:
            room_objects.append(placed_obj)
            self.used_categories.add(obj_config['category'])  # Reserve only after success

    # Optimize positions if we have objects
    if room_objects:
        if is_main_room and self.agent:
            # Include agent position in optimization for main room
            optimized = self._optimize_room_positions(room, room_objects, self.agent.pos)
        else:
            optimized = self._optimize_room_positions(room, room_objects)
        if optimized:
            print(f"[INFO] Room {room.room_id}: position optimization successful")
        else:
            print(f"[WARN] Room {room.room_id}: position optimization failed, using initial positions")

    # Debug: check minimum distances inside the room
    self._debug_min_center_gap(room.room_id, self.min_distance)

    # Incremental collinearity checks happen during placement
    # Final validation (confirm only, no fixes)
    if len(room_objects) >= 3:
        agent_pos = self.agent.pos if is_main_room and self.agent else None
        is_valid, collinear_groups = self.collinear_validator.validate_room_positions(
            room_objects, agent_pos, room.room_id, self.room_analyzer
        )

        if is_valid:
            print(f"[INFO] Room {room.room_id} collinearity check passed")
        else:
            print(f"[WARN] Room {room.room_id} still has {len(collinear_groups)} collinear groups (incremental checks not fully preventing)")

    return room_objects


def _generate_room_objects_exact_count(self, room: RoomInfo, target_count: int, is_main_room: bool) -> List[PlacedObject]:
    """Generate exact number of objects for a specific room with retry mechanism"""
    if target_count == 0:
        return []

    max_attempts = 10
    for attempt in range(max_attempts):
        print(f"[INFO] Room {room.room_id}: attempt {attempt + 1}/{max_attempts} to place {target_count} objects")

        # Reset used objects for this attempt (only for this room)
        room_used_objects = set()
        room_objects = []
        # Ensure checks can see objects already placed in this room
        self.rooms_objects[room.room_id] = room_objects

        # If this is the main room and first attempt, place agent first
        if is_main_room and attempt == 0:
            agent_success = self._place_agent_in_room(room)
            if not agent_success:
                print(f"[WARN] Failed to place agent in main room {room.room_id}")

        # Select object types for this room (ensuring no global duplicates)
        current_pool = self.get_current_object_pool()
        available_objects = sorted([
            obj_key for obj_key in current_pool.keys()
            if current_pool[obj_key]['category'] not in self.used_categories and obj_key not in room_used_objects
        ])

        if len(available_objects) < target_count:
            print(f"[WARN] Only {len(available_objects)} unique objects available for room {room.room_id}")
            actual_target = len(available_objects)
        else:
            actual_target = target_count

        if actual_target == 0:
            return []

        # Use the new category-first selection
        preferred_categories = []
        # Remove chair priority to keep selection fully random

        # Temporarily reset used_categories to allow re-selection for this room
        temp_used = self.used_categories.copy()
        self.used_categories = set()  # Reset to allow retries

        # Oriented object requirements by room type
        # Main room needs 3; other rooms need at least 2
        min_oriented = 3 if is_main_room else 2
        selected_objects = self.select_objects_by_categories(actual_target, preferred_categories, min_oriented_objects=min_oriented)

        # Restore used_categories, excluding this attempt
        self.used_categories = temp_used

        print(f"[INFO] Room {room.room_id} attempt {attempt + 1} selected objects: {[obj['name'] for obj in selected_objects]}")

        # Place each selected object using new method
        placement_success = True
        for obj_config in selected_objects:
            placed_obj = self._place_object_in_room_new(room, obj_config)
            if placed_obj:
                room_objects.append(placed_obj)
            else:
                print(f"[WARN] Failed to place {obj_config['name']} in room {room.room_id}")
                placement_success = False
                break

        # Check if we achieved the target count
        if len(room_objects) == target_count:
            print(f"[SUCCESS] Room {room.room_id}: placed all {target_count} objects successfully")
            # Mark these objects as used globally (by category)
            for obj_config in selected_objects:
                self.used_categories.add(obj_config['category'])

            # Optimize positions if we have objects
            if room_objects:
                if is_main_room and self.agent:
                    optimized = self._optimize_room_positions(room, room_objects, self.agent.pos)
                else:
                    optimized = self._optimize_room_positions(room, room_objects)
                if optimized:
                    print(f"[INFO] Room {room.room_id}: position optimization successful")
                else:
                    print(f"[WARN] Room {room.room_id}: position optimization failed, using initial positions")

            # Debug: check minimum distances inside the room
            self._debug_min_center_gap(room.room_id, self.min_distance)

            # Incremental collinearity checks happen during placement
            # Final validation (confirm only, no fixes)
            if len(room_objects) >= 3:
                agent_pos = self.agent.pos if is_main_room and self.agent else None
                is_valid, collinear_groups = self.collinear_validator.validate_room_positions(
                    room_objects, agent_pos, room.room_id, self.room_analyzer
                )

                if is_valid:
                    print(f"[INFO] Room {room.room_id} collinearity check passed")
                else:
                    print(f"[WARN] Room {room.room_id} still has {len(collinear_groups)} collinear groups (incremental checks not fully preventing)")

            return room_objects
        else:
            print(f"[WARN] Room {room.room_id} attempt {attempt + 1}: only placed {len(room_objects)}/{target_count} objects")
            # Clear for next attempt, but don't remove from used_categories yet
            room_objects = []

    # If we reach here, we couldn't place all objects
    print(f"[ERROR] Room {room.room_id}: failed to place {target_count} objects after {max_attempts} attempts")
    return room_objects  # Return whatever we managed to place in the last attempt


def _place_agent_in_room(self, room: RoomInfo) -> bool:
    """Place agent in the specified room at integer coordinates, far away from all doors"""
    print(f"[INFO] Attempting to place agent in room {room.room_id}")
    from .room_analyzer import is_door

    # First, identify ALL door positions in the entire mask
    door_positions = []
    for r in range(self.room_analyzer.rows):
        for c in range(self.room_analyzer.cols):
            if is_door(self.room_analyzer.mask[r][c]):
                door_positions.append((r, c))

    # Also get door center coordinates (world coordinates)
    door_center_positions = []
    if hasattr(self.room_analyzer, 'doors'):
        for door_id, door_info in self.room_analyzer.doors.items():
            door_center_x, door_center_z = door_info.center
            # Convert to integer coordinates for agent placement
            int_door_x = int(round(door_center_x))
            int_door_z = int(round(door_center_z))
            door_center_positions.append((int_door_x, int_door_z))

    print(f"[INFO] Found {len(door_positions)} door cells in mask: {door_positions}")
    print(f"[INFO] Found {len(door_center_positions)} door centers at: {door_center_positions}")

    # Get safe interior room cells with VERY strict validation (far from doors)
    valid_positions = []
    min_door_distance = 1  # Require at least 1 cell away from ANY door (Chebyshev distance)

    for row, col in room.cells:
        # First check: this cell must be exactly the target room
        cell_value = self.room_analyzer.mask[row][col]
        if cell_value != room.room_id:
            print(f"[DEBUG] Cell ({row},{col}) has value {cell_value}, not room {room.room_id}")
            continue

        # Second check: ensure this cell is NOT a door itself
        if is_door(cell_value):
            print(f"[DEBUG] Cell ({row},{col}) is a door (value {cell_value})")
            continue

        # Third check: ensure this cell is far from ALL doors
        min_distance_to_door = float('inf')
        for door_row, door_col in door_positions:
            distance = max(abs(row - door_row), abs(col - door_col))  # Chebyshev distance
            min_distance_to_door = min(min_distance_to_door, distance)

        if min_distance_to_door < min_door_distance:
            print(f"[DEBUG] Cell ({row},{col}) is too close to doors (distance {min_distance_to_door} < {min_door_distance})")
            continue

        # Fourth check: simplified - just check that the position itself is valid and far from doors
        # Don't require all 8 neighbors to be in the same room (too restrictive)
        is_safe_interior = True

        if is_safe_interior:
            # Convert to world coordinates and force to integers
            world_x, world_z = self.room_analyzer._cell_to_world(row, col)
            int_x, int_z = int(round(world_x)), int(round(world_z))

            # Verify the integer coordinates are still valid
            verify_row, verify_col = self.room_analyzer._world_to_cell(float(int_x), float(int_z))
            if (0 <= verify_row < self.room_analyzer.rows and
                0 <= verify_col < self.room_analyzer.cols and
                self.room_analyzer.mask[verify_row][verify_col] == room.room_id and
                not is_door(self.room_analyzer.mask[verify_row][verify_col])):

                # Check if this position conflicts with any door center coordinates
                conflicts_with_door_center = False
                for door_x, door_z in door_center_positions:
                    if int_x == door_x and int_z == door_z:
                        conflicts_with_door_center = True
                        print(f"[DEBUG] Position ({int_x}, {int_z}) conflicts with door center at ({door_x}, {door_z})")
                        break

                if conflicts_with_door_center:
                    continue

                # Use 8-neighborhood check to keep agent clear of doors/objects
                if self._check_8_neighbor_conflicts(int_x, int_z, room.room_id, check_doors=True, check_agent=False, check_objects=True):
                    print(f"[DEBUG] Agent candidate ({int_x},{int_z}) has 8-neighborhood conflicts")
                    continue

                # Occupy grid position
                rr, cc = self.room_analyzer._world_to_cell(float(int_x), float(int_z))
                if (int(rr), int(cc)) in self.occupied_rc:
                    continue
                valid_positions.append((float(int_x), float(int_z)))
                print(f"[DEBUG] SAFE position found: mask({row},{col}) -> world({int_x},{int_z}) -> verify({verify_row},{verify_col}) distance_to_doors={min_distance_to_door}")

    # If no positions with distance 3, try distance 2
    if not valid_positions and min_door_distance > 2:
        print(f"[WARN] No positions with distance {min_door_distance} from doors, trying distance 2")
        min_door_distance = 2
        # Repeat the same logic with smaller distance requirement
        for row, col in room.cells:
            cell_value = self.room_analyzer.mask[row][col]
            if cell_value != room.room_id or is_door(cell_value):
                continue

            # Check distance to doors
            min_distance_to_door = float('inf')
            for door_row, door_col in door_positions:
                distance = max(abs(row - door_row), abs(col - door_col))
                min_distance_to_door = min(min_distance_to_door, distance)

            if min_distance_to_door < min_door_distance:
                continue

            # Simplified check for distance 2 fallback
            is_safe_interior = True

            if is_safe_interior:
                world_x, world_z = self.room_analyzer._cell_to_world(row, col)
                int_x, int_z = int(round(world_x)), int(round(world_z))
                verify_row, verify_col = self.room_analyzer._world_to_cell(float(int_x), float(int_z))
                if (0 <= verify_row < self.room_analyzer.rows and
                    0 <= verify_col < self.room_analyzer.cols and
                    self.room_analyzer.mask[verify_row][verify_col] == room.room_id and
                    not is_door(self.room_analyzer.mask[verify_row][verify_col])):

                    # Check if this position conflicts with any door center coordinates
                    conflicts_with_door_center = False
                    for door_x, door_z in door_center_positions:
                        if int_x == door_x and int_z == door_z:
                            conflicts_with_door_center = True
                            print(f"[DEBUG] Position ({int_x}, {int_z}) conflicts with door center at ({door_x}, {door_z}) [distance 2 check]")
                            break

                    if conflicts_with_door_center:
                        continue

                    # Use 8-neighborhood check for agent safety (distance 2 fallback)
                    if self._check_8_neighbor_conflicts(int_x, int_z, room.room_id, check_doors=True, check_agent=False, check_objects=True):
                        print(f"[DEBUG] Agent candidate ({int_x},{int_z}) has 8-neighborhood conflicts [distance 2 check]")
                        continue

                    rr, cc = self.room_analyzer._world_to_cell(float(int_x), float(int_z))
                    if (int(rr), int(cc)) in self.occupied_rc:
                        continue
                    valid_positions.append((float(int_x), float(int_z)))
                    print(f"[DEBUG] SAFE position found (distance 2): mask({row},{col}) -> world({int_x},{int_z})")

    if not valid_positions:
        print(f"[ERROR] No valid positions found for agent in room {room.room_id} that are far enough from doors")
        print(f"[ERROR] All door positions: {door_positions}")
        print(f"[ERROR] Room {room.room_id} cells: {room.cells[:10]}...")  # Show first 10 cells
        return False

    print(f"[INFO] Found {len(valid_positions)} valid positions for agent in room {room.room_id} far from doors")

    # Sort for deterministic order but use seed-based offset for agent
    valid_positions.sort(key=lambda pos: (pos[0], pos[1]))

    # Use seed to determine agent's starting position offset
    agent_start_offset = self.rng.randint(0, len(valid_positions))

    # Try to place agent with thorough validation
    for attempt in range(len(valid_positions)):
        idx = (agent_start_offset + attempt) % len(valid_positions)
        x, z = valid_positions[idx]

        # Final verification
        verify_row, verify_col = self.room_analyzer._world_to_cell(x, z)
        if (0 <= verify_row < self.room_analyzer.rows and
            0 <= verify_col < self.room_analyzer.cols):

            actual_cell_value = self.room_analyzer.mask[verify_row][verify_col]
            if actual_cell_value == room.room_id and not is_door(actual_cell_value):
                # Calculate yaw to face room center
                room_center = room.center
                yaw = math.degrees(math.atan2(room_center[0] - x, room_center[1] - z))

                self.agent = AgentInfo(
                    pos={"x": x, "y": 0.85, "z": z},
                    room_id=room.room_id,
                    rotation={"x": 0, "y": yaw, "z": 0}
                )
                self.occupied_rc.add((int(verify_row), int(verify_col)))

                print(f"[INFO] Agent successfully placed in room {room.room_id} at ({x:.0f}, {z:.0f})")
                print(f"[DEBUG] Final verification: world({x:.0f},{z:.0f}) -> mask({verify_row},{verify_col}) = {actual_cell_value}")

                # Verify distance to all doors
                min_dist = float('inf')
                for door_row, door_col in door_positions:
                    dist = max(abs(verify_row - door_row), abs(verify_col - door_col))
                    min_dist = min(min_dist, dist)
                print(f"[DEBUG] Agent distance to nearest door: {min_dist} cells")

                return True
            else:
                print(f"[DEBUG] Position ({x:.0f}, {z:.0f}) final check failed: cell value {actual_cell_value}")

    print(f"[ERROR] Failed to place agent in room {room.room_id} after 100 attempts")
    return False


def _place_object_in_room(self, room: RoomInfo, obj_config: Dict, obj_key: str) -> Optional[PlacedObject]:
    """Place a single object in a room"""
    if not self.model_lib:
        print("[ERROR] Model library not set for object generation")
        return None

    is_custom_model = obj_config.get("is_custom_model", False)

    # Get object bounds
    try:
        w, d = self._get_object_bounds(obj_config["model"])
        w *= obj_config["scale"]
        d *= obj_config["scale"]
    except Exception as e:
        print(f"[WARN] Failed to get bounds for {obj_config['model']}: {e}")
        w, d = 1.0, 1.0  # Default size

    # Get only interior room cells (not adjacent to walls/doors)
    room = self.room_analyzer.rooms[room.room_id]
    valid_positions = []

    for row, col in room.cells:
        # Check if this cell is truly interior (all neighbors are the same room)
        is_interior = True
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if (0 <= nr < self.room_analyzer.rows and 0 <= nc < self.room_analyzer.cols):
                    if self.room_analyzer.mask[nr][nc] != room.room_id:
                        is_interior = False
                        break
                else:
                    is_interior = False
                    break
            if not is_interior:
                break

        if is_interior:
            world_x, world_z = self.room_analyzer._cell_to_world(row, col)
            # Round to integers
            int_x, int_z = int(round(world_x)), int(round(world_z))
            rr, cc = self.room_analyzer._world_to_cell(float(int_x), float(int_z))
            if (int(rr), int(cc)) in self.occupied_rc:
                continue
            valid_positions.append((float(int_x), float(int_z)))

    if not valid_positions:
        print(f"[WARN] No interior positions found for room {room.room_id}, using all room cells as fallback")
        # Fallback to all room cells if no interior positions found
        for row, col in room.cells:
            world_x, world_z = self.room_analyzer._cell_to_world(row, col)
            int_x, int_z = int(round(world_x)), int(round(world_z))
            rr, cc = self.room_analyzer._world_to_cell(float(int_x), float(int_z))
            if (int(rr), int(cc)) in self.occupied_rc:
                continue
            valid_positions.append((float(int_x), float(int_z)))

        if not valid_positions:
            print(f"[ERROR] No valid positions found for room {room.room_id}")
            return None

    # Sort for deterministic order but use object-specific offset
    valid_positions.sort(key=lambda pos: (pos[0], pos[1]))

    # Use object_id as seed for deterministic but varied starting position
    obj_specific_rng = np.random.RandomState(self.rng.randint(0, 1000000))
    start_offset = obj_specific_rng.randint(0, len(valid_positions))

    # Try to place object (reduce attempts to avoid hang)
    for attempt in range(min(50, len(valid_positions))):
        # Use deterministic selection with object-specific offset
        idx = (start_offset + attempt) % len(valid_positions)
        x, z = valid_positions[idx]

        # Check if position is valid (not on walls/doors and not overlapping)
        position_valid = self.room_analyzer.is_position_valid_for_placement(x, z, room.room_id)
        no_overlap = not self._overlaps_with_existing_with_default(x, z, w, d, room.room_id, obj_config)
        # Use final coordinates for collinearity check
        cand_fx, cand_fz = self._final_center_from_candidate(x, z, obj_config)
        safe_collinear = self._is_position_safe_from_collinearity(cand_fx, cand_fz, room.room_id)

        print(f"[PLACEMENT] Position ({x:.0f}, {z:.0f}): room_valid={position_valid}, no_overlap={no_overlap}, no_collinear={safe_collinear}")

        if position_valid and no_overlap and safe_collinear:

            # Create object with rotation
            object_id = self.rng.randint(1000000, 9999999)

            # Calculate base rotation using placement algorithm
            rotation_options = [0, 90, 180, 270]
            rotation_index = (int(x) + int(z) + object_id) % 4
            base_rotation_y = rotation_options[rotation_index]

            # Store only base rotation for all models (default_rotation will be applied in get_final_rotation)
            is_custom_model = obj_config.get("is_custom_model", False)
            rotation_y = base_rotation_y

            # Determine orientation
            has_orientation = obj_config.get("has_orientation", False)
            orientation = None
            if has_orientation:
                orientation = self._rotation_to_orientation(rotation_y)

            # Set base position and rotation (default_position handled in get_final_position)
            base_pos = {"x": x, "y": 0.05, "z": z}  # Floor height
            base_rot = {"x": 0, "y": rotation_y, "z": 0}

            # Note: do not apply default_position offset here; it is handled in PlacedObject.get_final_position
            # This keeps default_position tied to the object as it moves

            placed_obj = PlacedObject(
                object_id=object_id,
                model=obj_config["model"],
                name=obj_config["name"],
                pos=base_pos,  # Store base position without default offset
                rot=base_rot,  # Store base rotation without default x/z offset
                size=(w, d),
                scale=obj_config["scale"],
                color=obj_config["color"],
                room_id=room.room_id,
                has_orientation=has_orientation,
                orientation=orientation,
                is_custom_model=is_custom_model,
                custom_config=obj_config if is_custom_model else None,
                model_config=obj_config
            )
            vr, vc = self.room_analyzer._world_to_cell(x, z)
            self.occupied_rc.add((int(vr), int(vc)))

            print(f"[INFO] Placed {obj_config['name']} in room {room.room_id} at ({x:.2f}, {z:.2f})")
            return placed_obj

    print(f"[WARN] Failed to place {obj_config['name']} in room {room.room_id} after 100 attempts")
    return None
