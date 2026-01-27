"""
Strict placement helpers for ObjectGenerator.
"""

from typing import List, Dict, Optional

import numpy as np

from .room_analyzer import RoomInfo
from .object_generator_types import PlacedObject


def _try_place_single_object_strict(self, room: RoomInfo, is_main_room: bool, prefer_oriented: bool = False, force_oriented: bool = False) -> Optional[PlacedObject]:
    """Try to place a single object with strict collision detection and integer positions

    Args:
        room: Room to place object in
        is_main_room: Whether this is the main room
        prefer_oriented: If True, prioritize oriented objects
        force_oriented: If True, ONLY try oriented objects (for Phase 1)
    """
    # Select available object type - STRICT: Never allow category duplicates
    current_pool = self.get_current_object_pool()
    available_objects = sorted([obj for obj in current_pool.keys() if current_pool[obj]['category'] not in self.used_categories])

    if not available_objects:
        print(f"[INFO] No unique categories available for room {room.room_id}, skipping object placement")
        return None

    # If we force oriented, ONLY use oriented objects
    if force_oriented:
        oriented_objects = sorted([obj_key for obj_key in available_objects if current_pool[obj_key].get('has_orientation', False)])

        if not oriented_objects:
            print(f"[WARN] Room {room.room_id} REQUIRES oriented object but none available (force_oriented=True)")
            return None

        print(f"[INFO] Room {room.room_id} FORCING oriented objects, {len(oriented_objects)} available")
        trial_order = oriented_objects

    # If we prefer oriented objects, separate and prioritize them
    elif prefer_oriented:
        oriented_objects = sorted([obj_key for obj_key in available_objects if current_pool[obj_key].get('has_orientation', False)])
        non_oriented_objects = sorted([obj_key for obj_key in available_objects if not current_pool[obj_key].get('has_orientation', False)])

        if oriented_objects:
            print(f"[INFO] Room {room.room_id} prefers oriented objects, prioritizing {len(oriented_objects)} oriented choices")
            # Try oriented objects first
            trial_order = oriented_objects + non_oriented_objects
        else:
            print(f"[WARN] Room {room.room_id} prefers oriented objects but none available in pool")
            trial_order = available_objects
    else:
        trial_order = available_objects.copy()

    # Try different object types
    max_trials = min(len(trial_order), 10 if force_oriented else 5)  # More trials when forcing
    for _ in range(max_trials):
        if not trial_order:
            break

        obj_key = self.rng.choice(trial_order)
        obj_config = current_pool[obj_key]

        result = self._try_place_object_at_integer_positions(room, obj_config, obj_key)
        if result:
            self.used_categories.add(obj_config['category'])
            return result

        # Remove failed type from this attempt
        if obj_key in trial_order:
            trial_order.remove(obj_key)

    # Strict mode failed: do not fall back to relaxed mode to keep category uniqueness
    mode_str = "FORCED oriented" if force_oriented else "strict mode"
    print(f"[INFO] {mode_str} failed; skip relaxed mode to preserve category uniqueness")
    return None


def _try_place_object_at_integer_positions(self, room: RoomInfo, obj_config: dict, obj_key: str) -> Optional[PlacedObject]:
    """Try to place object at integer world coordinates with strict collision detection"""
    scale = obj_config["scale"]
    is_custom_model = obj_config.get("is_custom_model", False)

    # Get model dimensions
    if is_custom_model:
        # For custom models, use estimated or default size
        w, d = 1.0, 1.0  # Default size, adjust if needed
        # If custom config exists, it may provide more accurate size
    else:
        model_lib = self.model_lib
        record = model_lib.get_record(obj_config["model"])
        if not record:
            return None

        bounds = record.bounds
        w = abs(bounds["right"]["x"] - bounds["left"]["x"]) * scale
        d = abs(bounds["front"]["z"] - bounds["back"]["z"]) * scale

    # Get all integer positions within the room
    integer_positions = []
    for row, col in room.cells:
        world_x, world_z = self.room_analyzer._cell_to_world(row, col)
        # Round to integer coordinates
        int_x = int(round(world_x))
        int_z = int(round(world_z))

        # Verify the integer position is still within the room
        if self.room_analyzer.is_position_valid_for_placement(int_x, int_z, room.room_id):
            integer_positions.append((int_x, int_z))

    # Remove duplicates
    integer_positions = list(set(integer_positions))

    # Pick the (x%2, z%2) parity bucket with the most samples
    from collections import Counter, defaultdict
    bucket = defaultdict(list)
    for ix, iz in integer_positions:
        bucket[(ix & 1, iz & 1)].append((ix, iz))

    # Choose the largest parity bucket (usually yields ~9 points)
    # Sort bucket keys for deterministic tie-breaking
    best_key = max(sorted(bucket.keys()), key=lambda k: len(bucket[k]))
    primary_positions = bucket[best_key]

    # Prepare fallback: all bucket positions, sorted for determinism
    all_positions = [p for k in sorted(bucket.keys()) for p in bucket[k]]

    # Filter out occupied grid positions for primary positions
    if self.use_grid_occupancy:
        filtered_primary = []
        for ix, iz in primary_positions:
            rr, cc = self.room_analyzer._world_to_cell(float(ix), float(iz))
            if (int(rr), int(cc)) not in self.occupied_rc:
                filtered_primary.append((ix, iz))
        primary_positions = filtered_primary

        # Also filter all positions for fallback
        filtered_all = []
        for ix, iz in all_positions:
            rr, cc = self.room_analyzer._world_to_cell(float(ix), float(iz))
            if (int(rr), int(cc)) not in self.occupied_rc:
                filtered_all.append((ix, iz))
        all_positions = filtered_all

    # Try the best bucket positions first
    if primary_positions:
        primary_positions.sort(key=lambda pos: (pos[0], pos[1]))
        self.rng.shuffle(primary_positions)

        for x, z in primary_positions:
            x, z = float(x), float(z)
            # Enforce 8-neighborhood check
            has_8neighbor_conflict = self._overlaps_with_existing_with_default(x, z, w, d, room.room_id, obj_config)
            print(f"[STRICT] Best-bucket position ({x:.0f}, {z:.0f}): 8-neighborhood conflict={has_8neighbor_conflict}")

            if not has_8neighbor_conflict:
                # Check collinearity (use final coordinates)
                cand_fx, cand_fz = self._final_center_from_candidate(x, z, obj_config)
                if not self._is_position_safe_from_collinearity(cand_fx, cand_fz, room.room_id):
                    continue

                # Create object at integer position with rotation
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

                # Set base position/rotation (default_position handled in get_final_position)
                base_pos = {"x": x, "y": 0.05, "z": z}
                base_rot = {"x": 0, "y": rotation_y, "z": 0}

                placed_obj = PlacedObject(
                    object_id=object_id,
                    model=obj_config["model"],
                    name=obj_config["name"],
                    pos=base_pos,
                    rot=base_rot,
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
                self._mark_object_occupied(placed_obj)

                return placed_obj

    # If best bucket fails, try all positions as fallback
    print(f"[INFO] Best bucket failed; trying all available positions (fallback)")
    if all_positions:
        all_positions.sort(key=lambda pos: (pos[0], pos[1]))
        self.rng.shuffle(all_positions)

        for x, z in all_positions:
            x, z = float(x), float(z)  # Convert to float for calculations

            # Enforce 8-neighborhood check
            has_8neighbor_conflict = self._overlaps_with_existing_with_default(x, z, w, d, room.room_id, obj_config)
            print(f"[STRICT] Strict position ({x:.0f}, {z:.0f}): 8-neighborhood conflict={has_8neighbor_conflict}")

            if not has_8neighbor_conflict:
                # Check collinearity (use final coordinates)
                cand_fx, cand_fz = self._final_center_from_candidate(x, z, obj_config)
                if not self._is_position_safe_from_collinearity(cand_fx, cand_fz, room.room_id):
                    continue

                # Create object at integer position with rotation
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

                # Set base position/rotation (default_position handled in get_final_position)
                base_pos = {"x": x, "y": 0.05, "z": z}
                base_rot = {"x": 0, "y": rotation_y, "z": 0}

                placed_obj = PlacedObject(
                    object_id=object_id,
                    model=obj_config["model"],
                    name=obj_config["name"],
                    pos=base_pos,
                    rot=base_rot,
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
                self._mark_object_occupied(placed_obj)

                return placed_obj

    return None


def _generate_room_objects_force(self, room: RoomInfo, num_objects: int, is_main_room: bool, skip_agent: bool = False) -> List[PlacedObject]:
    """Force placement of objects with more relaxed constraints"""
    if num_objects == 0:
        return []

    room_objects = []

    # If this is the main room and agent isn't placed yet, place it first
    if is_main_room and not skip_agent and self.agent is None:
        agent_success = self._place_agent_in_room(room)
        if not agent_success:
            print(f"[WARN] Failed to place agent in main room {room.room_id}")

    # Enforce strict category uniqueness - never allow duplicates
    current_pool = self.get_current_object_pool()
    available_objects = sorted([obj for obj in current_pool.keys() if current_pool[obj]['category'] not in self.used_categories])

    if not available_objects:
        print(f"[INFO] No unique categories available for force placement in room {room.room_id}")
        return []

    for i in range(num_objects):
        if not available_objects:
            # If we run out of unique objects, stop force placement to maintain category uniqueness
            print(f"[INFO] Stopping force placement at object {i+1}/{num_objects} to maintain category uniqueness")
            break

        # Try multiple objects until one succeeds
        placed = False
        attempts_per_object = min(len(available_objects), 10)  # Limit attempts

        for attempt in range(attempts_per_object):
            if not available_objects:
                break

            obj_key = self.rng.choice(available_objects)
            obj_config = self.get_current_object_pool()[obj_key]

            # Try to place this object with more relaxed constraints
            success_obj = self._try_place_object_relaxed(room, obj_config, obj_key)
            if success_obj:
                room_objects.append(success_obj)
                self.used_categories.add(obj_config['category'])

                # Remove from available if we want to avoid duplicates
                if obj_key in available_objects:
                    available_objects.remove(obj_key)

                placed = True
                print(f"[INFO] Force-placed {obj_config['name']} in room {room.room_id}")
                break

        if not placed:
            print(f"[WARN] Could not force-place object {i+1} in room {room.room_id}")

    return room_objects


def _try_place_object_relaxed(self, room: RoomInfo, obj_config: dict, obj_key: str) -> Optional[PlacedObject]:
    """Try to place object with relaxed collision constraints"""
    scale = obj_config["scale"]
    is_custom_model = obj_config.get("is_custom_model", False)
    model_lib = self.model_lib

    # Get model dimensions
    record = model_lib.get_record(obj_config["model"])
    if not record:
        return None

    bounds = record.bounds
    w = abs(bounds["right"]["x"] - bounds["left"]["x"]) * scale
    d = abs(bounds["front"]["z"] - bounds["back"]["z"]) * scale

    # Use same minimum distance for relaxed placement (can be adjusted if needed)
    relaxed_min_distance = self.min_distance * 0.8  # Slightly relaxed distance

    # Try more positions with relaxed constraints
    max_attempts = 200  # Increased attempts
    for attempt in range(max_attempts):
        # More aggressive position selection
        if attempt < 100:
            # First 100 attempts: standard placement
            valid_cells = [(r, c) for r, c in room.cells
                          if self.room_analyzer.is_position_valid_for_placement(
                              *self.room_analyzer._cell_to_world(r, c), room.room_id)]
        else:
            # Later attempts: use all room cells
            valid_cells = list(room.cells)

        if not valid_cells:
            continue

        row, col = self.rng.choice(valid_cells)
        x, z = self.room_analyzer._cell_to_world(row, col)

        # Use relaxed overlap checking (also consider default_position)
        # For relaxed mode, use the existing relaxed function as is for now
        if not self._overlaps_with_existing_relaxed(x, z, w, d, room.room_id, relaxed_min_distance):
            # Check collinearity (use final coordinates)
            cand_fx, cand_fz = self._final_center_from_candidate(x, z, obj_config)
            if not self._is_position_safe_from_collinearity(cand_fx, cand_fz, room.room_id):
                continue
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

            # Set base position/rotation (default_position handled in get_final_position)
            base_pos = {"x": x, "y": 0.05, "z": z}
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
            return placed_obj

    return None


def _overlaps_with_existing_relaxed(self, x: float, z: float, w: float, d: float, room_id: int, min_distance: float) -> bool:
    """Relaxed collision detection with smaller safety margins"""
    # First check if the grid position is already occupied
    if self.use_grid_occupancy:
        check_row, check_col = self.room_analyzer._world_to_cell(x, z)
        if (int(check_row), int(check_col)) in self.occupied_rc:
            return True

    # Even in relaxed mode, door 8-neighborhood checks must be strict
    int_x, int_z = int(round(x)), int(round(z))

    print(f"[RELAXED] Checking relaxed-mode conflicts at ({int_x}, {int_z})")

    # Strictly check all door 8-neighborhoods, even in relaxed mode
    if hasattr(self.room_analyzer, 'doors'):
        print(f"[RELAXED] Strict door 8-neighborhood checks ({len(self.room_analyzer.doors)} doors)")
        for door_id, door_info in self.room_analyzer.doors.items():
            # Check all doors, regardless of connected rooms
            door_x, door_z = door_info.center
            int_door_x, int_door_z = int(round(door_x)), int(round(door_z))
            print(f"[RELAXED] Checking door {door_id} at ({int_door_x}, {int_door_z})")

            # Even relaxed mode enforces door 8-neighborhoods
            if abs(int_x - int_door_x) <= 1 and abs(int_z - int_door_z) <= 1:
                print(f"[RELAXED] ❌ Position ({int_x}, {int_z}) is within door {door_id} 8-neighborhood")
                return True
            else:
                print(f"[RELAXED] ✅ Position ({int_x}, {int_z}) is safe from door {door_id} ({int_door_x}, {int_door_z})")

    # Relaxed mode checks direct overlap only for agent and objects
    # Check agent direct conflicts only
    if self.agent and self.agent.room_id == room_id:
        agent_x, agent_z = int(round(self.agent.pos["x"])), int(round(self.agent.pos["z"]))
        if int_x == agent_x and int_z == agent_z:
            print(f"[RELAXED] ❌ Position ({int_x}, {int_z}) directly overlaps agent")
            return True

    # Check object direct conflicts only
    room_objects = self.rooms_objects.get(room_id, [])
    for obj in room_objects:
        obj_final_pos = obj.get_final_position()
        obj_x, obj_z = int(round(obj_final_pos["x"])), int(round(obj_final_pos["z"]))
        if int_x == obj_x and int_z == obj_z:
            print(f"[RELAXED] ❌ Position ({int_x}, {int_z}) directly overlaps object {obj.name}")
            return True

    print(f"[RELAXED] ✅ Position ({int_x}, {int_z}) passed relaxed-mode checks")
    return False
