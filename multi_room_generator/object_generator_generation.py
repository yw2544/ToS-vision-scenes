"""
Object generator room distribution logic.
"""

from typing import List, Dict, Tuple, Optional

from .room_analyzer import RoomInfo


def generate_all_rooms(self, main_room_id: Optional[int] = None, total_objects: Optional[int] = None, fix_object_n: Optional[List[int]] = None) -> bool:
    """
    Generate objects for all rooms

    Args:
        main_room_id: ID of main room for agent placement (if None, use largest room)
        total_objects: Total number of objects to distribute across all rooms (mutually exclusive with fix_object_n)
        fix_object_n: List specifying number of objects for each room (mutually exclusive with total_objects)

    Returns:
        True if generation successful, False otherwise
    """
    print("[INFO] Starting multi-room object generation...")
    self.occupied_rc.clear()

    # Validate mutually exclusive parameters
    if total_objects is not None and fix_object_n is not None:
        raise ValueError("total_objects and fix_object_n are mutually exclusive")
    if total_objects is None and fix_object_n is None:
        # If neither is provided, use original random logic
        total_objects = None
    # Determine main room for agent placement - always use room 1
    if main_room_id is None:
        if 1 in self.room_analyzer.rooms:
            main_room_id = 1
            print("[INFO] Agent will be placed in room 1 (fixed)")
        else:
            # Fallback to largest room if room 1 doesn't exist
            largest_room = self.room_analyzer.get_largest_room()
            if not largest_room:
                print("[ERROR] No rooms found for object generation")
                return False
            main_room_id = largest_room.room_id
            print(f"[WARN] Room 1 not found, using room {main_room_id} instead")

    print(f"[INFO] Main room for agent: {main_room_id}")

    # Sort rooms by size (largest first) to prioritize object placement
    sorted_rooms = sorted(
        self.room_analyzer.rooms.values(),
        key=lambda r: r.usable_area,
        reverse=True
    )

    # Generate objects room by room
    if fix_object_n is not None:
        # Use fixed object counts for each room
        self._distribute_objects_fixed_counts(sorted_rooms, main_room_id, fix_object_n)
    elif total_objects is not None:
        # Distribute specific total number of objects across rooms
        self._distribute_objects_across_rooms(sorted_rooms, main_room_id, total_objects)
    else:
        # Use original random distribution logic
        self._generate_objects_original_logic(sorted_rooms, main_room_id)

    print(f"[INFO] Generated total {len(self.all_objects)} objects across {len(self.rooms_objects)} rooms")
    print(f"[INFO] Used categories: {sorted(list(self.used_categories))}")

    return True


def _generate_objects_original_logic(self, sorted_rooms: List[RoomInfo], main_room_id: int):
    """Original random object distribution logic"""
    for room in sorted_rooms:
        room_id = room.room_id
        is_main_room = (room_id == main_room_id)

        print(f"[INFO] Generating objects for room {room_id} (main: {is_main_room})")

        # Determine number of objects for this room
        max_objects = self.room_analyzer.get_max_objects_for_room(room_id)
        if max_objects == 0:
            print(f"[INFO] Room {room_id} too small for objects")
            self.rooms_objects[room_id] = []
            continue

        # Random number of objects (1 to max_objects for rooms that can fit objects)
        # Ensure at least 1 object if room can fit any
        num_objects = self.rng.randint(1, max_objects + 1) if max_objects > 0 else 0
        print(f"[INFO] Room {room_id}: generating {num_objects}/{max_objects} objects")

        # Generate objects for this room
        room_objects = self._generate_room_objects(room, num_objects, is_main_room)
        self.rooms_objects[room_id] = room_objects
        self.all_objects.extend(room_objects)

        print(f"[INFO] Room {room_id}: successfully generated {len(room_objects)} objects")


def _distribute_objects_fixed_counts(self, sorted_rooms: List[RoomInfo], main_room_id: int, fix_object_n: List[int]):
    """Distribute objects using fixed counts for each room"""
    print(f"[INFO] Distributing objects using fixed counts: {fix_object_n}")

    room_ids = sorted(self.room_analyzer.rooms.keys())
    if len(fix_object_n) != len(room_ids):
        raise ValueError(f"fix_object_n length ({len(fix_object_n)}) must match number of rooms ({len(room_ids)})")

    room_targets = {room_id: int(fix_object_n[idx]) for idx, room_id in enumerate(room_ids)}

    valid_rooms: List[Tuple[RoomInfo, int]] = []
    for room in sorted_rooms:
        max_objects = self.room_analyzer.get_max_objects_for_room(room.room_id)
        target = room_targets.get(room.room_id, 0)

        if max_objects <= 0:
            if target > 0:
                raise RuntimeError(f"Room {room.room_id} cannot place {target} objects (capacity=0)")
            self.rooms_objects[room.room_id] = []
            continue

        if target > max_objects:
            raise RuntimeError(f"Room {room.room_id} target {target} exceeds capacity {max_objects}")

        valid_rooms.append((room, max_objects))

    if not valid_rooms:
        print("[ERROR] No rooms can accommodate objects")
        raise RuntimeError("Cannot place any objects - no valid rooms")

    target_total = sum(room_targets.values())
    objects_placed = 0
    max_overall_attempts = 50
    success = False

    for overall_attempt in range(max_overall_attempts):
        print(f"[INFO] Overall fixed-count attempt {overall_attempt + 1}/{max_overall_attempts}")

        self.all_objects = []
        for room in sorted_rooms:
            self.rooms_objects[room.room_id] = []
        self.used_categories = set()
        self.occupied_rc.clear()

        main_room = None
        for room, _ in valid_rooms:
            if room.room_id == main_room_id:
                main_room = room
                break

        if main_room is None:
            print(f"[ERROR] Main room {main_room_id} not found in valid rooms")
            break

        agent_success = self._place_agent_in_room(main_room)
        if not agent_success:
            print(f"[WARN] Failed to place agent in main room {main_room_id}")

        objects_placed = self._place_objects_with_retries(valid_rooms, main_room_id, target_total, room_targets=room_targets)

        success = all(len(self.rooms_objects.get(room_id, [])) == room_targets.get(room_id, 0) for room_id in room_targets)
        if success:
            print(f"[SUCCESS] Placed all fixed-count objects ({objects_placed}/{target_total})")
            break

        print(f"[WARN] Attempt {overall_attempt + 1} did not satisfy per-room targets, retrying...")

    if not success:
        print(f"[ERROR] Failed to satisfy fixed counts after {max_overall_attempts} attempts")
        print(f"[ERROR] Placed {objects_placed}/{target_total} objects")
        raise RuntimeError("Fixed count object generation failed to satisfy per-room targets")

    for room in sorted_rooms:
        self.rooms_objects.setdefault(room.room_id, [])

    for room_id, target in room_targets.items():
        actual = len(self.rooms_objects.get(room_id, []))
        if actual != target:
            print(f"[WARN] Room {room_id} expected {target} objects but placed {actual}")


def _distribute_objects_across_rooms(self, sorted_rooms: List[RoomInfo], main_room_id: int, total_objects: int):
    """Distribute specified total number of objects across all rooms - MUST achieve exact count with strict collision detection"""
    print(f"[INFO] Distributing {total_objects} objects across {len(sorted_rooms)} rooms (EXACT COUNT REQUIRED, STRICT COLLISION DETECTION)")

    # Filter rooms that can accommodate objects
    valid_rooms = []
    for room in sorted_rooms:
        max_objects = self.room_analyzer.get_max_objects_for_room(room.room_id)
        if max_objects > 0:
            valid_rooms.append((room, max_objects))
        else:
            self.rooms_objects[room.room_id] = []

    if not valid_rooms:
        print("[ERROR] No rooms can accommodate objects")
        raise RuntimeError("Cannot place any objects - no valid rooms")

    # Calculate total capacity
    total_capacity = sum(max_obj for _, max_obj in valid_rooms)
    if total_capacity < total_objects:
        print(f"[ERROR] Total room capacity ({total_capacity}) < required objects ({total_objects})")
        raise RuntimeError(f"Insufficient room capacity: need {total_objects}, have {total_capacity}")

    # Use persistent retry strategy to guarantee exact count
    objects_placed = 0
    max_overall_attempts = 50  # Increased overall attempts

    for overall_attempt in range(max_overall_attempts):
        print(f"[INFO] Overall placement attempt {overall_attempt + 1}/{max_overall_attempts}")

        # Reset for this attempt
        self.all_objects = []
        for room in sorted_rooms:
            self.rooms_objects[room.room_id] = []
        self.used_categories = set()
        self.occupied_rc.clear()
        # Place agent first in main room
        main_room = None
        for room, _ in valid_rooms:
            if room.room_id == main_room_id:
                main_room = room
                break

        if main_room is None:
            print(f"[ERROR] Main room {main_room_id} not found in valid rooms")
            continue

        agent_success = self._place_agent_in_room(main_room)
        if not agent_success:
            print(f"[WARN] Failed to place agent in main room {main_room_id}")

        # Try to place objects with persistent retry
        objects_placed = self._place_objects_with_retries(valid_rooms, main_room_id, total_objects)

        if objects_placed >= total_objects:
            print(f"[SUCCESS] Placed {objects_placed}/{total_objects} objects successfully with strict collision detection")
            break
        else:
            print(f"[WARN] Attempt {overall_attempt + 1} placed {objects_placed}/{total_objects} objects, retrying...")

    if objects_placed < total_objects:
        print(f"[ERROR] Failed to place required {total_objects} objects after {max_overall_attempts} attempts")
        print(f"[ERROR] Only managed to place {objects_placed} objects")
        raise RuntimeError("Total object generation failed to satisfy required count")

    # Fill remaining rooms with empty lists
    for room in sorted_rooms:
        if room.room_id not in self.rooms_objects:
            self.rooms_objects[room.room_id] = []


def _place_objects_with_retries(self, valid_rooms: List[Tuple[RoomInfo, int]], main_room_id: int, target_count: int, room_targets: Optional[Dict[int, int]] = None) -> int:
    """Place objects with persistent retries, strict collision detection, and integer positions

    Strategy: Two-phase placement
    Phase 1: Prioritize placing oriented objects to meet each room's requirement
    Phase 2: Place remaining objects normally
    """
    room_target_full = {rid: max(0, int(cnt)) for rid, cnt in (room_targets or {}).items()}
    remaining_per_room = room_target_full.copy()
    target_total = target_count if room_targets is None else sum(remaining_per_room.values())
    objects_placed = 0
    remaining_objects = target_total

    if target_total < 0:
        target_total = 0
        remaining_objects = 0
    # Sort rooms by capacity (largest first for better success rate)
    rooms_by_capacity = sorted(valid_rooms, key=lambda x: x[1], reverse=True)

    # Track oriented objects requirements per room
    # Main room (room 1) needs 3, others need 2
    oriented_requirements = {}
    total_oriented_needed = 0
    for room, _ in rooms_by_capacity:
        is_main = (room.room_id == main_room_id)
        required = 3 if is_main else 2
        if room_target_full:
            target_limit = room_target_full.get(room.room_id, None)
            if target_limit is not None:
                required = min(required, max(0, target_limit))
        oriented_requirements[room.room_id] = required
        total_oriented_needed += required

    print(f"[INFO] Oriented objects requirements: {oriented_requirements} (total: {total_oriented_needed})")

    # PHASE 1: Prioritize oriented objects for each room
    print(f"\n[PHASE 1] Placing oriented objects to meet requirements...")
    phase1_rounds = 30  # More rounds for phase 1
    for round_num in range(phase1_rounds):
        # Check if all rooms have met their oriented requirements
        all_satisfied = True
        for room, _ in rooms_by_capacity:
            current_oriented = sum(1 for obj in self.rooms_objects.get(room.room_id, []) if obj.has_orientation)
            if current_oriented < oriented_requirements[room.room_id]:
                all_satisfied = False
                break

        if all_satisfied:
            print(f"[PHASE 1] All rooms satisfied their oriented requirements after {round_num} rounds!")
            break

        if remaining_objects <= 0:
            break

        placed_this_round = 0

        # Prioritize rooms that haven't met their oriented requirements
        for room, max_capacity in rooms_by_capacity[:]:
            if remaining_objects <= 0:
                break

            if room_targets is not None and remaining_per_room.get(room.room_id, 0) <= 0:
                continue

            current_room_objects = self.rooms_objects.get(room.room_id, [])
            current_oriented_count = sum(1 for obj in current_room_objects if obj.has_orientation)
            required_oriented = oriented_requirements.get(room.room_id, 2)

            # Skip rooms that already met their oriented requirement
            if current_oriented_count >= required_oriented:
                continue

            current_room_count = len(current_room_objects)
            available_slots = max_capacity - current_room_count
            if room_targets is not None:
                available_slots = min(available_slots, remaining_per_room.get(room.room_id, 0))

            if available_slots <= 0:
                continue

            # FORCE oriented object placement
            is_main_room = (room.room_id == main_room_id)
            new_object = self._try_place_single_object_strict(room, is_main_room, prefer_oriented=True, force_oriented=True)

            if new_object:
                if room.room_id not in self.rooms_objects:
                    self.rooms_objects[room.room_id] = []
                self.rooms_objects[room.room_id].append(new_object)
                self.all_objects.append(new_object)

                placed_this_round += 1
                remaining_objects -= 1
                if room_targets is not None:
                    remaining_per_room[room.room_id] = max(0, remaining_per_room.get(room.room_id, 0) - 1)
                objects_placed += 1

                oriented_marker = "✓ ORIENTED" if new_object.has_orientation else "⚠ NOT ORIENTED"
                print(f"[PHASE1] Placed {new_object.name} {oriented_marker} in room {room.room_id} ({objects_placed}/{target_total} total)")

        if placed_this_round == 0:
            print(f"[PHASE 1 WARN] No oriented objects placed in round {round_num + 1}")
            continue

    # PHASE 2: Place remaining objects normally
    print(f"\n[PHASE 2] Placing remaining {remaining_objects} objects...")
    max_rounds = 30
    for round_num in range(max_rounds):
        if remaining_objects <= 0:
            break

        placed_this_round = 0

        for room, max_capacity in rooms_by_capacity[:]:
            if remaining_objects <= 0:
                break

            if room_targets is not None and remaining_per_room.get(room.room_id, 0) <= 0:
                continue

            current_room_objects = self.rooms_objects.get(room.room_id, [])
            current_room_count = len(current_room_objects)
            available_slots = max_capacity - current_room_count
            if room_targets is not None:
                available_slots = min(available_slots, remaining_per_room.get(room.room_id, 0))

            if available_slots <= 0:
                continue

            # Normal placement (still prefer oriented if needed, but not forced)
            is_main_room = (room.room_id == main_room_id)
            current_oriented_count = sum(1 for obj in current_room_objects if obj.has_orientation)
            required_oriented = oriented_requirements.get(room.room_id, 2)
            need_oriented = current_oriented_count < required_oriented

            new_object = self._try_place_single_object_strict(room, is_main_room, prefer_oriented=need_oriented, force_oriented=False)

            if new_object:
                if room.room_id not in self.rooms_objects:
                    self.rooms_objects[room.room_id] = []
                self.rooms_objects[room.room_id].append(new_object)
                self.all_objects.append(new_object)

                placed_this_round += 1
                remaining_objects -= 1
                if room_targets is not None:
                    remaining_per_room[room.room_id] = max(0, remaining_per_room.get(room.room_id, 0) - 1)
                objects_placed += 1

                oriented_marker = "✓ oriented" if new_object.has_orientation else ""
                print(f"[PHASE2] Placed {new_object.name} {oriented_marker} in room {room.room_id} ({objects_placed}/{target_total} total)")

        if placed_this_round == 0:
            print(f"[PHASE 2 WARN] No objects placed in round {round_num + 1}")
            continue

    # Print final oriented objects summary
    print(f"\n[INFO] Final oriented objects distribution:")
    for room, _ in rooms_by_capacity:
        room_objs = self.rooms_objects.get(room.room_id, [])
        oriented_count = sum(1 for obj in room_objs if obj.has_orientation)
        required = oriented_requirements.get(room.room_id, 2)
        status = "✓" if oriented_count >= required else "✗"
        print(f"  Room {room.room_id}: {oriented_count}/{required} oriented objects {status}")

    return objects_placed
