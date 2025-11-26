"""
Object Generator Module
======================

Generates objects in rooms based on:
- Room size constraints
- Object placement optimization (visibility + occlusion)
- No duplicate objects across all rooms
- Agent placement only in the largest/main room
"""

import math
import random
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set, Any
import numpy as np

from .room_analyzer import RoomAnalyzer, RoomInfo
from .collinear_validator import CollinearValidator, are_three_points_collinear
from .object_components import (
    COLORS,
    AgentInfo,
    PlacedObject,
    RoomPlacementCache,
    build_room_placement_cache,
)

class ObjectGenerator:
    """Generates objects in multiple rooms with optimization"""
    
    def __init__(self, room_analyzer: RoomAnalyzer, seed: int = 42, min_distance: float = 1.1, 
                 builtin_models_path: Optional[str] = "tos_data_gen/models/builtin_models.json",
                 custom_models_path: Optional[str] = "tos_data_gen/models/custom_models.json",
                 tolerance_width: float = 1.0,
                 debug_enabled: bool = False):
        """
        Initialize object generator
        
        Args:
            room_analyzer: Analyzed room layout
            seed: Random seed for reproducible generation
            min_distance: Minimum distance between objects
            builtin_models_path: Path to build_in_model.json
            custom_models_path: Path to tdw_models_compiled_complete.json
            tolerance_width: Tolerance width for collinear detection
        """
        self.room_analyzer = room_analyzer
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.min_distance = min_distance
        
        # Generation results
        self.rooms_objects: Dict[int, List[PlacedObject]] = {}  # room_id -> objects in that room
        self.all_objects: List[PlacedObject] = []  # All objects across all rooms
        self.agent: Optional[AgentInfo] = None  # Agent placement (only in main room)
        self.used_categories: Set[str] = set()  # Track used categories to avoid duplicates
        self.occupied_rc: Set[Tuple[int, int]] = set()  # Global occupied (row, col)
        self.use_grid_occupancy: bool = True            # Enable/disable integer grid occupancy check
        # Model library for bounds calculation (will be set from main script)
        self.model_lib = None
        self.debug_enabled = debug_enabled
        
        # Model loading from JSON files
        self.builtin_models_path = builtin_models_path
        self.custom_models_path = custom_models_path
        self.models_by_category: Dict[str, List[Dict]] = {}  # category -> [model1, model2, ...]
        
        # Collinear validation
        self.collinear_validator = CollinearValidator(tolerance_width=tolerance_width)
        self._room_position_cache: Dict[int, RoomPlacementCache] = {}
        
        self._load_models_from_json()
    
    def set_model_library(self, lib):
        """Set the TDW model library for bounds calculation"""
        self.model_lib = lib

    def _debug(self, message: str):
        if self.debug_enabled:
            print(message)

    def _get_room_position_cache(self, room: RoomInfo) -> RoomPlacementCache:
        cache = self._room_position_cache.get(room.room_id)
        if cache is None:
            cache = build_room_placement_cache(self.room_analyzer, room)
            self._room_position_cache[room.room_id] = cache
        return cache
    
    def _load_models_from_json(self):
        """Load all models from JSON and group them by category."""
        import json
        from pathlib import Path
        
        self.models_by_category = {}
        
        # 1. Load builtin models (build_in_model.json)
        if self.builtin_models_path and Path(self.builtin_models_path).exists():
            try:
                with open(self.builtin_models_path, 'r', encoding='utf-8') as f:
                    builtin_models = json.load(f)
                
                for model_data in builtin_models:
                    category = model_data.get("category", "unknown")
                    
                    # Normalize model payload
                    processed_model = {
                        "name": model_data["name"],
                        "model": model_data["model_name"],
                        "scale": model_data["scale"],
                        "color": self._map_color_name_to_value(model_data.get("color")),
                        "has_orientation": model_data["has_orientation"],
                        "category": category,
                        "source": "builtin",
                        "is_custom_model": False,
                        "default_position": model_data.get("default_position", {"x": 0, "y": 0, "z": 0}),
                        "default_rotation": model_data.get("default_rotation", {"x": 0, "y": 0, "z": 0})
                    }
                    
                    if category not in self.models_by_category:
                        self.models_by_category[category] = []
                    self.models_by_category[category].append(processed_model)
                
                builtin_count = sum(len(models) for models in self.models_by_category.values())
                print(f"[INFO] Loaded {builtin_count} builtin models from {self.builtin_models_path}")
            except Exception as e:
                print(f"[ERROR] Failed to load builtin models: {e}")
        
        # 2. Load custom models (tdw_models_compiled_complete.json)
        if self.custom_models_path and Path(self.custom_models_path).exists():
            try:
                with open(self.custom_models_path, 'r', encoding='utf-8') as f:
                    custom_models = json.load(f)
                
                custom_count = 0
                for model_data in custom_models:
                    category = model_data.get("category", model_data["name"])
                    
                    # Normalize model payload
                    processed_model = {
                        "name": model_data["name"],
                        "model": model_data["model_name"],
                        "scale": model_data["scale"],
                        "color": self._map_color_name_to_value(model_data.get("color")),
                        "has_orientation": model_data["has_orientation"],
                        "category": category,
                        "source": "custom",
                        "is_custom_model": True,
                        "default_position": model_data.get("default_position", {"x": 0, "y": 0, "z": 0}),
                        "default_rotation": model_data.get("default_rotation", {"x": 0, "y": 0, "z": 0}),
                        "record": model_data.get("record")
                    }
                    
                    if category not in self.models_by_category:
                        self.models_by_category[category] = []
                    self.models_by_category[category].append(processed_model)
                    custom_count += 1
                
                print(f"[INFO] Loaded {custom_count} custom models from {self.custom_models_path}")
            except Exception as e:
                print(f"[ERROR] Failed to load custom models: {e}")
        
        total_models = sum(len(models) for models in self.models_by_category.values())
        print(f"[INFO] Loaded {total_models} models across {len(self.models_by_category)} categories")
        print(f"[INFO] Available categories: {sorted(self.models_by_category.keys())}")
        print(f"[INFO] Model counts per category: {[(cat, len(models)) for cat, models in sorted(self.models_by_category.items())]}")
    
    def _map_color_name_to_value(self, color_name: Optional[str]) -> Optional[str]:
        """Map color identifiers to the generator color palette."""
        if color_name is None:
            return None
        
        if isinstance(color_name, dict):
            return color_name
        
        if isinstance(color_name, str):
            color_name_lower = color_name.lower()
            if color_name_lower in COLORS:
                return color_name_lower
        else:
                print(f"[WARN] Unknown color name: {color_name}, using default")
                return None
        
        return None
    
    def get_models_by_category(self) -> Dict[str, List[Dict]]:
        """Return all models grouped by category."""
        return self.models_by_category
    
    def select_objects_by_categories(self, num_objects: int, preferred_categories: Optional[List[str]] = None, min_oriented_objects: int = 0) -> List[Dict]:
        """Select objects without repeating categories while meeting orientation quotas."""
        if num_objects <= 0:
            return []
        
        available_categories = sorted([cat for cat in self.models_by_category.keys() 
                              if cat not in self.used_categories])
        
        if len(available_categories) < num_objects:
            print(f"[WARN] Available categories ({len(available_categories)}) fewer than requested objects ({num_objects})")
            num_objects = len(available_categories)
        
        if num_objects == 0:
            return []
        
        oriented_categories = []
        non_oriented_categories = []
        
        for cat in available_categories:
            models_in_cat = self.models_by_category[cat]
            has_oriented = any(model.get('has_orientation', False) for model in models_in_cat)
            if has_oriented:
                oriented_categories.append(cat)
            else:
                non_oriented_categories.append(cat)
        
        selected_categories = []
        
        if preferred_categories:
            for pref_cat in preferred_categories:
                if pref_cat in available_categories and len(selected_categories) < num_objects:
                    selected_categories.append(pref_cat)
                    available_categories.remove(pref_cat)
                    if pref_cat in oriented_categories:
                        oriented_categories.remove(pref_cat)
                    elif pref_cat in non_oriented_categories:
                        non_oriented_categories.remove(pref_cat)
        
        oriented_count = sum(1 for cat in selected_categories if cat in self.models_by_category and 
                           any(m.get('has_orientation', False) for m in self.models_by_category[cat]))
        
        oriented_needed = max(0, min_oriented_objects - oriented_count)
        remaining_slots = num_objects - len(selected_categories)
        
        if oriented_needed > 0 and oriented_categories:
            num_to_select = min(oriented_needed, len(oriented_categories), remaining_slots)
            if num_to_select > 0:
                sorted_oriented = sorted(oriented_categories)
                selected_oriented = self.rng.choice(
                    sorted_oriented,
                    size=num_to_select,
                    replace=False
                )
                selected_categories.extend(selected_oriented)
                for cat in selected_oriented:
                    oriented_categories.remove(cat)
                remaining_slots -= num_to_select
                print(f"[INFO] Added {num_to_select} oriented categories (min requirement {min_oriented_objects})")
        
        if remaining_slots > 0:
            remaining_categories = sorted(oriented_categories + non_oriented_categories)
            if remaining_categories:
                num_to_select = min(remaining_slots, len(remaining_categories))
                additional_categories = self.rng.choice(
                    remaining_categories,
                    size=num_to_select,
                    replace=False
                )
                selected_categories.extend(additional_categories)
        
        selected_objects = []
        for category in selected_categories:
            models_in_category = self.models_by_category[category]
            oriented_models = [m for m in models_in_category if m.get('has_orientation', False)]
            if oriented_models and category in (oriented_categories or []):
                selected_model = self.rng.choice(oriented_models)
            else:
                selected_model = self.rng.choice(models_in_category)
            selected_objects.append(selected_model)
        
        actual_oriented = sum(1 for obj in selected_objects if obj.get('has_orientation', False))
        print(f"[INFO] Selected {len(selected_objects)} objects from categories {[obj['category'] for obj in selected_objects]}")
        print(f"[INFO] Objects with orientation: {actual_oriented} (required >= {min_oriented_objects})")
        
        return selected_objects
    
    def _place_object_in_room_new(self, room: RoomInfo, obj_config: Dict) -> Optional[PlacedObject]:
        """use new model format"""
        return self._place_object_in_room(room, obj_config, f"new_{obj_config['category']}_{obj_config['model']}")
    
    def get_current_object_pool(self) -> Dict[str, Any]:
        """Compatibility helper to expose models in the legacy object_pool format."""
        object_pool = {}
        # Sort categories for deterministic order
        for category in sorted(self.models_by_category.keys()):
            models = self.models_by_category[category]
            for i, model in enumerate(models):
                key = f"{model['source']}_{category}_{model['model']}_{i}"
                object_pool[key] = model
        return object_pool
    
    def _debug_min_center_gap(self, room_id: int, min_gap: float = 1.1):
        """Check the minimum center-to-center distance between objects in a room"""
        objs = self.rooms_objects.get(room_id, [])
        if len(objs) < 2:
            return
            
        worst = None
        for i in range(len(objs)):
            pi = objs[i].get_final_position()
            for j in range(i+1, len(objs)):
                pj = objs[j].get_final_position()
                d = ((pi["x"]-pj["x"])**2 + (pi["z"]-pj["z"])**2) ** 0.5
                if worst is None or d < worst["d"]:
                    worst = {"pair": (objs[i].name, objs[j].name), "d": d, "pos": (pi, pj)}
        
        if worst:
            if worst["d"] <= min_gap:
                print(f"[VIOLATION][room {room_id}] {worst['pair']} dist={worst['d']:.3f} <= {min_gap} @ positions {worst['pos'][0]} <-> {worst['pos'][1]}")
            else:
                print(f"[OK][room {room_id}] min distance {worst['d']:.3f} > {min_gap}")
            
            if self.debug_enabled:
                self._debug(f"[DEBUG][room {room_id}] pairwise object distances:")
                for i in range(len(objs)):
                    pi = objs[i].get_final_position()
                    for j in range(i+1, len(objs)):
                        pj = objs[j].get_final_position()
                        d = ((pi["x"]-pj["x"])**2 + (pi["z"]-pj["z"])**2) ** 0.5
                        status = "VIOLATION" if d <= min_gap else "OK"
                        self._debug(f"  [{status}] {objs[i].name}({pi['x']:.1f},{pi['z']:.1f}) <-> {objs[j].name}({pj['x']:.1f},{pj['z']:.1f}) = {d:.3f}")
    
    
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
                print(f"[PHASE 1 WARN] No objects placed in round {round_num + 1}")
        
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
        
        # Strict attempt failed; skip relaxed placement to preserve category uniqueness
        mode_str = "FORCED oriented" if force_oriented else "strict mode"
        print(f"[INFO] {mode_str} failed but relaxed placement is skipped to keep categories unique")
        return None
    
    def _try_place_object_at_integer_positions(self, room: RoomInfo, obj_config: dict, obj_key: str) -> Optional[PlacedObject]:
        """Try to place object at integer world coordinates with strict collision detection"""
        self._debug(f"\n[DEBUG] Trying to place {obj_key} in room {room.room_id}")
        scale = obj_config["scale"]
        is_custom_model = obj_config.get("is_custom_model", False)

        if is_custom_model:
            w, d = 1.0, 1.0
        else:
            model_lib = self.model_lib
            record = model_lib.get_record(obj_config["model"])
            if not record:
                return None
            bounds = record.bounds
            w = abs(bounds["right"]["x"] - bounds["left"]["x"]) * scale
            d = abs(bounds["front"]["z"] - bounds["back"]["z"]) * scale

        cache = self._get_room_position_cache(room)
        if not cache.parity_buckets:
            return None

        best_key = max(sorted(cache.parity_buckets.keys()), key=lambda k: len(cache.parity_buckets[k]))
        primary_positions = list(cache.parity_buckets[best_key])
        all_positions = list(cache.all_positions)

        def _filter_unoccupied(positions: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
            if not self.use_grid_occupancy:
                return positions
            filtered: List[Tuple[int, int]] = []
            for ix, iz in positions:
                rr, cc = self.room_analyzer._world_to_cell(float(ix), float(iz))
                if (int(rr), int(cc)) not in self.occupied_rc:
                    filtered.append((ix, iz))
            return filtered

        primary_before = len(primary_positions)
        primary_positions = _filter_unoccupied(primary_positions)
        all_positions = _filter_unoccupied(all_positions)
        if self.debug_enabled:
            self._debug(f"[DEBUG] Grid occupancy filter (primary): {primary_before} -> {len(primary_positions)} positions")

        if primary_positions:
            primary_positions.sort(key=lambda pos: (pos[0], pos[1]))
            self.rng.shuffle(primary_positions)
            result = self._attempt_positions(room, obj_config, primary_positions, w, d, obj_key, "Primary")
            if result:
                return result

        print(f"[INFO] Primary bucket failed, falling back to all available positions")
        if all_positions:
            all_positions.sort(key=lambda pos: (pos[0], pos[1]))
            self.rng.shuffle(all_positions)
            return self._attempt_positions(room, obj_config, all_positions, w, d, obj_key, "Fallback")

        return None

    def _attempt_positions(
        self,
        room: RoomInfo,
        obj_config: dict,
        positions: List[Tuple[int, int]],
        width: float,
        depth: float,
        obj_key: str,
        stage: str,
    ) -> Optional[PlacedObject]:
        for x, z in positions:
            x_f, z_f = float(x), float(z)
            self._debug(f"[DEBUG] Trying {stage.lower()} position ({x_f:.0f}, {z_f:.0f}) for {obj_key}")
            has_conflict = self._overlaps_with_existing_with_default(x_f, z_f, width, depth, room.room_id, obj_config)
            self._debug(f"[STRICT] {stage} position ({x_f:.0f}, {z_f:.0f}): 8-neighbor conflict={has_conflict}")
            if has_conflict:
                continue
            cand_fx, cand_fz = self._final_center_from_candidate(x_f, z_f, obj_config)
            if not self._is_position_safe_from_collinearity(cand_fx, cand_fz, room.room_id):
                self._debug(f"[DEBUG] {stage} position ({x_f:.0f}, {z_f:.0f}) is collinear, skipping")
                continue
            self._debug(f"[DEBUG] {stage} position ({x_f:.0f}, {z_f:.0f}) passed collision and collinearity checks")
            object_id = self.rng.randint(1000000, 9999999)
            rotation_options = [0, 90, 180, 270]
            rotation_index = (int(x_f) + int(z_f) + object_id) % 4
            rotation_y = rotation_options[rotation_index]
            has_orientation = obj_config.get("has_orientation", False)
            orientation = self._rotation_to_orientation(rotation_y) if has_orientation else None
            base_pos = {"x": x_f, "y": 0.05, "z": z_f}
            base_rot = {"x": 0, "y": rotation_y, "z": 0}
            is_custom_model = obj_config.get("is_custom_model", False)
            placed_obj = PlacedObject(
                object_id=object_id,
                model=obj_config["model"],
                name=obj_config["name"],
                pos=base_pos,
                rot=base_rot,
                size=(width, depth),
                scale=obj_config["scale"],
                color=obj_config["color"],
                room_id=room.room_id,
                has_orientation=has_orientation,
                orientation=orientation,
                is_custom_model=is_custom_model,
                custom_config=obj_config if is_custom_model else None,
                model_config=obj_config,
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
                # Enforce collinearity constraint using final coordinates
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
                
                # Store base position/rotation only; default offsets apply later
                base_pos = {"x": x, "y": 0.05, "z": z}
                base_rot = {"x": 0, "y": rotation_y, "z": 0}
                
                placed_obj = PlacedObject(
                    object_id=object_id,
                    model=obj_config["model"],
                    name=obj_config["name"],
                    pos=base_pos,  # Base position; default offsets applied later
                    rot=base_rot,  # Base rotation; default offsets applied later
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
        
        int_x, int_z = int(round(x)), int(round(z))
        
        self._debug(f"[RELAXED] Checking relaxed conflicts at ({int_x}, {int_z})")
        
        if hasattr(self.room_analyzer, 'doors'):
            self._debug(f"[RELAXED] Verifying door 8-neighborhood constraints for {len(self.room_analyzer.doors)} doors")
            for door_id, door_info in self.room_analyzer.doors.items():
                door_x, door_z = door_info.center
                int_door_x, int_door_z = int(round(door_x)), int(round(door_z))
                self._debug(f"[RELAXED] Checking door {door_id} at ({int_door_x}, {int_door_z})")
                
                if abs(int_x - int_door_x) <= 1 and abs(int_z - int_door_z) <= 1:
                    self._debug(f"[RELAXED] ❌ Position ({int_x}, {int_z}) is in door {door_id}'s 8-neighborhood")
                    return True
                else:
                    self._debug(f"[RELAXED] ✅ Position ({int_x}, {int_z}) is safe relative to door {door_id}")
        
        # Relaxed mode only checks direct overlaps with agent/objects
        if self.agent and self.agent.room_id == room_id:
            agent_x, agent_z = int(round(self.agent.pos["x"])), int(round(self.agent.pos["z"]))
            if int_x == agent_x and int_z == agent_z:
                self._debug(f"[RELAXED] ❌ Position ({int_x}, {int_z}) overlaps with agent")
                return True
        
        room_objects = self.rooms_objects.get(room_id, [])
        for obj in room_objects:
            obj_final_pos = obj.get_final_position()
            obj_x, obj_z = int(round(obj_final_pos["x"])), int(round(obj_final_pos["z"]))
            if int_x == obj_x and int_z == obj_z:
                self._debug(f"[RELAXED] ❌ Position ({int_x}, {int_z}) overlaps with object {obj.name}")
                return True
        
        self._debug(f"[RELAXED] ✅ Position ({int_x}, {int_z}) passed relaxed checks")
        return False
    
    def _generate_room_objects(self, room: RoomInfo, num_objects: int, is_main_room: bool) -> List[PlacedObject]:
        """Generate objects for a specific room"""
        if num_objects == 0:
            return []
        
        room_objects = []
        # Ensure placement helpers can see already-placed objects in this room
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
        
        # Use category-first selection (no hard-coded chair priority)
        preferred_categories = []
        
        # Require different orientation counts per room type (main room needs more)
        min_oriented = 2 if is_main_room else 2
        selected_objects = self.select_objects_by_categories(num_objects, preferred_categories, min_oriented_objects=min_oriented)
        
        if len(selected_objects) < num_objects:
            print(f"[WARN] Room {room.room_id}: only selected {len(selected_objects)} objects instead of {num_objects}")
        
        print(f"[INFO] Room {room.room_id} selected objects: {[obj['name'] for obj in selected_objects]}")
        
        for obj_config in selected_objects:
            placed_obj = self._place_object_in_room_new(room, obj_config)
            if placed_obj:
                room_objects.append(placed_obj)
                self.used_categories.add(obj_config['category'])
        
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
        
        # Debug helper: ensure min distance inside the room
        self._debug_min_center_gap(room.room_id, self.min_distance)
        
        if len(room_objects) >= 3:
            agent_pos = self.agent.pos if is_main_room and self.agent else None
            is_valid, collinear_groups = self.collinear_validator.validate_room_positions(
                room_objects, agent_pos, room.room_id, self.room_analyzer
            )
            
            if is_valid:
                print(f"[INFO] Room {room.room_id} collinearity check passed")
            else:
                print(f"[WARN] Room {room.room_id} still has {len(collinear_groups)} collinear groups (post-check)")
        
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
            
            preferred_categories = []
            
            temp_used = self.used_categories.copy()
            self.used_categories = set()
            
            min_oriented = 3 if is_main_room else 2
            selected_objects = self.select_objects_by_categories(actual_target, preferred_categories, min_oriented_objects=min_oriented)
            
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
                
                # Debug helper: ensure min distance inside the room
                self._debug_min_center_gap(room.room_id, self.min_distance)
                
                if len(room_objects) >= 3:
                    agent_pos = self.agent.pos if is_main_room and self.agent else None
                    is_valid, collinear_groups = self.collinear_validator.validate_room_positions(
                        room_objects, agent_pos, room.room_id, self.room_analyzer
                    )
                    
                    if is_valid:
                        print(f"[INFO] Room {room.room_id} collinearity check passed")
                    else:
                        print(f"[WARN] Room {room.room_id} still has {len(collinear_groups)} collinear groups (post-check)")
                
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
                self._debug(f"[DEBUG] Cell ({row},{col}) has value {cell_value}, not room {room.room_id}")
                continue
            
            # Second check: ensure this cell is NOT a door itself
            if is_door(cell_value):
                self._debug(f"[DEBUG] Cell ({row},{col}) is a door (value {cell_value})")
                continue
            
            # Third check: ensure this cell is far from ALL doors
            min_distance_to_door = float('inf')
            for door_row, door_col in door_positions:
                distance = max(abs(row - door_row), abs(col - door_col))  # Chebyshev distance
                min_distance_to_door = min(min_distance_to_door, distance)
            
            if min_distance_to_door < min_door_distance:
                self._debug(f"[DEBUG] Cell ({row},{col}) is too close to doors (distance {min_distance_to_door} < {min_door_distance})")
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
                            self._debug(f"[DEBUG] Position ({int_x}, {int_z}) conflicts with door center at ({door_x}, {door_z})")
                            break
                    
                    if conflicts_with_door_center:
                        continue
                    
                    # Ensure 8-neighborhood around the agent candidate is free of doors/objects
                    if self._check_8_neighbor_conflicts(int_x, int_z, room.room_id, check_doors=True, check_agent=False, check_objects=True):
                        self._debug(f"[DEBUG] Agent candidate ({int_x},{int_z}) has 8-neighbor conflicts")
                        continue
                    
                    # Occupy grid position
                    rr, cc = self.room_analyzer._world_to_cell(float(int_x), float(int_z))
                    if (int(rr), int(cc)) in self.occupied_rc:
                        continue
                    valid_positions.append((float(int_x), float(int_z)))
                    self._debug(f"[DEBUG] SAFE position found: mask({row},{col}) -> world({int_x},{int_z}) -> verify({verify_row},{verify_col}) distance_to_doors={min_distance_to_door}")
        
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
                                self._debug(f"[DEBUG] Position ({int_x}, {int_z}) conflicts with door center at ({door_x}, {door_z}) [distance 2 check]")
                                break
                        
                        if conflicts_with_door_center:
                            continue
                        
                        # Distance-2 fallback still enforces 8-neighborhood safety
                        if self._check_8_neighbor_conflicts(int_x, int_z, room.room_id, check_doors=True, check_agent=False, check_objects=True):
                            self._debug(f"[DEBUG] Agent candidate ({int_x},{int_z}) has 8-neighbor conflicts [distance 2 check]")
                            continue
                        
                        rr, cc = self.room_analyzer._world_to_cell(float(int_x), float(int_z))
                        if (int(rr), int(cc)) in self.occupied_rc:
                            continue
                        valid_positions.append((float(int_x), float(int_z)))
                        self._debug(f"[DEBUG] SAFE position found (distance 2): mask({row},{col}) -> world({int_x},{int_z})")
        
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
                    self._debug(f"[DEBUG] Final verification: world({x:.0f},{z:.0f}) -> mask({verify_row},{verify_col}) = {actual_cell_value}")
                    
                    # Verify distance to all doors
                    min_dist = float('inf')
                    for door_row, door_col in door_positions:
                        dist = max(abs(verify_row - door_row), abs(verify_col - door_col))
                        min_dist = min(min_dist, dist)
                    self._debug(f"[DEBUG] Agent distance to nearest door: {min_dist} cells")
                    
                    return True
                else:
                    self._debug(f"[DEBUG] Position ({x:.0f}, {z:.0f}) final check failed: cell value {actual_cell_value}")
            
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
            # Evaluate collinearity with final coordinates
            cand_fx, cand_fz = self._final_center_from_candidate(x, z, obj_config)
            safe_collinear = self._is_position_safe_from_collinearity(cand_fx, cand_fz, room.room_id)
            
            print(f"[PLACEMENT] Position ({x:.0f}, {z:.0f}): valid={position_valid}, no_overlap={no_overlap}, no_collinearity={safe_collinear}")
            
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
                
                # Store base position/rotation; default offsets are applied later
                base_pos = {"x": x, "y": 0.05, "z": z}  # Floor height
                base_rot = {"x": 0, "y": rotation_y, "z": 0}
                
                placed_obj = PlacedObject(
                    object_id=object_id,
                    model=obj_config["model"],
                    name=obj_config["name"],
                    pos=base_pos,  # Base position; default offset handled downstream
                    rot=base_rot,  # Base rotation; default offset handled downstream
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
        Check whether a candidate coordinate collides with any door/agent/object 8-neighborhood.
        
        Args:
            center_x, center_z: Integer grid coordinates being evaluated.
            room_id: Room identifier used for filtering.
            check_doors: Whether to test against door cells.
            check_agent: Whether to test against the agent position.
            check_objects: Whether to test against already placed objects.
            
        Returns:
            True if the candidate violates any 8-neighborhood rule, otherwise False.
        """
        print("\n🔍 [8NEIGHBOR] ==========================================")
        print(f"🔍 [8NEIGHBOR] Checking candidate ({center_x}, {center_z}) in room {room_id}")
        print("🔍 [8NEIGHBOR] ==========================================")
        
        if check_doors and hasattr(self.room_analyzer, 'doors'):
            print(f"[8NEIGHBOR] Checking {len(self.room_analyzer.doors)} doors for conflicts")
            for door_id, door_info in self.room_analyzer.doors.items():
                connected_rooms = getattr(door_info, "connected_rooms", [])
                print(f"[8NEIGHBOR] Door {door_id} (rooms: {connected_rooms})")
                
                door_x, door_z = door_info.center
                int_door_x, int_door_z = int(round(door_x)), int(round(door_z))
                print(f"[8NEIGHBOR] Door {door_id} at ({int_door_x}, {int_door_z})")
                
                if abs(center_x - int_door_x) <= 1 and abs(center_z - int_door_z) <= 1:
                    if center_x == int_door_x and center_z == int_door_z:
                        print(f"[8NEIGHBOR] ❌ Candidate overlaps door {door_id}")
                    else:
                        print(f"[8NEIGHBOR] ❌ Candidate is inside door {door_id}'s 8-neighborhood")
                    print("[8NEIGHBOR] ❌ Rejecting placement due to door constraint")
                    return True
                else:
                    print(f"[8NEIGHBOR] ✅ Safe distance from door {door_id}")
        
        if check_agent and self.agent and self.agent.room_id == room_id:
            agent_x, agent_z = int(round(self.agent.pos["x"])), int(round(self.agent.pos["z"]))
            print(f"[8NEIGHBOR] Checking agent at ({agent_x}, {agent_z})")
            if abs(center_x - agent_x) <= 1 and abs(center_z - agent_z) <= 1:
                print(f"[8NEIGHBOR] ❌ Candidate touches agent 8-neighborhood")
                return True
            else:
                print("[8NEIGHBOR] ✅ Safe distance from agent")
        
        if check_objects:
            room_objects = self.rooms_objects.get(room_id, [])
            print(f"[8NEIGHBOR] Checking {len(room_objects)} existing objects")
            for obj in room_objects:
                obj_final_pos = obj.get_final_position()
                obj_x, obj_z = int(round(obj_final_pos["x"])), int(round(obj_final_pos["z"]))
                print(f"[8NEIGHBOR] Object {obj.name} at ({obj_x}, {obj_z})")
                if abs(center_x - obj_x) <= 1 and abs(center_z - obj_z) <= 1:
                    print(f"[8NEIGHBOR] ❌ Candidate touches object {obj.name}'s 8-neighborhood")
                    return True
                else:
                    print(f"[8NEIGHBOR] ✅ Safe distance from object {obj.name}")
        
        print(f"[8NEIGHBOR] ✅ Candidate ({center_x}, {center_z}) passed all checks")
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
        Validate that two object centers are at least min_distance apart.
        """
        distance = math.sqrt((x1 - x2)**2 + (z1 - z2)**2)
        result = distance < min_distance
        if result:
            self._debug(f"[DEBUG] Distance check failed: actual {distance:.3f} < minimum {min_distance}")
        return result
    
    def _final_center_from_candidate(self, x: float, z: float, obj_cfg_or_obj) -> Tuple[float, float]:
        """
        Convert a candidate base position into final center coordinates (includes default_position offsets).
        """
        if isinstance(obj_cfg_or_obj, dict):
            dp = obj_cfg_or_obj.get("default_position", {}) or {}
        else:
            # PlacedObject
            cfg = obj_cfg_or_obj.model_config or {}
            dp = cfg.get("default_position", {}) or {}
        
        return x + float(dp.get("x", 0.0)), z + float(dp.get("z", 0.0))
    
    def _mark_object_occupied(self, placed_obj: PlacedObject):
        """
        Record both base and final grid cells as occupied for this object.
        """
        base_r, base_c = self.room_analyzer._world_to_cell(placed_obj.pos["x"], placed_obj.pos["z"])
        self.occupied_rc.add((int(base_r), int(base_c)))
        
        final_pos = placed_obj.get_final_position()
        final_r, final_c = self.room_analyzer._world_to_cell(final_pos["x"], final_pos["z"])
        if (int(final_r), int(final_c)) != (int(base_r), int(base_c)):
            self.occupied_rc.add((int(final_r), int(final_c)))
    
    def _unmark_object_occupied(self, placed_obj: PlacedObject):
        """
        Clear both base and final grid cells associated with this object.
        """
        base_r, base_c = self.room_analyzer._world_to_cell(placed_obj.pos["x"], placed_obj.pos["z"])
        self.occupied_rc.discard((int(base_r), int(base_c)))
        
        final_pos = placed_obj.get_final_position()
        final_r, final_c = self.room_analyzer._world_to_cell(final_pos["x"], final_pos["z"])
        if (int(final_r), int(final_c)) != (int(base_r), int(base_c)):
            self.occupied_rc.discard((int(final_r), int(final_c)))
    
    def _is_position_safe_from_collinearity(self, x: float, z: float, room_id: int) -> bool:
        """
        Incrementally verify that adding a new object will not introduce collinearity issues.
        """
        room_objects = self.rooms_objects.get(room_id, [])
        
        if len(room_objects) < 2:
            return True
        
        positions = []
        names = []
        
        for obj in room_objects:
            final_pos = obj.get_final_position()
            positions.append((final_pos["x"], final_pos["z"]))
            names.append(obj.name)
        
        if hasattr(self.room_analyzer, 'doors'):
            for door_id, door_info in self.room_analyzer.doors.items():
                if room_id in door_info.connected_rooms:
                    door_x, door_z = door_info.center
                    positions.append((door_x, door_z))
                    names.append(f"door_{door_id}")
        
        if self.agent and self.agent.room_id == room_id:
            positions.append((self.agent.pos["x"], self.agent.pos["z"]))
            names.append("agent")
        
        # Append candidate (assumed final coordinates already)
        cand_idx = len(positions)
        positions.append((x, z))
        names.append("new_object")
        
        tolerance = self.collinear_validator.tolerance
        
        for i in range(cand_idx):
            for j in range(i + 1, cand_idx):
                if are_three_points_collinear(positions[i], positions[j], positions[cand_idx], tolerance):
                    self._debug(f"[DEBUG] Position ({x:.1f}, {z:.1f}) introduces collinearity: {names[i]} - {names[j]} - {names[cand_idx]}")
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
                    
                    # Check validity, overlap, collinearity, and 8-neighbor constraints
                    position_valid = self.room_analyzer.is_position_valid_for_placement(new_x, new_z, room.room_id)
                    no_overlap = not self._would_overlap_at_position(obj, new_x, new_z, room.room_id, agent_pos, room_objects)
                    # Use final coordinates for collinearity evaluation
                    safe_collinear = self._is_position_safe_from_collinearity_during_optimization(new_x, new_z, room.room_id, obj)
                    no_8neighbor_conflict = not self._check_8_neighbor_conflicts(int(round(new_x)), int(round(new_z)), room.room_id)
                    
                    print(f"[OPTIMIZE] Position ({new_x:.0f}, {new_z:.0f}): valid={position_valid}, no_overlap={no_overlap}, no_collinearity={safe_collinear}, no_8neighbor_conflict={no_8neighbor_conflict}")
                    
                    if position_valid and no_overlap and safe_collinear and no_8neighbor_conflict:

                        old_r, old_c = self.room_analyzer._world_to_cell(original_pos["x"], original_pos["z"])
                        new_r, new_c = self.room_analyzer._world_to_cell(new_x, new_z)

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
        Check whether relocating a specific object during optimization introduces collinearity.
        """
        room_objects = self.rooms_objects.get(room_id, [])
        other_objects = [obj for obj in room_objects if obj.object_id != moving_obj.object_id]
        
        if len(other_objects) < 2:
            return True
        
        positions = []
        names = []
        
        for obj in other_objects:
            final_pos = obj.get_final_position()
            positions.append((final_pos["x"], final_pos["z"]))
            names.append(obj.name)
        
        if hasattr(self.room_analyzer, 'doors'):
            for door_id, door_info in self.room_analyzer.doors.items():
                if room_id in door_info.connected_rooms:
                    door_x, door_z = door_info.center
                    positions.append((door_x, door_z))
                    names.append(f"door_{door_id}")
        
        if self.agent and self.agent.room_id == room_id:
            positions.append((self.agent.pos["x"], self.agent.pos["z"]))
            names.append("agent")
        
        cand_idx = len(positions)
        cand_fx, cand_fz = self._final_center_from_candidate(x, z, moving_obj)
        positions.append((cand_fx, cand_fz))
        names.append(moving_obj.name)
        
        tolerance = self.collinear_validator.tolerance
        
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
            
            test_base_pos = {"x": x, "y": test_obj.pos["y"], "z": z}
            test_final_pos = test_obj.get_final_position(test_base_pos)
            
            if self._check_center_distance(test_final_pos["x"], test_final_pos["z"], obj_final_pos["x"], obj_final_pos["z"], self.min_distance):
                return True
        
        return False
    
    def _get_object_bounds(self, model_name: str) -> Tuple[float, float]:
        """Get object 2D bounds from model library or custom models"""
        # First, see if the model is listed as custom in models_by_category
        for category, models in self.models_by_category.items():
            for obj_config in models:
                if obj_config["model"] == model_name and obj_config.get("is_custom_model", False):
                    # Custom model: try reading bounds from its record file
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
                            print(f"[WARN] Failed to read bounds for custom model {model_name} from record file: {e}")
                    
                    print(f"[WARN] Custom model {model_name} falling back to default bounds")
                    return (1.0, 1.0)
        
        # Standard library models
        if not self.model_lib:
            return (1.0, 1.0)  # Default
        
        try:
            record = self.model_lib.get_record(model_name)
            if record is None:
                print(f"[WARN] Model {model_name} not found in standard library, using default bounds")
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
    
    def get_all_objects(self) -> List[PlacedObject]:
        """Get all objects across all rooms"""
        return self.all_objects.copy()
    
    def get_room_objects(self, room_id: int) -> List[PlacedObject]:
        """Get objects in a specific room"""
        return self.rooms_objects.get(room_id, []).copy()
    
    def get_agent(self) -> Optional[AgentInfo]:
        """Get agent information"""
        return self.agent
    
    def validate_category_uniqueness(self) -> bool:
        """Ensure every placed object uses a unique category."""
        all_categories = []
        duplicate_categories = []
        
        for obj in self.all_objects:
            if hasattr(obj, 'model_config') and obj.model_config:
                category = obj.model_config.get('category', 'unknown')
            else:
                category = 'unknown'
            
            if category in all_categories:
                if category not in duplicate_categories:
                    duplicate_categories.append(category)
            else:
                all_categories.append(category)
        
        if duplicate_categories:
            print(f"[ERROR] Duplicate categories detected: {duplicate_categories}")
            print(f"[ERROR] All categories: {all_categories}")
            
            for obj in self.all_objects:
                category = obj.model_config.get('category', 'unknown') if obj.model_config else 'unknown'
                print(f"[ERROR] Object: {obj.name}, category: {category}, room: {obj.room_id}")
            
            return False
        else:
            print(f"[INFO] Category uniqueness validated with {len(all_categories)} unique categories")
            return True
    
    def export_summary(self) -> Dict:
        """Export generation summary."""
        # Validate uniqueness before exporting
        is_unique = self.validate_category_uniqueness()
        
        return {
            "total_objects": len(self.all_objects),
            "rooms_with_objects": len([room_id for room_id, objs in self.rooms_objects.items() if objs]),
            "agent_room": self.agent.room_id if self.agent else None,
            "agent_position": self.agent.pos if self.agent else None,
            "used_categories": sorted(list(self.used_categories)),
            "category_unique": is_unique,
            "rooms_objects": {
                room_id: [
                    {
                        "object_id": obj.object_id,
                        "name": obj.name,
                        "model": obj.model,
                        "pos": obj.get_final_position(),
                        "rot": obj.rot,  # Store base rotation; default offsets applied later
                        "attributes": obj.attributes
                    }
                    for obj in objs
                ]
                for room_id, objs in self.rooms_objects.items()
            }
        } 