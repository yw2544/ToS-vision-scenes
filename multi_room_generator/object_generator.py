"""
Object Generator Module
======================

Generates objects in rooms based on:
- Room size constraints
- Object placement optimization (visibility + occlusion)
- No duplicate objects across all rooms
- Agent placement only in the largest/main room
"""

from typing import List, Dict, Tuple, Optional, Set, Any

import numpy as np

from .room_analyzer import RoomAnalyzer, RoomInfo
from .object_generator_types import COLORS, PlacedObject, AgentInfo

# Import collinear validator
from .collinear_validator import CollinearValidator

from .object_generator_models import (
    _load_models_from_json,
    _map_color_name_to_value,
    get_models_by_category,
    select_objects_by_categories,
    _place_object_in_room_new,
    get_current_object_pool,
    _debug_min_center_gap,
)
from .object_generator_generation import (
    generate_all_rooms,
    _generate_objects_original_logic,
    _distribute_objects_fixed_counts,
    _distribute_objects_across_rooms,
    _place_objects_with_retries,
)
from .object_generator_placement_strict import (
    _try_place_single_object_strict,
    _try_place_object_at_integer_positions,
    _generate_room_objects_force,
    _try_place_object_relaxed,
    _overlaps_with_existing_relaxed,
)
from .object_generator_room_placement import (
    _generate_room_objects,
    _generate_room_objects_exact_count,
    _place_agent_in_room,
    _place_object_in_room,
)
from .object_generator_collision import (
    _overlaps_with_existing,
    _overlaps_with_existing_with_default,
    _check_8_neighbor_conflicts,
    _check_bounding_box_overlap,
    _check_center_distance,
    _final_center_from_candidate,
    _mark_object_occupied,
    _unmark_object_occupied,
    _get_object_bounds,
    _rotation_to_orientation,
)
from .object_generator_optimization import (
    _is_position_safe_from_collinearity,
    _optimize_room_positions,
    _is_position_safe_from_collinearity_during_optimization,
    _would_overlap_at_position,
)
from .object_generator_export import (
    get_all_objects,
    get_room_objects,
    get_agent,
    validate_category_uniqueness,
    export_summary,
)


class ObjectGenerator:
    """Generates objects in multiple rooms with optimization"""

    def __init__(self, room_analyzer: RoomAnalyzer, seed: int = 42, min_distance: float = 1.1,
                 builtin_models_path: Optional[str] = None,
                 custom_models_path: Optional[str] = None,
                 tolerance_width: float = 1.0):
        """
        Initialize object generator

        Args:
            room_analyzer: Analyzed room layout
            seed: Random seed for reproducible generation
            min_distance: Minimum distance between objects
            builtin_models_path: Path to models/builtin_models.json
            custom_models_path: Path to models/custom_models.json
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

        # Model loading from JSON files
        self.builtin_models_path = builtin_models_path
        self.custom_models_path = custom_models_path
        self.models_by_category: Dict[str, List[Dict]] = {}  # category -> [model1, model2, ...]

        # Collinear validation
        self.collinear_validator = CollinearValidator(tolerance_width=tolerance_width)

        self._load_models_from_json()

    def set_model_library(self, lib):
        """Set the TDW model library for bounds calculation"""
        self.model_lib = lib


# Attach implementations
ObjectGenerator._load_models_from_json = _load_models_from_json
ObjectGenerator._map_color_name_to_value = _map_color_name_to_value
ObjectGenerator.get_models_by_category = get_models_by_category
ObjectGenerator.select_objects_by_categories = select_objects_by_categories
ObjectGenerator._place_object_in_room_new = _place_object_in_room_new
ObjectGenerator.get_current_object_pool = get_current_object_pool
ObjectGenerator._debug_min_center_gap = _debug_min_center_gap

ObjectGenerator.generate_all_rooms = generate_all_rooms
ObjectGenerator._generate_objects_original_logic = _generate_objects_original_logic
ObjectGenerator._distribute_objects_fixed_counts = _distribute_objects_fixed_counts
ObjectGenerator._distribute_objects_across_rooms = _distribute_objects_across_rooms
ObjectGenerator._place_objects_with_retries = _place_objects_with_retries

ObjectGenerator._try_place_single_object_strict = _try_place_single_object_strict
ObjectGenerator._try_place_object_at_integer_positions = _try_place_object_at_integer_positions
ObjectGenerator._generate_room_objects_force = _generate_room_objects_force
ObjectGenerator._try_place_object_relaxed = _try_place_object_relaxed
ObjectGenerator._overlaps_with_existing_relaxed = _overlaps_with_existing_relaxed
ObjectGenerator._generate_room_objects = _generate_room_objects
ObjectGenerator._generate_room_objects_exact_count = _generate_room_objects_exact_count
ObjectGenerator._place_agent_in_room = _place_agent_in_room
ObjectGenerator._place_object_in_room = _place_object_in_room
ObjectGenerator._overlaps_with_existing = _overlaps_with_existing
ObjectGenerator._overlaps_with_existing_with_default = _overlaps_with_existing_with_default
ObjectGenerator._check_8_neighbor_conflicts = _check_8_neighbor_conflicts
ObjectGenerator._check_bounding_box_overlap = _check_bounding_box_overlap
ObjectGenerator._check_center_distance = _check_center_distance
ObjectGenerator._final_center_from_candidate = _final_center_from_candidate
ObjectGenerator._mark_object_occupied = _mark_object_occupied
ObjectGenerator._unmark_object_occupied = _unmark_object_occupied
ObjectGenerator._get_object_bounds = _get_object_bounds
ObjectGenerator._rotation_to_orientation = _rotation_to_orientation

ObjectGenerator._is_position_safe_from_collinearity = _is_position_safe_from_collinearity
ObjectGenerator._optimize_room_positions = _optimize_room_positions
ObjectGenerator._is_position_safe_from_collinearity_during_optimization = _is_position_safe_from_collinearity_during_optimization
ObjectGenerator._would_overlap_at_position = _would_overlap_at_position

ObjectGenerator.get_all_objects = get_all_objects
ObjectGenerator.get_room_objects = get_room_objects
ObjectGenerator.get_agent = get_agent
ObjectGenerator.validate_category_uniqueness = validate_category_uniqueness
ObjectGenerator.export_summary = export_summary
