#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Scene generation helpers for EnhancedMask2Scene.
Extracted from mask2scene_enhanced.py to reduce file length.
"""

import json
import math
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np

from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.add_ons.image_capture import ImageCapture
from tdw.librarian import ModelLibrarian
from tdw.librarian import HDRISkyboxLibrarian

from scene import mask2scene
from multi_room_generator import RoomAnalyzer, ObjectGenerator, DoorHandler
from validation.pre_render_validator import extract_scene_data_for_validation


def generate_complete_scene(self) -> bool:
    """
    Generate complete multi-room scene with objects, doors, and photography

    Returns:
        True if generation successful, False otherwise
    """
    try:
        print("[INFO] Starting complete multi-room scene generation...")

        # Step 1: Load and analyze mask
        if not self._load_and_analyze_mask():
            return False

        # Step 2: Process doors (before building scene to get colors)
        if not self._process_doors():
            return False

        # Step 3: Generate and validate objects FIRST (before TDW initialization)
        if not self._generate_and_validate_objects_with_retry():
            return False

        # Step 4: Generate base metadata (NEW - before TDW initialization)
        # This allows metadata to be available with object IDs and positions
        # before starting expensive TDW rendering
        if not self._generate_base_metadata():
            return False

        # Step 4.5: Generate task viewpoints from VAGEN tasks
        if not self._generate_task_viewpoints():
            print("[WARNING] Failed to generate task viewpoints, continuing without them...")

        # Step 5: Initialize TDW and build basic scene (only after validation passes)
        print("[INFO] ✅ Validation passed! Initializing TDW...")
        if not self._initialize_tdw_and_build_scene():
            return False

        # Step 5.5: Capture empty top-down before placing any objects/agent (512x512)
        empty_id_img = None
        try:
            empty_id_img = self._capture_topdown_empty()
        except Exception as e:
            print(f"[WARN] Failed to capture top_down_empty: {e}")

        # Step 5.6: Generate grid point mapping BEFORE placing objects (to avoid occlusion, 512x512)
        if self._should_save_topdown_map():
            try:
                print("[INFO] Generating grid point mapping before object placement (512x512)...")
                if empty_id_img is not None:
                    mask = mask2scene.load_mask(self.mask_path)
                    self._generate_topdown_pixel_map(empty_id_img, mask)
                    print("[INFO] ✅ Grid point mapping completed before object placement")
                else:
                    print("[WARN] Empty _id image not available, skipping grid mapping")
            except Exception as e:
                print(f"[WARN] Failed to generate grid point mapping before object placement: {e}")
        else:
            print("[INFO] save_topdown_map disabled; skipping grid mapping")

        # Step 5.7: Switch to 768x768 resolution before placing objects
        print("[INFO] Switching screen size to 768x768 for object placement and image capture...")
        self.controller.communicate([
            {"$type": "set_screen_size", "width": 768, "height": 768}
        ])
        self.controller.communicate([])

        # Step 6: Build TDW scene with validated objects (now at 768x768)
        if not self._build_tdw_scene_with_objects():
            return False

        # Step 7: Capture all images (regular)
        if not self._capture_all_images():
            return False

        # Step 7.5: Capture task viewpoint images
        if self.task_viewpoints:
            if not self._capture_task_viewpoint_images():
                print("[WARNING] Failed to capture task viewpoint images, continuing...")

        # Step 8: Update metadata with captured images (NEW - after image capture)
        # This only updates the 'images' field in the existing metadata file
        if not self._update_metadata_with_images():
            return False

        # Step 9: Generate orientation instruction image
        if not self._generate_orientation_instruction():
            print("[WARNING] Failed to generate orientation instruction, but continuing...")

        print(f"[SUCCESS] Complete multi-room scene generated successfully!")
        print(f"[INFO] Output directory: {self.output_dir.resolve()}")
        return True

    except Exception as e:
        print(f"[ERROR] Scene generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        self._cleanup()


def generate_and_validate_objects_with_retry(self) -> bool:
    """
    Generate objects and validate with RAGEN, retry if validation fails.

    Returns:
        True if objects generated and validated successfully
    """
    # Check if validation is explicitly enabled in config
    validation_config = self.config.get('text_based_validity', {})
    validation_enabled = validation_config.get('enabled', False)
    pre_render_validation = validation_config.get('pre_render_validation', False)

    if not validation_enabled or not pre_render_validation:
        # If validation is disabled, just generate objects normally
        print("[INFO] Pre-render validation disabled, generating objects without validation...")
        return self._generate_room_objects()

    # Validation is enabled - MUST use RAGEN
    if not self.pre_render_validator.is_available():
        print("[ERROR] RAGEN validation is enabled in config but RAGEN components are not available!")
        print("[ERROR] Please ensure RAGEN is properly installed and accessible.")
        return False

    print("[INFO] ✅ Pre-render validation ENABLED - will validate using RAGEN")

    max_retries = self.pre_render_validator.max_retries
    seed_increment = self.pre_render_validator.seed_increment
    original_seed = self.seed

    print(f"\n🔍 Pre-render validation enabled, max retries: {max_retries}")

    for attempt in range(max_retries):
        current_seed = original_seed + (attempt * seed_increment)
        self.seed = current_seed

        print(f"\n🎲 Attempt {attempt + 1}/{max_retries} with seed {current_seed}")

        # Generate objects for this attempt
        if not self._generate_room_objects():
            print(f"❌ Object generation failed on attempt {attempt + 1}")
            continue

        # Extract scene data for validation
        try:
            mask, objects_data, agent_data = extract_scene_data_for_validation(
                self.object_generator, self.room_analyzer
            )

            # Create random generator for this validation
            validation_rng = np.random.default_rng(current_seed + 12345)

            # Validate the scene
            is_valid, validation_summary = self.pre_render_validator.validate_scene_before_render(
                mask, objects_data, agent_data, validation_rng, attempt
            )

            if is_valid:
                print(f"✅ Validation PASSED on attempt {attempt + 1}!")
                print(f"🎯 Using final seed: {current_seed}")

                # Store validation results in metadata for later use
                if not hasattr(self, '_validation_metadata'):
                    self._validation_metadata = {}
                self._validation_metadata.update({
                    'pre_render_validation': validation_summary,
                    'final_seed': current_seed,
                    'attempts_needed': attempt + 1,
                    'original_seed': original_seed
                })

                return True
            else:
                print(f"❌ Validation FAILED on attempt {attempt + 1}")
                print(f"📝 Failed tasks: {validation_summary.get('failed_tasks', [])}")

                # Clear the current object generator to prepare for retry
                self.object_generator = None

        except Exception as e:
            print(f"❌ Validation error on attempt {attempt + 1}: {e}")
            self.object_generator = None
            continue

    print(f"\n❌ All {max_retries} attempts failed validation!")
    print("💡 Consider:")
    print("  - Adjusting task parameters in config")
    print("  - Increasing max_retries")
    print("  - Checking room layout constraints")

    return False


def load_and_analyze_mask(self) -> bool:
    """Load mask and analyze room structure"""
    try:
        print("[INFO] Loading and analyzing mask...")

        # Load mask
        mask = mask2scene.load_mask(self.mask_path)
        rows, cols = len(mask), len(mask[0])
        print(f"[INFO] Loaded mask: {rows}x{cols}")

        # Analyze rooms and doors
        self.room_analyzer = RoomAnalyzer(mask, self.cell_size)

        if not self.room_analyzer.rooms:
            print("[ERROR] No valid rooms found in mask")
            return False

        print(f"[INFO] Found {len(self.room_analyzer.rooms)} rooms and {len(self.room_analyzer.doors)} doors")

        # Print room analysis summary
        summary = self.room_analyzer.export_summary()
        print("[INFO] Room analysis summary:")
        for room_id, room_data in summary["rooms"].items():
            print(f"  Room {room_id}: area={room_data['area']}, max_objects={room_data['max_objects']}")

        # Validate fix_object_n if provided
        if self.fix_object_n is not None:
            self._validate_fix_object_n()

        return True

    except Exception as e:
        print(f"[ERROR] Failed to load and analyze mask: {e}")
        return False


def validate_fix_object_n(self):
    """Validate fix_object_n parameter against room configuration"""
    if not isinstance(self.fix_object_n, list):
        raise ValueError("fix_object_n must be a list")

    # Check if length matches number of rooms
    num_rooms = len(self.room_analyzer.rooms)
    if len(self.fix_object_n) != num_rooms:
        raise ValueError(f"fix_object_n length ({len(self.fix_object_n)}) must equal number of rooms ({num_rooms})")

    # Check room capacities
    room_ids = sorted(self.room_analyzer.rooms.keys())
    for i, room_id in enumerate(room_ids):
        requested_objects = self.fix_object_n[i]
        if not isinstance(requested_objects, int) or requested_objects < 0:
            raise ValueError(f"fix_object_n[{i}] must be a non-negative integer, got {requested_objects}")

        max_capacity = self.room_analyzer.get_max_objects_for_room(room_id)
        if requested_objects > max_capacity:
            raise ValueError(f"Room {room_id} capacity ({max_capacity}) is less than requested objects ({requested_objects})")

    print(f"[INFO] fix_object_n validation passed: {self.fix_object_n} for rooms {room_ids}")


def calculate_proportional_distribution(self) -> list:
    """Calculate proportional object distribution based on room areas"""
    print(f"[INFO] Calculating proportional object distribution for {self.total_objects} objects")

    # Get room areas
    room_ids = sorted(self.room_analyzer.rooms.keys())
    room_areas = []
    for room_id in room_ids:
        room_info = self.room_analyzer.rooms[room_id]
        area = room_info.area
        room_areas.append(area)

    print(f"[INFO] Room areas: {dict(zip(room_ids, room_areas))}")

    # Calculate total area
    total_area = sum(room_areas)

    # Calculate proportional objects (keeping at least 1 object per room)
    proportional_objects = []
    remaining_objects = self.total_objects

    # First, assign at least 1 object to each room
    base_objects = [1] * len(room_ids)
    remaining_objects -= len(room_ids)

    # Then distribute remaining objects proportionally
    if remaining_objects > 0:
        for i, area in enumerate(room_areas):
            additional = int(round((area / total_area) * remaining_objects))
            base_objects[i] += additional

    # Ensure total matches exactly (handle rounding errors)
    current_total = sum(base_objects)
    if current_total != self.total_objects:
        diff = self.total_objects - current_total
        # Adjust the largest room
        max_area_idx = room_areas.index(max(room_areas))
        base_objects[max_area_idx] += diff

    # Validate against room capacities
    for i, room_id in enumerate(room_ids):
        max_capacity = self.room_analyzer.get_max_objects_for_room(room_id)
        if base_objects[i] > max_capacity:
            print(f"[WARNING] Room {room_id} requested {base_objects[i]} objects but capacity is {max_capacity}")
            # Redistribute excess to other rooms
            excess = base_objects[i] - max_capacity
            base_objects[i] = max_capacity

            # Find rooms that can accommodate more objects
            for j, other_room_id in enumerate(room_ids):
                if i != j and excess > 0:
                    other_capacity = self.room_analyzer.get_max_objects_for_room(other_room_id)
                    available = other_capacity - base_objects[j]
                    if available > 0:
                        transfer = min(excess, available)
                        base_objects[j] += transfer
                        excess -= transfer

            if excess > 0:
                print(f"[WARNING] Could not redistribute {excess} objects - reducing total")

    proportional_objects = base_objects
    actual_total = sum(proportional_objects)

    print(f"[INFO] Proportional distribution: {dict(zip(room_ids, proportional_objects))}")
    print(f"[INFO] Total objects: {actual_total}/{self.total_objects}")

    return proportional_objects


def initialize_tdw_and_build_scene(self) -> bool:
    """Initialize TDW and build basic scene structure"""
    try:
        print("[INFO] Initializing TDW and building basic scene...")

        # Initialize TDW controller
        self.controller = Controller(launch_build=True, port=self.port)
        # original light setting
        # Initialize model library if not already done
        if self.model_lib is None:
            from tdw.librarian import ModelLibrarian
            self.model_lib = ModelLibrarian("models_core.json")
            print("[INFO] 🔧 Model library initialized during TDW setup")
        sky_lib = HDRISkyboxLibrarian()
        record = sky_lib.records[77]
        self.controller.communicate({"$type": "add_hdri_skybox",
            "name": record.name,
            "url": record.get_url(),
            # "exposure": record.exposure,
            "exposure": 0,
            # "initial_skybox_rotation": record.initial_skybox_rotation,
            "initial_skybox_rotation": 180,
            "sun_elevation": 60,
            "sun_initial_angle": 60,
            # "sun_intensity": record.sun_intensity
            "sun_intensity": 1
        })

        self.controller.communicate([
    {"$type": "set_render_quality", "render_quality": 4},  # Highest render quality
    {"$type": "set_shadow_strength", "strength": 0.8}      # Full shadow strength
])

        # # Adjust directional light rotation (pitch down 30 degrees for better indoor lighting)
        # self.controller.communicate({
        #     "$type": "rotate_directional_light_by",
        #     "angle": 30.0,
        #     "axis": "pitch",
        #     "index": 0
        # })

        # floor_visual_material = "parquet_wood_mahogany"
        # self.controller.communicate([self.controller.get_add_material(material_name=floor_visual_material),
        # {"$type": "set_floor_material",
        #  "name": floor_visual_material}])

        # Calculate scene dimensions based on mask size with proper offset
        mask = mask2scene.load_mask(self.mask_path)
        rows, cols = len(mask), len(mask[0])

        # Calculate scene bounds in world coordinates
        cell_size = self.cell_size
        offset_x = -(cols - 1) / 2.0 * cell_size
        offset_z = -(rows - 1) / 2.0 * cell_size

        # Scene bounds
        min_x = offset_x - cell_size  # Add margin
        max_x = offset_x + cols * cell_size + cell_size
        min_z = offset_z - cell_size
        max_z = offset_z + rows * cell_size + cell_size

        scene_width = int(max_x - min_x)
        scene_depth = int(max_z - min_z)
        print(f"[INFO] Calculated scene dimensions: {scene_width}x{scene_depth} (mask {rows}x{cols})")
        print(f"[INFO] World bounds: X[{min_x:.1f}, {max_x:.1f}], Z[{min_z:.1f}, {max_z:.1f}]")

        # Setup cameras
        self.main_cam = ThirdPersonCamera(
            avatar_id="main_cam",
            position={"x": 0, "y": 0.8, "z": 0},
            field_of_view=95
        )

        # Top-down camera (height should be sufficient to see entire scene)
        top_height = max(scene_width, scene_depth) * 0.9  # Adjust for better viewing angle
        self.top_cam = ThirdPersonCamera(
            avatar_id="top_down",
            position={"x": 0, "y": top_height, "z": 0},
            look_at={"x": 0, "y": 0, "z": 0},
            field_of_view=60
        )

        # Add cameras to controller (removed oblique camera)
        self.controller.add_ons.extend([self.main_cam, self.top_cam])

        # Setup image capture
        self.image_capture = ImageCapture(
            path=str(self.output_dir),
            avatar_ids=["main_cam", "top_down"],
            pass_masks=["_img", "_id"],
            png=True
        )
        self.controller.add_ons.append(self.image_capture)
        self.image_capture.set(frequency="never")

        # Build basic scene structure (walls, floors, doors)
        mask = mask2scene.load_mask(self.mask_path)

        # Create large room to push walls out of view
        large_room_w = max(scene_width + 10, 40)
        large_room_d = max(scene_depth + 10, 40)

        room_commands = [
            TDWUtils.create_empty_room(large_room_w, large_room_d),
            {"$type": "set_screen_size", "width": 512, "height": 512}
        ]
        self.controller.communicate(room_commands)

        # Create a position-aware door color mapping strategy
        # We'll use a custom approach: monkey-patch the _add_door_visual function
        print("[INFO] Setting up position-aware door coloring...")

        # Store door positions and colors for lookup during door creation
        self._door_position_colors = {}
        if hasattr(self, 'door_handler') and self.door_handler:
            for door_id, proc in self.door_handler.processed_doors.items():
                door_center = proc.door_info.center
                # Convert world coordinates back to mask coordinates for lookup
                mask_data = mask2scene.load_mask(self.mask_path)
                rows, cols = len(mask_data), len(mask_data[0])
                offset_x = -(cols - 1) / 2.0 * self.cell_size
                offset_z = -(rows - 1) / 2.0 * self.cell_size
                mask_col = round((door_center[0] - offset_x) / self.cell_size)
                mask_row = round((door_center[1] - offset_z) / self.cell_size)

                # Store position -> color mapping
                self._door_position_colors[(mask_row, mask_col)] = proc.color
                print(f"[INFO] Position ({mask_row}, {mask_col}) -> {proc.color_name}")

        # Patch the door visual creation to use position-specific colors
        original_add_door_visual = mask2scene._add_door_visual

        def position_aware_add_door_visual(ctrl, *, x, z, wall_h, wall_t, door_w, horizontal, color=None):
            # Find the position in mask coordinates that corresponds to this x, z
            mask_data = mask2scene.load_mask(self.mask_path)
            rows, cols = len(mask_data), len(mask_data[0])
            offset_x = -(cols - 1) / 2.0 * self.cell_size
            offset_z = -(rows - 1) / 2.0 * self.cell_size

            # Convert world coordinates back to mask coordinates
            mask_col = round((x - offset_x) / self.cell_size)
            mask_row = round((z - offset_z) / self.cell_size)

            # Look up the color for this specific position
            position_key = (mask_row, mask_col)
            if position_key in self._door_position_colors:
                color = self._door_position_colors[position_key]
                print(f"[INFO] Using custom color for door at position ({mask_row}, {mask_col})")

            # Call the original function with the correct color
            return original_add_door_visual(ctrl, x=x, z=z, wall_h=wall_h, wall_t=wall_t,
                                           door_w=door_w, horizontal=horizontal, color=color)

        # Apply the patch
        mask2scene._add_door_visual = position_aware_add_door_visual

        try:
            # Build walls, floors, and doors using original build_scene function
            # Get door/filler tracking info
            door_ids, filler_ids, door_pos, filler_pos = mask2scene.build_scene(
                ctrl=self.controller,
                mask=mask,
                cell=self.cell_size,
                wall_t=self.wall_thickness,
                wall_h=self.wall_height,
                door_w=self.door_width,
                door_colors=None  # Colors handled by our patch
            )

            # Store door/filler tracking info
            self.door_object_ids = door_ids
            self.filler_object_ids = filler_ids
            self.door_positions = door_pos
            self.filler_positions = filler_pos

            print(f"[INFO] Tracked {len(self.door_object_ids)} doors and {len(self.filler_object_ids)} fillers for teleportation")

        finally:
            # Restore the original function
            mask2scene._add_door_visual = original_add_door_visual

        print("[INFO] Basic scene structure built successfully")
        return True

    except Exception as e:
        print(f"[ERROR] Failed to initialize TDW and build scene: {e}")
        return False


def generate_room_objects(self) -> bool:
    """Generate objects for all rooms"""
    try:
        print("[INFO] Generating objects for all rooms...")

        # Initialize object generator with new JSON-based model loading
        # Use paths from constructor parameters, with fallback to default locations
        repo_root = Path(__file__).resolve().parent.parent
        if self.builtin_models_path is None:
            builtin_models_path = repo_root / "models" / "builtin_models.json"
            builtin_models_path_str = str(builtin_models_path) if builtin_models_path.exists() else None
        else:
            builtin_models_path = Path(self.builtin_models_path)
            builtin_models_path_str = str(builtin_models_path) if builtin_models_path.exists() else None

        if self.custom_models_path is None:
            custom_models_path = repo_root / "models" / "custom_models.json"
            custom_models_path_str = str(custom_models_path) if custom_models_path.exists() else None
        else:
            custom_models_path = Path(self.custom_models_path)
            custom_models_path_str = str(custom_models_path) if custom_models_path.exists() else None

        print(f"[INFO] Using builtin models path: {builtin_models_path_str}")
        print(f"[INFO] Using custom models path: {custom_models_path_str}")

        # Initialize model library for object generation (before TDW initialization)
        if self.model_lib is None:
            from tdw.librarian import ModelLibrarian
            self.model_lib = ModelLibrarian("models_core.json")
            print("[INFO] 🔧 Model library initialized for object generation")

        # Read distance constraints and collinearity settings from config
        distance_config = self.config.get('object_generation', {}).get('distance_constraints', {})
        min_distance = distance_config.get('min_distance', 1.1)  # Default 1.1m

        collinear_config = self.config.get('object_generation', {}).get('collinear_detection', {})
        tolerance_width = collinear_config.get('tolerance_width', 1.0)  # Default 1.0m

        self.object_generator = ObjectGenerator(
            self.room_analyzer,
            seed=self.seed,
            min_distance=min_distance,  # Min distance from config
            builtin_models_path=builtin_models_path_str,
            custom_models_path=custom_models_path_str,
            tolerance_width=tolerance_width  # Collinearity tolerance from config
        )
        print(f"[INFO] Min object distance set to: {self.object_generator.min_distance} m")
        print(f"[INFO] Collinearity tolerance set to: {tolerance_width} m")
        self.object_generator.set_model_library(self.model_lib)

        # Generate objects for all rooms based on selected mode
        if self.object_mode == "fixed":
            success = self.object_generator.generate_all_rooms(main_room_id=1, fix_object_n=self.fix_object_n)
        elif self.object_mode == "total":
            success = self.object_generator.generate_all_rooms(main_room_id=1, total_objects=self.total_objects)
        elif self.object_mode == "proportional":
            # Calculate proportional distribution based on room areas
            proportional_objects = self._calculate_proportional_distribution()
            success = self.object_generator.generate_all_rooms(main_room_id=1, fix_object_n=proportional_objects)
        else:
            raise ValueError(f"Unknown object_mode: {self.object_mode}")
        if not success:
            print("[ERROR] Failed to generate room objects")
            return False

        # Print generation summary
        summary = self.object_generator.export_summary()
        print(f"[INFO] Generated {summary['total_objects']} objects across {summary['rooms_with_objects']} rooms")
        print(f"[INFO] Used categories: {summary['used_categories']}")

        agent = self.object_generator.get_agent()
        if agent:
            print(f"[INFO] Agent placed in room {agent.room_id} at ({agent.pos['x']:.2f}, {agent.pos['z']:.2f})")

        return True

    except Exception as e:
        import traceback
        print(f"[ERROR] Failed to generate room objects: {e}")
        print(f"[ERROR] Detailed error: {traceback.format_exc()}")
        return False


def process_doors(self) -> bool:
    """Process doors: assign colors and setup camera positions"""
    try:
        print("[INFO] Processing doors...")

        # Initialize door handler
        self.door_handler = DoorHandler(self.room_analyzer, seed=self.seed)

        # Validate door positions
        if not self.door_handler.validate_door_positions():
            print("[WARN] Some door camera positions may be problematic")

        # Print door summary
        print(self.door_handler.get_door_summary())

        return True

    except Exception as e:
        print(f"[ERROR] Failed to process doors: {e}")
        return False


def build_tdw_scene_with_objects(self) -> bool:
    """Add generated objects to TDW scene"""
    try:
        print("[INFO] Adding generated objects to TDW scene...")

        all_objects = self.object_generator.get_all_objects()
        print(f"[INFO] Total objects to add: {len(all_objects)}")

        # Add all objects to scene
        for i, obj in enumerate(all_objects):
            obj_final_pos = obj.get_final_position()
            print(f"[INFO] Adding object {i+1}/{len(all_objects)}: {obj.name} at ({obj_final_pos['x']:.1f}, {obj_final_pos['y']:.1f}, {obj_final_pos['z']:.1f})")
            print(f"[DEBUG] Object attributes: is_custom_model={obj.is_custom_model}, custom_config={obj.custom_config is not None}")

            # Choose loading path based on custom model flag
            if obj.is_custom_model and obj.custom_config:
                # Custom model: load via URL
                custom_config = obj.custom_config
                try:
                    from tdw.librarian import ModelRecord
                    # Load record.json path
                    import json
                    # Get record path from dict
                    record_file = custom_config.get("record")
                    if not record_file:
                        print(f"[ERROR] Custom model {obj.name} has no record path")
                        continue

                    print(f"[DEBUG] Trying to load record file: {record_file}")
                    print(f"[DEBUG] Model name: {obj.name}, model: {obj.model}")

                    # record_file points to record.json
                    from pathlib import Path
                    record_path = Path(record_file)

                    # Load record.json
                    with open(record_path, 'r', encoding='utf-8') as f:
                        record_data = json.load(f)

                    # Each custom model uses ModelRecord
                    print(f"[DEBUG] Creating ModelRecord for {obj.name}...")
                    object_record = ModelRecord(record_data)
                    print(f"[DEBUG] ModelRecord result: {object_record}")

                    if object_record is None:
                        print(f"[ERROR] Failed to create ModelRecord for {obj.name}")
                        continue

                    print(f"[DEBUG] Calling object_record.get_url()...")
                    try:
                        model_name = object_record.name
                        model_url = object_record.get_url()
                        print(f"[DEBUG] get_url() success: {model_url}")
                    except Exception as get_url_error:
                        print(f"[ERROR] object_record.get_url() failed: {get_url_error}")
                        import traceback
                        print(f"[ERROR] Detailed error: {traceback.format_exc()}")
                        continue

                    if model_url is None:
                        print(f"[ERROR] ModelRecord.get_url() returned None for {obj.name}")
                        continue

                    print(f"[DEBUG] Model name: {model_name}")
                    print(f"[DEBUG] URL: {model_url}")

                    if not model_url:
                        print(f"[ERROR] Failed to get model URL: {obj.name}")
                        continue

                    # Use get_final_position/get_final_rotation to apply default offsets
                    final_pos = obj.get_final_position()
                    final_rot = obj.get_final_rotation()

                    self.controller.communicate([{
                        "$type": "add_object",
                        "name": model_name,
                        "url": model_url,
                        "scale_factor": obj.scale * self.overall_scale,
                        "position": {"x": final_pos["x"], "y": final_pos["y"]+1, "z": final_pos["z"]},
                        "rotation": {"x": final_rot["x"], "y": final_rot["y"]+1, "z": final_rot["z"]},
                        "category": "misc",
                        "id": obj.object_id
                    },{
                        "$type": "set_kinematic_state",
                        "id": obj.object_id,
                        "is_kinematic": True,  # Let object settle naturally
                        "use_gravity": False
                    }])
                    print(f"[INFO] Custom model loaded: {obj.name} (URL: {model_url}), position: {final_pos}, rotation: {final_rot}")

                    # Set physics for custom models
                    self.controller.communicate([{
                        "$type": "set_mass",
                        "id": obj.object_id,
                        "mass": 1.0  # Reasonable mass
                    }])

                    print(f"[INFO] Custom model loaded: {obj.name} (URL: {model_url})")
                except Exception as e:
                    import traceback
                    print(f"[ERROR] Failed to load custom model {obj.name}: {e}")
                    print(f"[ERROR] Detailed error: {traceback.format_exc()}")
                    continue
            else:
                # Library model: use existing loading path
                # Use get_final_position/get_final_rotation for consistency
                final_pos = obj.get_final_position()
                final_rot = obj.get_final_rotation()

                self.controller.communicate([
                    self.controller.get_add_object(
                        model_name=obj.model,
                        position=final_pos,
                        rotation=final_rot,
                        object_id=obj.object_id,
                        library="models_core.json"
                    )
                ])
                print(f"[INFO] Library model loaded: {obj.model} (ID: {obj.object_id}), position: {final_pos}, rotation: {final_rot}")

                # Apply scale if needed (multiply by overall_scale)
                final_scale = obj.scale * self.overall_scale
                if final_scale != 1.0:
                    self.controller.communicate([{
                        "$type": "scale_object",
                        "id": obj.object_id,
                        "scale_factor": {
                            "x": final_scale,
                            "y": final_scale,
                            "z": final_scale
                        }
                    }])

                # Set physics for library models
                self.controller.communicate([{
                    "$type": "set_mass",
                    "id": obj.object_id,
                    "mass": 1.0  # Reasonable mass
                }])

            # Apply color if specified
            if obj.color:
                if obj.is_custom_model and obj.custom_config and obj.custom_config.get("color"):
                    # Custom model colors come from config
                    color_values = obj.custom_config.get("color")
                else:
                    # Library model colors come from COLORS
                    from multi_room_generator.object_generator import COLORS
                    color_values = COLORS[obj.color]

                self.controller.communicate([{
                    "$type": "set_color",
                    "id": obj.object_id,
                    "color": color_values
                }])

            # CRITICAL: Enable physics and gravity for each object initially
            self.controller.communicate([{
                "$type": "set_kinematic_state",
                "id": obj.object_id,
                "is_kinematic": True,  # Enable physics
                "use_gravity": False     # Enable gravity
            }])

            print(f"[DEBUG] Enabled physics and gravity for object {obj.object_id}")

        # Wait for all objects to settle with gravity initially
        print("[INFO] Waiting for all objects to settle with gravity...")
        import time
        time.sleep(self.physics_settle_time)  # Use configurable settle time

        # Now set all objects to kinematic to prevent movement during capture
        print("[INFO] Setting all objects to kinematic state for stable capture...")
        kinematic_commands = []
        for obj in all_objects:
            kinematic_commands.append({
                "$type": "set_kinematic_state",
                "id": obj.object_id,
                "is_kinematic": True,  # Disable physics during capture
                "use_gravity": False   # Disable gravity during capture
            })

        # Send all kinematic commands at once for efficiency
        self.controller.communicate(kinematic_commands)
        print(f"[INFO] Set {len(all_objects)} objects to kinematic state")

        # Door colors are now applied during scene construction
        print("[INFO] Door colors applied during scene construction")

        print(f"[INFO] Added {len(self.object_generator.get_all_objects())} objects to TDW scene")
        return True

    except Exception as e:
        print(f"[ERROR] Failed to build TDW scene with objects: {e}")
        return False


def apply_door_colors(self):
    """Apply colors to doors by finding them in the scene"""
    try:
        print("[INFO] Applying door colors...")

        # Wait a moment for objects to be created
        import time
        time.sleep(0.2)

        # Get all objects in the scene
        resp = self.controller.communicate({"$type": "send_objects"})
        if not resp or len(resp) == 0:
            print("[WARN] No response from send_objects")
            return

        # Check if response contains the expected data structure
        try:
            # resp[0] should contain the objects data
            objects_data = resp[0]

            # Check if it's a dictionary with objects key
            if not hasattr(objects_data, 'get') or "objects" not in objects_data:
                print(f"[WARN] Unexpected objects_data structure: {type(objects_data)}")
                return

            objects_list = objects_data["objects"]
            print(f"[INFO] Found {len(objects_list)} objects in scene")

        except (IndexError, TypeError, AttributeError) as e:
            print(f"[WARN] Failed to parse objects data: {e}")
            return

        # Get door positions from door handler
        door_positions = {}
        for door_id, processed_door in self.door_handler.processed_doors.items():
            door_center = processed_door.door_info.center
            door_positions[door_id] = {
                "position": door_center,
                "color": processed_door.color,
                "color_name": processed_door.color_name
            }
            print(f"[INFO] Door {door_id} at ({door_center[0]:.2f}, {door_center[1]:.2f}) - {processed_door.color_name}")

        if not door_positions:
            print("[WARN] No doors to color")
            return

        # Find and color doors - look for prim_cube objects near door positions
        colored_doors = 0
        colored_objects = set()  # Track which objects have been colored

        # Group door positions by their 2D coordinates to handle multiple doors at same position
        position_to_doors = {}
        for door_id, door_data in door_positions.items():
            pos_key = (round(door_data["position"][0], 1), round(door_data["position"][1], 1))
            if pos_key not in position_to_doors:
                position_to_doors[pos_key] = []
            position_to_doors[pos_key].append((door_id, door_data))

        for obj in objects_list:
            try:
                obj_pos = obj["position"]
                obj_id = obj["id"]
                obj_name = obj.get("name", "unknown")

                # Skip if already colored
                if obj_id in colored_objects:
                    continue

                # Only consider prim_cube objects (doors are made of cubes)
                if "prim_cube" in obj_name.lower():
                    # Check if this object is close to any door position
                    best_match = None
                    best_distance = float('inf')

                    for door_id, door_data in door_positions.items():
                        door_pos = door_data["position"]
                        distance = ((obj_pos["x"] - door_pos[0])**2 + (obj_pos["z"] - door_pos[1])**2)**0.5

                        # If object is close to door center (within 2 units) and at door height
                        if distance < 2.0 and 0.5 < obj_pos["y"] < 2.5:
                            if distance < best_distance:
                                best_distance = distance
                                best_match = (door_id, door_data)

                    # Apply color for the closest door match
                    if best_match:
                        door_id, door_data = best_match
                        try:
                            self.controller.communicate([{
                                "$type": "set_color",
                                "id": obj_id,
                                "color": door_data["color"]
                            }])
                            colored_doors += 1
                            colored_objects.add(obj_id)
                            print(f"[INFO] Colored door {door_id} (object {obj_id}) with {door_data['color_name']} at distance {best_distance:.2f}")
                        except Exception as color_error:
                            print(f"[WARN] Failed to color object {obj_id}: {color_error}")

            except Exception as obj_error:
                print(f"[WARN] Error processing object: {obj_error}")
                continue

        print(f"[INFO] Successfully applied colors to {colored_doors} door objects")

    except Exception as e:
        print(f"[WARN] Failed to apply door colors: {e}")
        # Don't print full traceback to avoid spam, just the error message


def cleanup(self):
    """Clean up TDW controller"""
    if self.controller is not None:
        try:
            self.controller.communicate({"$type": "terminate"})
            print("[INFO] TDW controller terminated")
        except:
            pass  # Ignore cleanup errors
