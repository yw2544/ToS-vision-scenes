#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Enhanced Mask -> TDW Multi-Room Scene Generator
==============================================

Generates complete multi-room environments with:
- Room-specific object placement (0-4 objects per room based on size)
- Agent placement in largest room only
- Door handling with special colors and 4-direction photography
- Comprehensive metadata for all rooms, objects, and doors
- No duplicate objects across all rooms

Enhanced features:
- Multi-room object generation
- Door photography system with cameras at door center coordinates
- Room-specific camera positions
- Comprehensive metadata management
- Spatial relationship tracking
- Annotated top-down view with segmentation-based labeling
- Red dashed direction rays in all perspective views
- Door cameras positioned at exact door center coordinates
"""

import argparse, json, sys, yaml
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# Ensure repo root is on sys.path for script execution
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from validation.pre_render_validator import PreRenderValidator

# Extracted helpers
from scene.mask2scene_enhanced_topdown import (
    capture_topdown_empty,
    build_door_segmentation_map_topdown,
    generate_topdown_pixel_map,
)
from scene.mask2scene_enhanced_capture import (
    find_doors_at_position,
    temporarily_hide_items_for_camera,
    restore_hidden_items,
    capture_all_images,
    capture_task_viewpoint_images,
    capture_regular_task_images,
    capture_false_belief_images,
    capture_view_with_rays,
)
from scene.mask2scene_enhanced_visibility import (
    get_object_bounds,
    is_object_bbox_in_view,
    is_object_center_in_view,
    compute_visibility_ratio,
    calculate_object_ratios,
    create_direction_ray,
)
from scene.mask2scene_enhanced_metadata import (
    generate_base_metadata,
    update_metadata_with_images,
    generate_task_viewpoints_from_metadata,
    generate_orientation_instruction,
    find_object_id_by_name,
)
from scene.mask2scene_enhanced_scene import (
    generate_complete_scene,
    generate_and_validate_objects_with_retry,
    load_and_analyze_mask,
    validate_fix_object_n,
    calculate_proportional_distribution,
    initialize_tdw_and_build_scene,
    generate_room_objects,
    process_doors,
    build_tdw_scene_with_objects,
    apply_door_colors,
    cleanup,
)

class EnhancedMask2Scene:
    """Enhanced mask2scene with multi-room object generation capabilities"""
    
    def _generate_door_id_from_coordinates(self, door_center: List[float], door_id: int) -> int:
        """
        Generate unique door ID based on coordinates and original door_id
        Format: 20000 + (original_door_id * 100) + position_hash
        Ensures IDs are globally unique and above regular object IDs
        
        Args:
            door_center: [x, z] coordinates of door center
            door_id: Original door ID from mask analysis
            
        Returns:
            Unique integer ID for the door
        """
        # Create position-based hash for uniqueness
        position_hash = abs(hash((round(door_center[0], 2), round(door_center[1], 2)))) % 100
        
        # Format: 20000 (base) + (door_id * 100) + position_hash
        # This ensures global uniqueness even for doors with same mask ID
        door_object_id = 20000 + (door_id % 100) * 100 + position_hash
        
        # Additional safeguard: ensure it's unique in our scene
        # Keep incrementing until we find a unique ID
        used_ids = getattr(self, '_used_door_ids', set())
        while door_object_id in used_ids:
            door_object_id += 1
        
        if not hasattr(self, '_used_door_ids'):
            self._used_door_ids = set()
        self._used_door_ids.add(door_object_id)
        
        return door_object_id
    
    def _find_doors_at_position(self, camera_pos: Dict[str, float], tolerance: float = 0.1) -> Tuple[List[int], List[int]]:
        """
        Find door and filler object IDs at the given camera position.

        Args:
            camera_pos: camera position {"x": x, "y": y, "z": z}
            tolerance: position matching tolerance

        Returns:
            (door_ids_at_position, filler_ids_at_position) tuple
        """
        return find_doors_at_position(self, camera_pos, tolerance)
    
    def _temporarily_hide_items_for_camera(self, position: Dict[str, float], tolerance: float = 0.5):
        """Temporarily hide objects/doors/fillers located at camera position."""
        return temporarily_hide_items_for_camera(self, position, tolerance)
    
    def _restore_hidden_items(self, hidden_items):
        """Restore previously hidden objects/doors/fillers."""
        return restore_hidden_items(self, hidden_items)
    
    def __init__(self, mask_path: str, output_dir: str, cell_size: float = 1.0,
                 wall_thickness: float = 0.15, wall_height: float = 3.0,
                 door_width: float = 0.8, seed: int = 42, total_objects: int = None, port: int = 1071, x_offset=0.0, z_offset=0.0, with_ray: bool = False, overall_scale: float = 1.0, enable_gravity_fix: bool = True, physics_settle_time: float = 0.2, fix_object_n: list = None, object_mode: str = "total", proportional_to_area: bool = False, 
                 # New model system parameters
                 builtin_models_path: str = None, custom_models_path: str = None,
                 # Validation configuration
                 config: dict = None,
                 # Run seed (for metadata)
                 run_seed: int = None,
                 # Legacy parameters (deprecated)
                 custom_models_config: str = None, use_custom_models: bool = True):
        """
        Initialize enhanced mask2scene generator
        
        Args:
            mask_path: Path to mask file
            output_dir: Output directory for all generated content
            cell_size: Size of each mask cell in world units
            wall_thickness: Wall thickness
            wall_height: Wall height
            door_width: Door width
            seed: Random seed for object generation
            total_objects: Total number of objects to generate across all rooms (mutually exclusive with fix_object_n)
            port: TDW port
            overall_scale: Overall scale multiplier for all objects (doors excluded)
            enable_gravity_fix: Enable physics-based gravity fix for teleported objects
            physics_settle_time: Time to wait for physics to settle after object restoration
            fix_object_n: List specifying number of objects for each room (mutually exclusive with total_objects)
        """
        # Validate object distribution mode and parameters
        if object_mode not in ["fixed", "total", "proportional"]:
            raise ValueError(f"object_mode must be one of 'fixed', 'total', or 'proportional', got '{object_mode}'")
        
        if total_objects is None:
            raise ValueError("total_objects must be provided for all modes")
        
        if object_mode == "fixed":
            if fix_object_n is None:
                raise ValueError("fix_object_n must be provided when object_mode='fixed'")
            if sum(fix_object_n) != total_objects:
                raise ValueError(f"fix_object_n must sum to total_objects ({total_objects}), got sum={sum(fix_object_n)}")
        elif object_mode in ["total", "proportional"]:
            if fix_object_n is not None:
                print(f"[WARNING] fix_object_n is ignored when object_mode='{object_mode}'")
        
        self.mask_path = Path(mask_path)
        self.output_dir = Path(output_dir)
        self.cell_size = cell_size
        self.wall_thickness = wall_thickness
        self.wall_height = wall_height
        self.door_width = door_width
        self.seed = seed
        self.run_seed = run_seed  # Run seed for metadata
        self.total_objects = total_objects
        self.fix_object_n = fix_object_n
        self.object_mode = object_mode
        self.proportional_to_area = proportional_to_area
        self.port = port
        self.x_offset = x_offset
        self.z_offset = z_offset
        self.with_ray = with_ray
        self.overall_scale = overall_scale
        self.enable_gravity_fix = enable_gravity_fix
        self.physics_settle_time = physics_settle_time
        # New model system paths
        self.builtin_models_path = builtin_models_path
        self.custom_models_path = custom_models_path
        # Legacy settings (deprecated)
        self.custom_models_config = custom_models_config
        self.use_custom_models = use_custom_models
        # Scene parameters
        self.scene_params = {
            "mask_path": str(mask_path),
            "cell_size": cell_size,
            "wall_thickness": wall_thickness,
            "wall_height": wall_height,
            "door_width": door_width,
            "seed": seed
        }
        
        # TDW components
        self.controller = None
        self.main_cam = None
        self.top_cam = None
        self.image_capture = None
        self.model_lib = None
        
        # Multi-room components
        self.room_analyzer = None
        self.object_generator = None
        self.door_handler = None
        
        # Image metadata tracking
        self.captured_images = []
        self.camera_capture = None
        self.metadata_manager = None
        
        # Orientation instruction generator
        self.orientation_instruction_generator = None
        
        # Task viewpoints (generated from VAGEN tasks)
        self.task_viewpoints = []
        
        # Pre-render validation setup
        self.config = config or {}
        self.pre_render_validator = PreRenderValidator(self.config)
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"[INFO] Enhanced Mask2Scene initialized")
        print(f"[INFO] Input: {self.mask_path}")
        print(f"[INFO] Output: {self.output_dir}")

    def _should_save_topdown_map(self) -> bool:
        output_cfg = (self.config or {}).get("output", {})
        return bool(output_cfg.get("save_topdown_map", False))
    
    def generate_complete_scene(self) -> bool:
        """
        Generate complete multi-room scene with objects, doors, and photography
        
        Returns:
            True if generation successful, False otherwise
        """
        return generate_complete_scene(self)
    
    def _generate_and_validate_objects_with_retry(self) -> bool:
        """
        Generate objects and validate with RAGEN, retry if validation fails.
        
        Returns:
            True if objects generated and validated successfully
        """
        return generate_and_validate_objects_with_retry(self)
    

    def _load_and_analyze_mask(self) -> bool:
        """Load mask and analyze room structure"""
        return load_and_analyze_mask(self)
    
    def _validate_fix_object_n(self):
        """Validate fix_object_n parameter against room configuration"""
        return validate_fix_object_n(self)
    
    def _calculate_proportional_distribution(self) -> list:
        """Calculate proportional object distribution based on room areas"""
        return calculate_proportional_distribution(self)
    
    def _initialize_tdw_and_build_scene(self) -> bool:
        """Initialize TDW and build basic scene structure"""
        return initialize_tdw_and_build_scene(self)
    
    def _generate_room_objects(self) -> bool:
        """Generate objects for all rooms"""
        return generate_room_objects(self)
    
    def _process_doors(self) -> bool:
        """Process doors: assign colors and setup camera positions"""
        return process_doors(self)
    
    def _build_tdw_scene_with_objects(self) -> bool:
        """Add generated objects to TDW scene"""
        return build_tdw_scene_with_objects(self)
    
    def _apply_door_colors(self):
        """Apply colors to doors by finding them in the scene"""
        return apply_door_colors(self)
    
    def _capture_all_images(self) -> bool:
        """Capture all required images including annotated top-down view"""
        return capture_all_images(self)
    
    def _generate_base_metadata(self) -> bool:
        """
        Generate base metadata (all fields except images) before TDW initialization.
        This allows metadata to be available before starting TDW rendering.
        """
        return generate_base_metadata(self)
    
    def _update_metadata_with_images(self) -> bool:
        """
        Update metadata with captured images after TDW rendering completes.
        This only modifies the 'images' field in the existing metadata file.
        """
        return update_metadata_with_images(self)
    
    def _generate_task_viewpoints(self) -> bool:
        """Generate task viewpoints from VAGEN evaluation tasks.
        
        This method reads the base metadata and generates additional viewpoints
        for backward tasks (bwd_nav, bwd_loc, bwd_pov) and false_belief task.
        It also updates the metadata with camera configurations for these viewpoints.
        
        Returns:
            True if viewpoints generated successfully, False otherwise
        """
        return generate_task_viewpoints_from_metadata(self)
    
    def _capture_task_viewpoint_images(self) -> bool:
        """Capture images from task viewpoints.
        
        This method captures images for bwd_nav, bwd_loc, bwd_pov tasks normally,
        and handles false_belief tasks specially (rotate object, capture, restore).
        
        Returns:
            True if all images captured successfully, False otherwise
        """
        return capture_task_viewpoint_images(self)
    
    def _capture_regular_task_images(self, viewpoints: List[Dict]) -> None:
        """Capture images for regular task viewpoints (bwd_nav, bwd_loc, bwd_pov).
        
        Args:
            viewpoints: List of viewpoint dicts
        """
        return capture_regular_task_images(self, viewpoints)
    
    def _capture_false_belief_images(self, viewpoints: List[Dict]) -> None:
        """Capture images for false_belief task with object rotation.
        
        For each false_belief viewpoint:
        1. Rotate the specified object
        2. Capture image
        3. Restore object to original rotation
        
        Args:
            viewpoints: List of false_belief viewpoint dicts
        """
        return capture_false_belief_images(self, viewpoints)
    
    def _find_object_id_by_name(self, object_name: str) -> Optional[int]:
        """Find TDW object ID by object name from metadata.
        
        Supports fuzzy matching (spaces vs underscores).
        
        Args:
            object_name: Name of the object (e.g., 'chair', 'table', 'fire extinguisher')
            
        Returns:
            Object ID or None if not found
        """
        return find_object_id_by_name(self, object_name)
    
    def _generate_orientation_instruction(self) -> bool:
        """Generate orientation instruction image from metadata"""
        return generate_orientation_instruction(self)
    
    def _capture_view_with_rays(self, pos, deg, tag, label, room=None):
        """Capture image with direction ray (adapted from pipeline_partial_label_only.py)"""
        return capture_view_with_rays(self, pos, deg, tag, label, room)

    def get_object_bounds(self, model_name: str, rotation: dict = {"x": 0, "y": 0, "z": 0}, scale: float = 1.0):
        """Get object bounds from TDW ModelLibrarian with scale applied"""
        return get_object_bounds(self, model_name, rotation, scale)

    def is_object_bbox_in_view(self, obj, cam_pos: Dict[str, float], look_direction: Tuple[float, float], fov_deg: float = 90) -> bool:
        """Check if object bounding box is within camera field of view using geometric intersection"""
        return is_object_bbox_in_view(self, obj, cam_pos, look_direction, fov_deg)
    
    def is_object_center_in_view(self, obj, cam_pos: Dict[str, float], look_direction: Tuple[float, float], fov_deg: float = 90) -> bool:
        """Check if object center is within camera field of view"""
        return is_object_center_in_view(self, obj, cam_pos, look_direction, fov_deg)

    def compute_visibility_ratio(self, target_obj, cam_pos, look_direction, fov_deg=90, samples=64):
        """Compute visibility ratio based on top-down view geometry using bounding box intersection"""
        return compute_visibility_ratio(self, target_obj, cam_pos, look_direction, fov_deg, samples)

    def _calculate_object_ratios(self, camera_pos, camera_deg, cam_id, room_filter=None):
        """
        Calculate object visibility ratios using proper bounding box intersection
        For door cameras, room_filter specifies which room's objects to include
        Returns list of object ratio dicts with object_id, model, visibility_ratio, occlusion_ratio
        """
        return calculate_object_ratios(self, camera_pos, camera_deg, cam_id, room_filter)

    def _create_direction_ray(self, pos, deg):
        """
        Create a dashed red direction ray on the ground pointing in the specified direction.
        Return a list of all segment IDs so they can be destroyed later.
        Based exactly on pipeline_partial_label_only.py implementation
        """
        return create_direction_ray(self, pos, deg)

    def _cleanup(self):
        """Clean up TDW controller"""
        return cleanup(self)

    def _capture_topdown_empty(self):
        """Capture top-down image of the empty scene (no objects/agent placed)."""
        return capture_topdown_empty(self)

    def _build_door_segmentation_map_topdown(self):
        """Use segmentation to locate door centers in top-down view."""
        return build_door_segmentation_map_topdown(self)

    def _generate_topdown_pixel_map(self, id_img, mask, annotated_path=None):
        """Map every integer grid cell to its pixel center in top-down view and store into metadata."""
        return generate_topdown_pixel_map(self, id_img, mask, annotated_path)

def main():
    """Main function with command-line interface"""
    parser = argparse.ArgumentParser(description="Enhanced Multi-Room Scene Generator")
    parser.add_argument("--mask_path", type=str, required=True, 
                       help="Path to mask file")
    parser.add_argument("--output", type=str, required=True,
                       help="Output directory")
    parser.add_argument("--cell_size", type=float, default=1.0,
                       help="Cell size in world units")
    parser.add_argument("--wall_thickness", type=float, default=0.15,
                       help="Wall thickness")
    parser.add_argument("--wall_height", type=float, default=3.0,
                       help="Wall height")
    parser.add_argument("--door_width", type=float, default=0.8,
                       help="Door width")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for object generation")
    parser.add_argument("--run_seed", type=int, default=None,
                       help="Run seed (e.g., 0 for run00, 1 for run01, etc.)")
    # Object generation options
    parser.add_argument("--total_objects", type=int, required=True,
                       help="Total number of objects to distribute across all rooms")
    parser.add_argument("--object_mode", type=str, default="total", 
                       choices=["fixed", "total", "proportional"],
                       help="Object distribution mode: fixed (use fix_object_n), total (random distribution), proportional (based on room area)")
    parser.add_argument("--fix_object_n", type=str,
                       help="Comma-separated list of object counts for each room (e.g., '3,3,4' for 10 total objects in 3 rooms)")
    parser.add_argument("--proportional_to_area", action="store_true",
                       help="When object_mode='proportional', distribute objects proportionally to room areas")
    parser.add_argument("--port", type=int, default=1071,
                       help="TDW port")
    parser.add_argument("--with_ray", action="store_true",
                       help="Add direction rays to captured images")
    parser.add_argument("--overall_scale", type=float, default=1.0,
                       help="Overall scale multiplier for all objects (doors excluded)")
    parser.add_argument("--disable_gravity_fix", action="store_true",
                       help="Disable gravity fix for teleported objects (use simple ground placement)")
    parser.add_argument("--physics_settle_time", type=float, default=0.2,
                       help="Time to wait for physics to settle after object restoration (seconds)")
    # Model system support (new version)
    parser.add_argument("--builtin_models_path", type=str,
                       help="Path to built-in models JSON file (models/builtin_models.json)")
    parser.add_argument("--custom_models_path", type=str,
                       help="Path to custom models JSON file (models/custom_models.json)")
    
    # Custom model support (backward compatible)
    parser.add_argument("--custom_models_config", type=str,
                       help="[DEPRECATED] Path to custom models configuration file (models/custom_models.json)")
    parser.add_argument("--disable_custom_models", action="store_true",
                       help="[DEPRECATED] Disable custom models and use only library models")
    
    # Configuration file support
    parser.add_argument("--config", type=str,
                       help="Path to YAML configuration file (for validation settings)")
    
    args = parser.parse_args()
    
    # Load configuration file if provided
    config = {}
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                print(f"[INFO] Loaded configuration from: {config_path}")
            except Exception as e:
                print(f"[WARNING] Failed to load config file: {e}")
                config = {}
        else:
            print(f"[WARNING] Config file not found: {config_path}")

    # Fixed parameters (hidden from config/CLI)
    args.cell_size = 1.0
    args.wall_thickness = 0.01
    args.wall_height = 2.0
    args.door_width = 0.6
    args.with_ray = False
    args.physics_settle_time = 0.1
    args.disable_gravity_fix = False

    args.object_mode = "fixed"
    args.proportional_to_area = True

    # Inject fixed config values (preserve other sections like eval_tasks)
    config = config or {}
    obj_cfg = config.setdefault("object_generation", {})
    obj_cfg["distance_constraints"] = {"enable": True, "min_distance": 1.42}
    obj_cfg["collinear_detection"] = {"enable": True, "tolerance_width": 0.8}

    validity_cfg = config.setdefault("text_based_validity", {})
    validity_cfg["enabled"] = True
    validity_cfg["pre_render_validation"] = True
    validity_cfg["max_retries"] = 1000
    validity_cfg["seed_increment"] = 137
    
    # Validate inputs
    mask_path = Path(args.mask_path)
    if not mask_path.exists():
        print(f"[ERROR] Mask file not found: {mask_path}")
        return 1
    
    output_dir = Path(args.output)
    
    # Load mask to calculate proper offset
    mask_data = json.loads(mask_path.read_text())
    rows, cols = len(mask_data), len(mask_data[0])
    
    # Calculate offset to center world at (0, 0) - consistent with _cell_to_world
    x_offset = -(cols - 1) / 2.0 * args.cell_size
    z_offset = -(rows - 1) / 2.0 * args.cell_size
    
    # Parse fix_object_n if provided
    fix_object_n = None
    if args.fix_object_n:
        try:
            fix_object_n = [int(x.strip()) for x in args.fix_object_n.split(',')]
            print(f"[INFO] Using fixed object counts per room: {fix_object_n}")
        except ValueError as e:
            print(f"[ERROR] Invalid fix_object_n format: {args.fix_object_n}")
            print(f"[ERROR] Expected comma-separated integers, e.g., '6,6,3'")
            return 1
    
    # Validate object mode and parameters
    if args.object_mode == "fixed":
        if not fix_object_n:
            print(f"[ERROR] object_mode='fixed' requires --fix_object_n to be specified")
            return 1
        if sum(fix_object_n) != args.total_objects:
            print(f"[ERROR] fix_object_n must sum to total_objects ({args.total_objects}), got sum={sum(fix_object_n)}")
            return 1

    print("Enhanced Multi-Room Scene Generator")
    print("=" * 50)
    print(f"Mask file: {mask_path}")
    print(f"Output directory: {output_dir}")
    print(f"Seed: {args.seed}")
    print("")
    
    # Create generator and run
    generator = EnhancedMask2Scene(
        mask_path=str(mask_path),
        output_dir=str(output_dir),
        cell_size=args.cell_size,
        wall_thickness=args.wall_thickness,
        wall_height=args.wall_height,
        door_width=args.door_width,
        seed=args.seed,
        run_seed=args.run_seed,  # Pass run_seed
        total_objects=args.total_objects,
        port=args.port,
        x_offset=x_offset,
        z_offset=z_offset,
        with_ray=args.with_ray,
        overall_scale=args.overall_scale,
        enable_gravity_fix=not args.disable_gravity_fix,
        physics_settle_time=args.physics_settle_time,
        fix_object_n=fix_object_n,
        object_mode=args.object_mode,
        proportional_to_area=args.proportional_to_area,
        # New model system parameters
        builtin_models_path=args.builtin_models_path,
        custom_models_path=args.custom_models_path,
        # Configuration for validation
        config=config,
        # Legacy parameters for backward compatibility
        custom_models_config=args.custom_models_config,
        use_custom_models=not args.disable_custom_models
    )

    success = generator.generate_complete_scene()
    
    if success:
        print("\n" + "=" * 50)
        print("Scene generation completed successfully!")
        print(f"Check output directory: {output_dir.resolve()}")
        return 0
    else:
        print("\n" + "=" * 50) 
        print("Scene generation failed!")
        return 1

if __name__ == "__main__":
    exit(main()) 