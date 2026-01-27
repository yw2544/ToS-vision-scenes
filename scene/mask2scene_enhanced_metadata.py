#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Metadata and task-viewpoint helpers for EnhancedMask2Scene.
Extracted from mask2scene_enhanced.py to reduce file length.
"""

import json
import math
from typing import Dict, List, Optional

from scene import mask2scene
from utils.orientation_instruction_generator import OrientationInstructionGenerator
from utils.task_viewpoint_generator import generate_task_viewpoints, viewpoints_to_camera_metadata


def generate_base_metadata(self) -> bool:
    """
    Generate base metadata (all fields except images) before TDW initialization.
    This allows metadata to be available before starting TDW rendering.
    """
    try:
        print("[INFO] Generating base meta_data.json (before TDW initialization)...")

        # Calculate scene dimensions using mask size
        mask = mask2scene.load_mask(self.mask_path)
        rows, cols = len(mask), len(mask[0])
        scene_width = cols * self.cell_size
        scene_depth = rows * self.cell_size
        offset = self.room_analyzer.get_offset() if self.room_analyzer else (0, 0)

        # Create metadata in standard format with mask and offset
        metadata = {
            "room_size": [scene_width, scene_depth],
            "screen_size": [768, 768],
            "seed": self.seed,
            "run_seed": self.run_seed,
            "min_distance": self.object_generator.min_distance,
            "objects": [],
            "cameras": [],
            "images": [],  # Empty at this stage, will be filled after image capture
            "rooms": {},
            "doors": {},
            "room_object_assignments": {},
            "mask": mask,
            "offset": (-offset[0], -offset[1])
        }

        # Add object information in standard format
        if self.object_generator:
            all_objects = self.object_generator.get_all_objects()
            for i, obj in enumerate(all_objects, 1):
                final_pos = obj.get_final_position()
                # Keep base rotation (no default_rotation) for 90/180/270 labeling
                obj_data = {
                    "object_id": obj.object_id,
                    "model": obj.model,
                    "name": obj.name,
                    "label": str(i),
                    "pos": final_pos,
                    "rot": obj.rot,
                    "size": list(obj.size),
                    "attributes": obj.attributes
                }
                metadata["objects"].append(obj_data)

        # Add door information to objects with numeric IDs
        # CRITICAL: Generate door_id_mapping only once to ensure consistency
        if not hasattr(self, 'door_id_mapping'):
            self.door_id_mapping = {}

        if self.door_handler:
            # Only generate door IDs if mapping is empty
            if not self.door_id_mapping:
                print(f"[INFO] Generating unique door IDs for {len(self.door_handler.processed_doors)} doors...")
                for door_id, processed_door in self.door_handler.processed_doors.items():
                    door_center = processed_door.door_info.center
                    door_object_id = self._generate_door_id_from_coordinates(door_center, door_id)
                    self.door_id_mapping[door_id] = door_object_id
                    print(f"[INFO] Door {door_id} at ({door_center[0]:.1f}, {door_center[1]:.1f}) -> object_id: {door_object_id}")
            else:
                print(f"[INFO] Reusing existing door_id_mapping (already initialized)")

            # Verify all doors have IDs before proceeding
            missing_doors = set(self.door_handler.processed_doors.keys()) - set(self.door_id_mapping.keys())
            if missing_doors:
                raise RuntimeError(f"Door ID mapping incomplete! Missing doors: {missing_doors}")

            # Add door objects to metadata using the mapping
            for door_id, processed_door in self.door_handler.processed_doors.items():
                door_object_id = self.door_id_mapping[door_id]
                door_center = processed_door.door_info.center
                door_data = {
                    "object_id": door_object_id,
                    "model": f"{processed_door.color_name}_door",
                    "name": f"{processed_door.color_name} door",
                    "pos": {"x": door_center[0], "y": 0, "z": door_center[1]},
                    "rot": {"x": 0, "y": 0, "z": 0},
                    "size": [processed_door.door_info.width, 2.0],
                    "attributes": {
                        "color": processed_door.color_name,
                        "orientation": processed_door.door_info.orientation,
                        "connected_rooms": processed_door.door_info.connected_rooms
                    }
                }
                metadata["objects"].append(door_data)

        # Add camera information in standard format
        # Agent camera
        agent = self.object_generator.get_agent() if self.object_generator else None
        if agent:
            metadata["cameras"].append({
                "id": "agent",
                "label": "A",
                "position": agent.pos,
                "rotation": {"y": 0.0}
            })

        # Object cameras
        if self.object_generator:
            all_objects = self.object_generator.get_all_objects()
            for i, obj in enumerate(all_objects, 1):
                obj_final_pos = obj.get_final_position()
                yaw = math.degrees(math.atan2(-obj_final_pos["x"], -obj_final_pos["z"]))

                metadata["cameras"].append({
                    "id": str(obj.object_id),
                    "label": str(i),
                    "position": {
                        "x": obj_final_pos["x"],
                        "y": 0.8,
                        "z": obj_final_pos["z"]
                    },
                    "rotation": {"y": yaw}
                })

        # Add door cameras using numeric IDs
        if self.door_handler and self.door_id_mapping:
            for door_id, processed_door in self.door_handler.processed_doors.items():
                door_object_id = self.door_id_mapping[door_id]
                door_center = processed_door.door_info.center
                metadata["cameras"].append({
                    "id": str(door_object_id),
                    "label": f"D{door_object_id}",
                    "position": {
                        "x": door_center[0],
                        "y": 0.8,
                        "z": door_center[1]
                    },
                    "rotation": {"y": 0.0}
                })

        # Add room information
        if self.room_analyzer:
            for room_id, room in self.room_analyzer.rooms.items():
                bounds = self.room_analyzer.get_room_bounds_world(room_id)
                metadata["rooms"][room_id] = {
                    "center": room.center,
                    "bounds": bounds,
                    "area": room.area,
                    "connected_rooms": list(self.room_analyzer.room_connections.get(room_id, []))
                }
                metadata["room_object_assignments"][room_id] = []

        # Assign objects to rooms
        if self.object_generator:
            all_objects = self.object_generator.get_all_objects()
            for obj in all_objects:
                if obj.room_id in metadata["room_object_assignments"]:
                    metadata["room_object_assignments"][obj.room_id].append({
                        "object_id": obj.object_id,
                        "name": obj.name,
                        "label": next((cam["label"] for cam in metadata["cameras"] if cam["id"] == str(obj.object_id)), str(obj.object_id))
                    })

        # Add door information
        if self.door_handler:
            for door_id, processed_door in self.door_handler.processed_doors.items():
                metadata["doors"][door_id] = {
                    "center": processed_door.door_info.center,
                    "color": processed_door.color_name,
                    "connected_rooms": processed_door.door_info.connected_rooms,
                    "orientation": processed_door.door_info.orientation,
                    "width": processed_door.door_info.width
                }

        # Add only validated_tasks (simplified validation metadata)
        if hasattr(self, '_validation_metadata') and self._validation_metadata:
            validation_data = self._validation_metadata.get('pre_render_validation', {})
            validated_tasks = validation_data.get('validated_tasks', [])
            if validated_tasks:
                metadata['validated_tasks'] = validated_tasks
                print(f"[INFO] Added validated tasks to metadata: {', '.join(validated_tasks)}")

        # Save metadata to JSON file
        metadata_path = self.output_dir / "meta_data.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"[INFO] Base meta_data.json exported to: {metadata_path}")
        print(f"[INFO] (images field is empty and will be updated after TDW rendering)")

        return True

    except Exception as e:
        print(f"[ERROR] Failed to generate base metadata: {e}")
        import traceback
        traceback.print_exc()
        return False


def update_metadata_with_images(self) -> bool:
    """
    Update metadata with captured images after TDW rendering completes.
    This only modifies the 'images' field in the existing metadata file.
    """
    try:
        print("[INFO] Updating meta_data.json with captured images...")

        metadata_path = self.output_dir / "meta_data.json"

        # Load existing metadata
        if not metadata_path.exists():
            print(f"[ERROR] Base metadata file not found: {metadata_path}")
            return False

        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        # Update images field with captured images
        metadata["images"] = self.captured_images

        # Save updated metadata
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"[INFO] Updated meta_data.json with {len(self.captured_images)} captured images")

        return True

    except Exception as e:
        print(f"[ERROR] Failed to update metadata with images: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_task_viewpoints_from_metadata(self) -> bool:
    """Generate task viewpoints from VAGEN evaluation tasks.

    This method reads the base metadata and generates additional viewpoints
    for backward tasks (bwd_nav, bwd_loc, bwd_pov) and false_belief task.
    It also updates the metadata with camera configurations for these viewpoints.

    Returns:
        True if viewpoints generated successfully, False otherwise
    """
    try:
        print("[INFO] Generating task viewpoints from VAGEN tasks...")

        # Load the base metadata
        metadata_path = self.output_dir / "meta_data.json"
        if not metadata_path.exists():
            print("[ERROR] Base metadata not found, cannot generate viewpoints")
            return False

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Generate viewpoints using the module
        # Use run_seed (0, 1, 2...) to match VAGEN EvaluationManager's seed
        self.task_viewpoints = generate_task_viewpoints(
            metadata=metadata,
            run_seed=self.run_seed,
            num_questions_per_task=3
        )

        if not self.task_viewpoints:
            print("[WARNING] No task viewpoints were generated")
            return False

        print(f"[INFO] ✅ Generated {len(self.task_viewpoints)} task viewpoints")

        # Convert viewpoints to camera metadata and add to metadata
        task_cameras = viewpoints_to_camera_metadata(self.task_viewpoints)
        metadata["cameras"].extend(task_cameras)

        # Save updated metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"[INFO] Added {len(task_cameras)} task cameras to metadata")

        return True

    except Exception as e:
        print(f"[ERROR] Failed to generate task viewpoints: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_orientation_instruction(self) -> bool:
    """Generate orientation instruction image from metadata"""
    try:
        print("[INFO] Generating orientation instruction image...")

        # Initialize orientation instruction generator if not already done
        if self.orientation_instruction_generator is None:
            # Default instruction images directory - adjust path as needed
            instruction_images_dir = "./models/new_models"
            self.orientation_instruction_generator = OrientationInstructionGenerator(
                instruction_images_dir=instruction_images_dir,
                output_dir=self.output_dir
            )

        # Find metadata file
        metadata_path = self.output_dir / "meta_data.json"
        if not metadata_path.exists():
            print(f"[ERROR] Metadata file not found: {metadata_path}")
            return False

        # Generate orientation instruction image
        instruction_path = self.orientation_instruction_generator.generate_instruction_image(
            metadata_path=metadata_path,
            output_filename="orientation_instruction.png"
        )

        if instruction_path:
            print(f"[INFO] Orientation instruction image generated successfully: {instruction_path}")
            return True
        else:
            print("[ERROR] Failed to generate orientation instruction image")
            return False

    except Exception as e:
        print(f"[ERROR] Failed to generate orientation instruction: {e}")
        import traceback
        traceback.print_exc()
        return False


def find_object_id_by_name(self, object_name: str) -> Optional[int]:
    """Find TDW object ID by object name from metadata.

    Supports fuzzy matching (spaces vs underscores).

    Args:
        object_name: Name of the object (e.g., 'chair', 'table', 'fire extinguisher')

    Returns:
        Object ID or None if not found
    """
    # Try to load metadata first (most reliable)
    metadata_path = self.output_dir / "meta_data.json"
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            # Normalize the search name (for fuzzy matching)
            search_name_normalized = object_name.lower().replace(' ', '_')

            # Search in metadata objects
            for obj in metadata.get('objects', []):
                obj_name = obj.get('name', '')
                obj_name_normalized = obj_name.lower().replace(' ', '_')

                # Try exact match first
                if obj_name == object_name:
                    print(f"[DEBUG] Found object '{object_name}' with ID {obj['object_id']} (exact match)")
                    return obj['object_id']

                # Try normalized match (fuzzy)
                if obj_name_normalized == search_name_normalized:
                    print(f"[DEBUG] Found object '{object_name}' as '{obj_name}' with ID {obj['object_id']} (fuzzy match)")
                    return obj['object_id']

            print(f"[DEBUG] Object '{object_name}' not found in metadata")
        except Exception as e:
            print(f"[DEBUG] Failed to read metadata: {e}")

    # Fallback: search in object_generator (if metadata not available)
    if self.object_generator and self.object_generator.all_objects:
        search_name_normalized = object_name.lower().replace(' ', '_')

        for obj in self.object_generator.all_objects:
            obj_name_normalized = obj.name.lower().replace(' ', '_')

            if obj.name == object_name or obj_name_normalized == search_name_normalized:
                print(f"[DEBUG] Found object '{object_name}' with ID {obj.object_id} (from object_generator)")
                return obj.object_id

    return None
