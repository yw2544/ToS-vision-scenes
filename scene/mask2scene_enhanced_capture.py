#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Capture-related helpers for EnhancedMask2Scene.
Extracted from mask2scene_enhanced.py to reduce file length.
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from scene import mask2scene
from scene.mask2scene_enhanced_topdown import (
    _compute_object_id_array,
    build_seg_to_object_map_topdown,
    annotate_topdown_with_segmentation,
)


def find_doors_at_position(self, camera_pos: Dict[str, float], tolerance: float = 0.1) -> Tuple[List[int], List[int]]:
    """
    Find door and filler object IDs at the given camera position.

    Args:
        camera_pos: camera position {"x": x, "y": y, "z": z}
        tolerance: position matching tolerance

    Returns:
        (door_ids_at_position, filler_ids_at_position) tuple
    """
    doors_at_pos = []
    fillers_at_pos = []

    cam_x, cam_z = camera_pos["x"], camera_pos["z"]

    # Find doors
    for door_id, door_pos in self.door_positions.items():
        door_x, door_z = door_pos["x"], door_pos["z"]
        distance = ((cam_x - door_x)**2 + (cam_z - door_z)**2)**0.5

        if distance <= tolerance:
            doors_at_pos.append(door_id)

    # Find fillers
    for filler_id, filler_pos in self.filler_positions.items():
        filler_x, filler_z = filler_pos["x"], filler_pos["z"]
        distance = ((cam_x - filler_x)**2 + (cam_z - filler_z)**2)**0.5

        if distance <= tolerance:
            fillers_at_pos.append(filler_id)

    return doors_at_pos, fillers_at_pos


def temporarily_hide_items_for_camera(self, position: Dict[str, float], tolerance: float = 0.5):
    """Temporarily hide objects/doors/fillers located at camera position."""
    hidden_items = []
    commands = []

    if self.object_generator:
        for obj in self.object_generator.get_all_objects():
            final_pos = obj.get_final_position()
            if abs(final_pos["x"] - position["x"]) < 1e-4 and abs(final_pos["z"] - position["z"]) < 1e-4:
                commands.append({
                    "$type": "teleport_object",
                    "id": obj.object_id,
                    "position": {"x": 999, "y": -999, "z": 999}
                })
                hidden_items.append(("object", obj))
                break

    doors_at_pos, fillers_at_pos = self._find_doors_at_position(position, tolerance)

    for door_id in doors_at_pos:
        commands.append({
            "$type": "teleport_object",
            "id": door_id,
            "position": {"x": 999, "y": -999, "z": 999}
        })
        hidden_items.append(("door", door_id))

    for filler_id in fillers_at_pos:
        commands.append({
            "$type": "teleport_object",
            "id": filler_id,
            "position": {"x": 999, "y": -999, "z": 999}
        })
        hidden_items.append(("filler", filler_id))

    if commands:
        self.controller.communicate(commands)

    return hidden_items


def restore_hidden_items(self, hidden_items):
    """Restore previously hidden objects/doors/fillers."""
    if not hidden_items:
        return

    import time

    for item_type, data in hidden_items:
        if item_type == "object":
            obj = data
            self.controller.communicate([{
                "$type": "teleport_object",
                "id": obj.object_id,
                "position": obj.pos
            }])

            if self.enable_gravity_fix:
                self.controller.communicate([{
                    "$type": "set_kinematic_state",
                    "id": obj.object_id,
                    "is_kinematic": True,
                    "use_gravity": False
                }])

                for _ in range(3):
                    self.controller.communicate([{
                        "$type": "apply_force_to_object",
                        "id": obj.object_id,
                        "force": {"x": 0, "y": -0.2, "z": 0}
                    }])
                    self.controller.communicate([])
                    time.sleep(0.15)

                time.sleep(self.physics_settle_time)

                self.controller.communicate([{
                    "$type": "set_kinematic_state",
                    "id": obj.object_id,
                    "is_kinematic": True,
                    "use_gravity": False
                }])
            else:
                time.sleep(0.1)

            time.sleep(self.physics_settle_time)
        elif item_type == "door" and hasattr(self, 'door_positions'):
            original_pos = self.door_positions.get(data)
            if original_pos:
                self.controller.communicate([{
                    "$type": "teleport_object",
                    "id": data,
                    "position": original_pos
                }])
        elif item_type == "filler" and hasattr(self, 'filler_positions'):
            original_pos = self.filler_positions.get(data)
            if original_pos:
                self.controller.communicate([{
                    "$type": "teleport_object",
                    "id": data,
                    "position": original_pos
                }])

    time.sleep(0.1)


def capture_all_images(self) -> bool:
    """Capture all required images including annotated top-down view"""
    try:
        print("[INFO] Starting comprehensive image capture...")

        # CRITICAL: Must reuse door_id_mapping from _generate_base_metadata
        # to ensure metadata and photo filenames use identical IDs
        if not hasattr(self, 'door_id_mapping') or not self.door_id_mapping:
            raise RuntimeError(
                "door_id_mapping not initialized! _generate_base_metadata must be called before _capture_all_images"
            )

        print(f"[INFO] Using door_id_mapping with {len(self.door_id_mapping)} doors from metadata generation")

        # Use local reference for top-down annotation only
        # Camera positions below will use self.door_id_mapping directly
        door_id_mapping = self.door_id_mapping

        # Capture top-down view with segmentation for annotation
        print("[INFO] Capturing top-down view with segmentation...")

        # Add a small cube at agent position for top-down view marking (similar to oblique approach)
        agent = self.object_generator.get_agent()
        agent_marker_id = None
        if agent:
            agent_marker_id = self.controller.get_unique_id()
            print(f"[INFO] Adding agent marker cube at position ({agent.pos['x']}, {agent.pos['z']}) with ID {agent_marker_id}")

            # Add larger cube at agent position for better segmentation detection
            agent_cube_commands = [
                {
                    "$type": "load_primitive_from_resources",
                    "primitive_type": "Cube",
                    "id": agent_marker_id,
                    "position": {"x": agent.pos["x"], "y": 0.1, "z": agent.pos["z"]},  # Slightly higher
                    "rotation": {"x": 0, "y": 0, "z": 0}
                },
                # Scale it to be larger (10cm cube for better segmentation detection)
                {
                    "$type": "scale_object",
                    "id": agent_marker_id,
                    "scale_factor": {"x": 0.1, "y": 0.1, "z": 0.1}
                },
                # Make it bright red for unique segmentation ID
                {
                    "$type": "set_color",
                    "id": agent_marker_id,
                    "color": {"r": 1.0, "g": 0.0, "b": 0.0, "a": 1.0}
                }
            ]
            self.controller.communicate(agent_cube_commands)

        self.image_capture.set(frequency="once", avatar_ids=["top_down"], pass_masks=["_img", "_id"], save=False)
        self.controller.communicate([])
        topdown_imgs = self.image_capture.get_pil_images()

        # Save original top-down view (without global suffix)
        topdown_imgs["top_down"]["_img"].save(self.output_dir / "top_down.png")

        # Record top-down image metadata
        self.captured_images.append({
            "file": "top_down.png",
            "cam_id": "top_down",
            "pos": {"x": 0, "y": 20, "z": 0},  # High above center
            "direction": "down",
            "object_ratios": []
        })

        # Annotate top-down view using segmentation data
        if "_id" in topdown_imgs["top_down"]:
            rgb_img = topdown_imgs["top_down"]["_img"]
            id_img = topdown_imgs["top_down"]["_id"]

            # Get objects data
            all_objects = self.object_generator.get_all_objects()

            # Build segmentation mapping for objects only
            print("[INFO] Building segmentation ID to object ID mapping for accurate annotation...")
            seg_to_obj_map = build_seg_to_object_map_topdown(
                self.controller, self.image_capture, all_objects, agent_marker_id
            )

            # Build door segmentation pixel map
            door_pixel_map = None
            if hasattr(self, "door_object_ids") and hasattr(self, "door_positions"):
                try:
                    door_pixel_map = self._build_door_segmentation_map_topdown()
                except Exception as e:
                    print(f"[WARN] Failed to build door segmentation map: {e}")

            # Use precomputed grid mapping if available
            grid_pixel_map = getattr(self, "topdown_pixel_map", None)

            # Get agent position using segmentation ID difference method
            agent_pixel_position = None
            if agent_marker_id:
                print("[INFO] Detecting agent position using segmentation ID difference method...")

                # Get unique segmentation IDs with agent marker
                with_agent_arr = _compute_object_id_array(np.array(id_img))
                with_agent_ids = set(np.unique(with_agent_arr).tolist())
                with_agent_ids.discard(0)  # Remove background
                print(f"[DEBUG] Segmentation IDs with agent: {len(with_agent_ids)} unique IDs")

                # Hide agent marker and capture without agent
                self.controller.communicate([{"$type": "teleport_object", "id": agent_marker_id,
                                           "position": {"x": 999, "y": -999, "z": 999}}])
                self.controller.communicate([])

                # Capture image without agent
                self.image_capture.set(frequency="once", avatar_ids=["top_down"], pass_masks=["_id"], save=False)
                self.controller.communicate([])
                without_agent_img = self.image_capture.get_pil_images()["top_down"]["_id"]
                without_agent_arr = _compute_object_id_array(np.array(without_agent_img))
                without_agent_ids = set(np.unique(without_agent_arr).tolist())
                without_agent_ids.discard(0)  # Remove background
                print(f"[DEBUG] Segmentation IDs without agent: {len(without_agent_ids)} unique IDs")

                # Find the segmentation ID that exists with agent but not without agent
                agent_seg_ids = with_agent_ids - without_agent_ids
                print(f"[DEBUG] Agent segmentation IDs (difference): {agent_seg_ids}")

                if agent_seg_ids and getattr(self, "save_agent_seg_debug", True):
                    # Use the agent segmentation ID to find position
                    agent_seg_id = list(agent_seg_ids)[0]  # Take the first (should be only one)
                    agent_mask = (with_agent_arr == agent_seg_id)

                    if agent_mask.any():
                        # Find the center of the agent marker
                        agent_coords = np.where(agent_mask)
                        agent_y = int(np.mean(agent_coords[0]))
                        agent_x = int(np.mean(agent_coords[1]))
                        agent_pixel_count = int(agent_mask.sum())
                        agent_pixel_position = (agent_x, agent_y)
                        print(f"[DEBUG] Found agent at segmentation ID {agent_seg_id} with {agent_pixel_count} pixels at position ({agent_x}, {agent_y})")

                        # Save debug image showing agent segmentation
                        try:
                            from PIL import Image
                            agent_debug_img = Image.fromarray((agent_mask * 255).astype(np.uint8))
                            debug_path = str(self.output_dir / "agent_seg_debug.png")
                            agent_debug_img.save(debug_path)
                            print(f"[DEBUG] Saved agent segmentation mask to {debug_path}")
                        except Exception as e:
                            print(f"[DEBUG] Could not save debug image: {e}")
                    else:
                        print(f"[WARN] Agent segmentation ID {agent_seg_id} has no pixels")
                else:
                    print(f"[WARN] No agent segmentation ID found in difference")

                # Remove the agent marker cube after detection
                self.controller.communicate([{"$type": "destroy_object", "id": agent_marker_id}])
                print(f"[INFO] Removed agent marker cube (ID: {agent_marker_id}) after detection")

            # Get agent data
            agent = self.object_generator.get_agent()
            if agent:
                agent_data = {"x": agent.pos["x"], "z": agent.pos["z"]}
            else:
                agent_data = {"x": 0, "z": 0}

            # Get doors data
            doors_data = {}
            if self.door_handler:
                for door_id, processed_door in self.door_handler.processed_doors.items():
                    doors_data[door_id] = {
                        "center": processed_door.door_info.center,
                        "color": processed_door.color_name
                    }

            # Calculate scene bounds using mask dimensions and proper offset
            scene_bounds = None
            if self.room_analyzer:
                mask = mask2scene.load_mask(self.mask_path)
                rows, cols = len(mask), len(mask[0])
                cell_size = self.cell_size

                # Calculate world bounds using the same offset as in coordinate conversion
                offset_x = -(cols - 1) / 2.0 * cell_size
                offset_z = -(rows - 1) / 2.0 * cell_size

                min_x = offset_x
                max_x = offset_x + (cols - 1) * cell_size
                min_z = offset_z
                max_z = offset_z + (rows - 1) * cell_size
                scene_bounds = (min_x, max_x, min_z, max_z)
                print(f"[DEBUG] Scene bounds for annotation: X[{min_x:.1f}, {max_x:.1f}], Z[{min_z:.1f}, {max_z:.1f}]")

            # Annotate top-down view with mapping
            annotate_topdown_with_segmentation(
                rgb_img, id_img, all_objects, doors_data, agent_data,
                str(self.output_dir / "top_down_annotated.png"), scene_bounds,
                seg_to_obj_map=seg_to_obj_map, door_id_mapping=door_id_mapping,
                agent_pixel_position=agent_pixel_position, door_pixel_map=door_pixel_map, grid_pixel_map=grid_pixel_map
            )

            # Record annotated top-down image metadata
            self.captured_images.append({
                "file": "top_down_annotated.png",
                "cam_id": "top_down_annotated",
                "pos": {"x": 0, "y": 20, "z": 0},  # High above center
                "direction": "down",
                "object_ratios": []
            })

            # Build mask->pixel mapping for every integer grid cell (store in metadata)
            # Only if not already generated before object placement
            if not hasattr(self, 'topdown_pixel_map') or not self.topdown_pixel_map:
                try:
                    print("[INFO] Grid point mapping not found, generating now (may have occlusion issues)...")
                    mask = mask2scene.load_mask(self.mask_path)
                    self._generate_topdown_pixel_map(id_img, mask)
                except Exception as e:
                    print(f"[WARN] Failed to build top-down pixel map: {e}")
            else:
                print("[INFO] Grid point mapping already generated before object placement, skipping")

        # Capture room views with direction rays
        print("[INFO] Capturing room views with direction rays...")

        # Get all camera positions (agent + objects + doors)
        all_cam_positions = []

        # Add agent camera position (most important view)
        agent = self.object_generator.get_agent()
        if agent:
            all_cam_positions.append({
                "id": "agent",
                "label": "A",
                "position": {"x": agent.pos["x"], "y": 0.8, "z": agent.pos["z"]},
                "type": "agent"
            })

        # Add object camera positions (4 directions per object)
        if self.object_generator:
            all_objects = self.object_generator.get_all_objects()
            for i, obj in enumerate(all_objects, 1):
                obj_final_pos = obj.get_final_position()  # Includes default_position offset
                all_cam_positions.append({
                    "id": str(obj.object_id),
                    "label": str(i),
                    "position": {"x": obj_final_pos["x"], "y": 0.8, "z": obj_final_pos["z"]},
                    "type": "object"
                })

        # Add door camera positions (at door center coordinates) using numeric IDs
        if self.door_handler and hasattr(self, 'door_id_mapping'):
            for door_id, processed_door in self.door_handler.processed_doors.items():
                door_object_id = self.door_id_mapping[door_id]
                door_center = processed_door.door_info.center
                connected_rooms = processed_door.door_info.connected_rooms

                # For each connected room, create camera positions
                for room_id in connected_rooms:
                    all_cam_positions.append({
                        "id": str(door_object_id),
                        "label": f"D{door_object_id}",
                        "position": {"x": door_center[0], "y": 0.8, "z": door_center[1]},
                        "type": "door",
                        "room": room_id,  # Add room parameter for door cameras
                        "original_door_id": door_id
                    })

        # Capture views for all positions with direction rays (with error recovery)
        successful_captures = 0
        total_expected_captures = len(all_cam_positions) * 4

        for cam_idx, cam_spec in enumerate(all_cam_positions):
            pos = cam_spec["position"]
            cam_type = cam_spec.get("type", "unknown")

            print(f"[INFO] Capturing {cam_type} camera {cam_idx + 1}/{len(all_cam_positions)} (ID: {cam_spec['id']})")

            # Capture 4 directions with direction rays
            for direction in [0, 90, 180, 270]:
                try:
                    # Pass room parameter for door cameras
                    room_param = cam_spec.get("room", None)
                    self._capture_view_with_rays(pos, direction, cam_spec["id"], cam_spec["label"], room_param)
                    successful_captures += 1

                    # Add delay between captures to ensure object stability
                    import time
                    time.sleep(0.1)  # Increased from 0.02 to 0.1 seconds for better stability

                except Exception as e:
                    print(f"[ERROR] Failed to capture {cam_spec['id']} facing {direction}°: {e}")
                    # Continue with next direction instead of failing completely
                    continue

        print(f"[INFO] Successfully captured {successful_captures}/{total_expected_captures} perspective views")

        # Print capture summary
        total_images = len(all_cam_positions) * 4 + 2  # 4 directions per position + 2 top-down views
        print(f"[INFO] Captured {total_images} images:")
        print(f"  - Top-down views: 2 (original + annotated)")
        print(f"  - Agent views: {len([c for c in all_cam_positions if c['type'] == 'agent']) * 4}")
        print(f"  - Object views: {len([c for c in all_cam_positions if c['type'] == 'object']) * 4}")
        print(f"  - Door views: {len([c for c in all_cam_positions if c['type'] == 'door']) * 4}")

        return True

    except Exception as e:
        print(f"[ERROR] Failed to capture images: {e}")
        return False


def capture_task_viewpoint_images(self) -> bool:
    """Capture images from task viewpoints.

    This method captures images for bwd_nav, bwd_loc, bwd_pov tasks normally,
    and handles false_belief tasks specially (rotate object, capture, restore).

    Returns:
        True if all images captured successfully, False otherwise
    """
    try:
        print(f"\n[INFO] Capturing {len(self.task_viewpoints)} task viewpoint images...")

        if not self.controller or not self.image_capture:
            print("[ERROR] TDW controller or image capture not initialized")
            return False

        # Group viewpoints by task type for organized processing
        viewpoints_by_task = {}
        for vp in self.task_viewpoints:
            task_group = vp.get('task_group', vp['task_type'])
            if task_group not in viewpoints_by_task:
                viewpoints_by_task[task_group] = []
            viewpoints_by_task[task_group].append(vp)

        # Process each task type
        for task_type in ['bwd_nav', 'bwd_loc', 'bwd_pov', 'false_belief']:
            if task_type not in viewpoints_by_task:
                continue

            viewpoints = viewpoints_by_task[task_type]
            print(f"\n📷 Capturing {len(viewpoints)} images for {task_type}...")

            if task_type == 'false_belief':
                # Special handling for false_belief
                self._capture_false_belief_images(viewpoints)
            else:
                # Normal capture for other tasks
                self._capture_regular_task_images(viewpoints)

        print(f"[INFO] ✅ Captured all task viewpoint images")
        return True

    except Exception as e:
        print(f"[ERROR] Failed to capture task viewpoint images: {e}")
        import traceback
        traceback.print_exc()
        return False


def capture_regular_task_images(self, viewpoints: List[Dict]) -> None:
    """Capture images for regular task viewpoints (bwd_nav, bwd_loc, bwd_pov).

    Args:
        viewpoints: List of viewpoint dicts
    """
    for vp in viewpoints:
        pos_tdw = vp['pos_tdw']
        yaw = vp['yaw']
        image_name = vp['image_name']
        cam_id = vp.get('cam_id', f"task_{image_name}")
        direction_label = vp.get('direction_label', f"task_{vp.get('task_group', vp['task_type'])}")

        # Position camera
        x, z = float(pos_tdw[0]), float(pos_tdw[1])
        y = 0.8  # Standard camera height

        # Use same method as agent/object cameras: teleport + look_at
        # look_at with same y-height ensures horizontal view
        rad = math.radians(yaw)
        look_at_target = {
            "x": x + math.sin(rad) * 3,
            "y": y,  # Same height = horizontal view
            "z": z + math.cos(rad) * 3
        }

        self.main_cam.teleport({"x": x, "y": y, "z": z})
        self.main_cam.look_at(look_at_target)

        camera_position = {"x": x, "y": 0.0, "z": z}
        hidden_items = self._temporarily_hide_items_for_camera(camera_position)

        try:
            # Capture image using correct ImageCapture API
            self.image_capture.set(frequency="once", avatar_ids=["main_cam"], pass_masks=["_img"], save=False)
            self.controller.communicate([])
            images = self.image_capture.get_pil_images()

            # Save image
            if "main_cam" in images and "_img" in images["main_cam"]:
                img = images["main_cam"]["_img"]
                output_file = self.output_dir / f"{image_name}.png"
                img.save(output_file)
                print(f"  ✅ Saved: {image_name}.png")

                # Record in captured_images
                task_group = vp.get('task_group', vp['task_type'])
                self.captured_images.append({
                    "file": f"{image_name}.png",
                    "cam_id": cam_id,
                    "pos": {"x": x, "y": y, "z": z},
                    "rotation": {"y": float(yaw)},
                    "direction": direction_label,
                    "task_type": vp['task_type'],
                    "task_group": task_group,
                    "object_ratios": []  # Task images don't track object ratios
                })
            else:
                print(f"  ⚠️  Failed to capture {image_name}.png")
        finally:
            self._restore_hidden_items(hidden_items)


def capture_false_belief_images(self, viewpoints: List[Dict]) -> None:
    """Capture images for false_belief task with object rotation.

    For each false_belief viewpoint:
    1. Rotate the specified object
    2. Capture image
    3. Restore object to original rotation

    Args:
        viewpoints: List of false_belief viewpoint dicts
    """
    import time

    for vp in viewpoints:
        pos_tdw = vp['pos_tdw']
        yaw = vp['yaw']
        image_name = vp['image_name']
        rotation_info = vp.get('rotation_info', {})

        object_to_rotate = rotation_info.get('object_to_rotate')
        rotation_degrees = rotation_info.get('rotation_degrees', 0)
        original_yaw = rotation_info.get('original_yaw', 0)

        if not object_to_rotate:
            print(f"  ⚠️  No rotation info for {image_name}, skipping")
            continue

        # Find the object from object_generator (to get correct rotation with default)
        obj = None
        if self.object_generator and self.object_generator.all_objects:
            for o in self.object_generator.all_objects:
                # Try exact match first, then fuzzy match
                if o.name == object_to_rotate or o.name.lower().replace(' ', '_') == object_to_rotate.lower().replace(' ', '_'):
                    obj = o
                    break

        if obj is None:
            print(f"  ⚠️  Object '{object_to_rotate}' not found in object_generator, skipping {image_name}")
            continue

        object_id = obj.object_id

        # Get the ACTUAL rotation used in TDW (includes default_rotation)
        final_rot = obj.get_final_rotation()
        actual_yaw = final_rot['y']

        print(f"  🔄 Rotating {object_to_rotate} by {rotation_degrees}° (from actual TDW yaw {actual_yaw:.1f}° to {(actual_yaw + rotation_degrees) % 360:.1f}°)...")

        # Step 1: Rotate object (use actual_yaw as base)
        new_rotation_y = (actual_yaw + rotation_degrees) % 360
        rotate_cmd = {
            "$type": "rotate_object_to",
            "id": object_id,
            "rotation": {"x": 0, "y": new_rotation_y, "z": 0}
        }
        self.controller.communicate(rotate_cmd)

        # Wait for rotation to complete before capturing
        time.sleep(0.2)

        # Send an extra frame to ensure rotation is applied
        self.controller.communicate([])

        # Step 2: Position camera and capture
        x, z = float(pos_tdw[0]), float(pos_tdw[1])
        y = 0.8

        # Use same method as agent/object cameras: teleport + look_at
        # look_at with same y-height ensures horizontal view
        rad = math.radians(yaw)
        look_at_target = {
            "x": x + math.sin(rad) * 3,
            "y": y,  # Same height = horizontal view
            "z": z + math.cos(rad) * 3
        }

        self.main_cam.teleport({"x": x, "y": y, "z": z})
        self.main_cam.look_at(look_at_target)

        camera_position = {"x": x, "y": 0.0, "z": z}
        hidden_items = self._temporarily_hide_items_for_camera(camera_position)

        try:
            # Capture image using correct ImageCapture API
            self.image_capture.set(frequency="once", avatar_ids=["main_cam"], pass_masks=["_img"], save=False)
            self.controller.communicate([])
            images = self.image_capture.get_pil_images()

            # Save image
            if "main_cam" in images and "_img" in images["main_cam"]:
                img = images["main_cam"]["_img"]
                output_file = self.output_dir / f"{image_name}.png"
                img.save(output_file)
                print(f"  ✅ Saved: {image_name}.png")

                # Record in captured_images
                self.captured_images.append({
                    "file": f"{image_name}.png",
                    "cam_id": f"task_{image_name}",
                    "pos": {"x": x, "y": y, "z": z},
                    "direction": "task_false_belief",
                    "object_ratios": [],
                    "rotation_applied": {
                        "object": object_to_rotate,
                        "degrees": rotation_degrees
                    }
                })
            else:
                print(f"  ⚠️  Failed to capture {image_name}.png")
        finally:
            self._restore_hidden_items(hidden_items)

        # Step 3: Restore to actual TDW rotation (includes default_rotation)
        print(f"  ↩️  Restoring {object_to_rotate} to actual TDW rotation (y={actual_yaw:.1f}°)...")
        restore_cmd = {
            "$type": "rotate_object_to",
            "id": object_id,
            "rotation": {"x": 0, "y": actual_yaw, "z": 0}
        }
        self.controller.communicate(restore_cmd)

        # Wait for rotation to complete before next capture
        time.sleep(0.2)

        # Send an extra frame to ensure rotation is applied
        self.controller.communicate([])
        print(f"  ✅ Restored {object_to_rotate} to original rotation")


def capture_view_with_rays(self, pos, deg, tag, label, room=None):
    """Capture image with direction ray (adapted from pipeline_partial_label_only.py)"""
    try:
        rad = math.radians(deg)
        look_direction = (math.sin(rad), math.cos(rad))
        look_at = {
            "x": pos["x"] + math.sin(rad) * 3,  # Look further in direction
            "y": pos["y"],
            "z": pos["z"] + math.cos(rad) * 3
        }

        # Position camera and look at target
        self.main_cam.teleport(pos)
        self.main_cam.look_at(look_at)

        # Hide object at camera position if this is an object camera (using teleport like pipeline)
        hidden_object = None
        if tag != "agent" and not tag.startswith("door_") and tag.isdigit():
            # This is an object camera, find and hide the object at this position
            object_id = int(tag)

            # Find the object in our object list (similar to pipeline approach)
            if self.object_generator:
                all_objects = self.object_generator.get_all_objects()
                for obj in all_objects:
                    if obj.object_id == object_id:
                        obj_final_pos = obj.get_final_position()
                        # Check if this object is at the camera position
                        if (abs(obj_final_pos["x"] - pos["x"]) < 1e-4 and
                            abs(obj_final_pos["z"] - pos["z"]) < 1e-4):
                            hidden_object = obj
                            break

            # Teleport object away (exactly like pipeline_partial_label_only.py)
            if hidden_object:
                try:
                    self.controller.communicate([{
                        "$type": "teleport_object",
                        "id": hidden_object.object_id,
                        "position": {"x": 999, "y": -999, "z": 999}
                    }])
                except Exception as e:
                    print(f"[ERROR] Failed to teleport object {hidden_object.object_id}: {e}")
                    hidden_object = None

        # Check if this is a door camera, and if so, hide doors/fillers at this position
        hidden_doors_fillers = []
        if tag.startswith("D") or (hasattr(self, 'door_id_mapping') and any(str(door_obj_id) == tag for door_obj_id in self.door_id_mapping.values())):
            # Find doors and fillers at this camera position
            doors_at_pos, fillers_at_pos = self._find_doors_at_position(pos, tolerance=0.5)

            # Hide doors at this position
            for door_id in doors_at_pos:
                try:
                    self.controller.communicate([{
                        "$type": "teleport_object",
                        "id": door_id,
                        "position": {"x": 999, "y": -999, "z": 999}
                    }])
                    hidden_doors_fillers.append(('door', door_id))
                except Exception as e:
                    print(f"[WARN] Failed to hide door {door_id}: {e}")

            # Hide fillers at this position
            for filler_id in fillers_at_pos:
                try:
                    self.controller.communicate([{
                        "$type": "teleport_object",
                        "id": filler_id,
                        "position": {"x": 999, "y": -999, "z": 999}
                    }])
                    hidden_doors_fillers.append(('filler', filler_id))
                except Exception as e:
                    print(f"[WARN] Failed to hide filler {filler_id}: {e}")

            if hidden_doors_fillers:
                print(f"[INFO] Hid {len(hidden_doors_fillers)} doors/fillers for door camera at ({pos['x']:.2f}, {pos['z']:.2f})")

        # Create direction ray only if with_ray is enabled (with reduced segments to avoid TDW crash)
        rayids = []
        if self.with_ray:
            rayids = self._create_direction_ray(pos, deg)

        # Wait for objects to be properly processed
        import time
        time.sleep(0.05)  # Reduced wait time

        # Capture color image
        self.image_capture.set(frequency="once", avatar_ids=["main_cam"], pass_masks=["_img"], save=False)
        self.controller.communicate([])
        color_images = self.image_capture.get_pil_images()["main_cam"]

        # Remove direction ray immediately after capture (before restoring object)
        if rayids:
            try:
                destroy_commands = [{"$type": "destroy_object", "id": rid} for rid in rayids]
                self.controller.communicate(destroy_commands)
                print(f"[DEBUG] Removed {len(rayids)} ray segments")
            except Exception as e:
                print(f"[WARN] Failed to remove some ray segments: {e}")

        # Restore hidden object (teleport back to original position)
        if hidden_object is not None:
            try:
                if self.enable_gravity_fix:
                    # Method 1: Use physics to settle object naturally
                    # First teleport back to original position
                    self.controller.communicate([{
                        "$type": "teleport_object",
                        "id": hidden_object.object_id,
                        "position": hidden_object.pos
                    }])

                    # Enable gravity and physics to let object settle naturally
                    self.controller.communicate([
                        {
                            "$type": "set_kinematic_state",
                            "id": hidden_object.object_id,
                            "is_kinematic": True,
                            "use_gravity": False  # Explicitly enable gravity
                        }
                    ])

                    # Apply multiple small forces to ensure settling and wait between each
                    for _ in range(3):  # Apply force multiple times for better settling
                        self.controller.communicate([{
                            "$type": "apply_force_to_object",
                            "id": hidden_object.object_id,
                            "force": {"x": 0, "y": -0.2, "z": 0}  # Increased downward force
                        }])
                        self.controller.communicate([])  # Process the physics step
                        import time
                        time.sleep(0.15)  # Wait between each force application

                    # Final settling time (configurable)
                    time.sleep(self.physics_settle_time)

                    # Set back to kinematic to prevent further movement
                    self.controller.communicate([{
                        "$type": "set_kinematic_state",
                        "id": hidden_object.object_id,
                        "is_kinematic": True,
                        "use_gravity": False  # Disable gravity to keep object stable
                    }])

                else:
                    # Method 2: Simple teleport to ground level
                    # Calculate ground position (y=0 for most objects, or use bounds)
                    ground_pos = hidden_object.pos.copy()
                    ground_pos["y"] = 0.0  # Reset to ground level

                    self.controller.communicate([{
                        "$type": "teleport_object",
                        "id": hidden_object.object_id,
                        "position": ground_pos
                    }])

                    # Wait a moment for simple teleport to settle
                    import time
                    time.sleep(0.1)

                    print(f"[DEBUG] Restored object {hidden_object.object_id} to ground level position")

                # Additional wait time after object restoration to ensure stability
                time.sleep(self.physics_settle_time)  # Extra stabilization time (configurable)

            except Exception as e:
                print(f"[WARN] Failed to restore object {hidden_object.object_id}: {e}")

        # Restore doors and fillers that were hidden for this door camera
        if hidden_doors_fillers:
            for obj_type, obj_id in hidden_doors_fillers:
                try:
                    if obj_type == 'door' and obj_id in self.door_positions:
                        original_pos = self.door_positions[obj_id]
                        self.controller.communicate([{
                            "$type": "teleport_object",
                            "id": obj_id,
                            "position": original_pos
                        }])
                    elif obj_type == 'filler' and obj_id in self.filler_positions:
                        original_pos = self.filler_positions[obj_id]
                        self.controller.communicate([{
                            "$type": "teleport_object",
                            "id": obj_id,
                            "position": original_pos
                        }])
                except Exception as e:
                    print(f"[WARN] Failed to restore {obj_type} {obj_id}: {e}")

            # Wait for doors and fillers to settle
            import time
            time.sleep(0.1)  # Allow time for doors and fillers to settle

        # Save image
        dir_name = {0: "north", 90: "east", 180: "south", 270: "west"}[deg]
        fname = f"{tag}_facing_{dir_name}.png"

        if "_img" in color_images:
            img = color_images["_img"].copy()
            img.save(self.output_dir / fname)

            # Record image metadata with calculated object ratios
            image_info = {
                "file": fname,
                "cam_id": tag,
                "pos": {"x": pos["x"], "y": pos["y"], "z": pos["z"]},
                "direction": dir_name,
                "object_ratios": self._calculate_object_ratios(pos, deg, tag, room)
            }
            self.captured_images.append(image_info)

            if self.with_ray:
                print(f"[INFO] Captured {tag} facing {dir_name} with direction ray")
            else:
                print(f"[INFO] Captured {tag} facing {dir_name}")
        else:
            print(f"[ERROR] Failed to capture image for {tag} facing {dir_name}")

    except Exception as e:
        print(f"[ERROR] Failed to capture view with rays for {tag} facing {deg}°: {e}")
        import traceback
        traceback.print_exc()
