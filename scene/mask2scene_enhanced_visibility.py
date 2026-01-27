#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visibility and ray helpers for EnhancedMask2Scene.
Extracted from mask2scene_enhanced.py to reduce file length.
"""

import math
from typing import Dict, List, Optional, Tuple


def get_object_bounds(self, model_name: str, rotation: dict = {"x": 0, "y": 0, "z": 0}, scale: float = 1.0):
    """Get object bounds from TDW ModelLibrarian with scale applied"""
    try:
        record = self.model_lib.get_record(model_name)
        bounds = record.bounds

        # Apply scale to all dimensions
        width = (bounds['right']['x'] - bounds['left']['x']) * scale
        depth = (bounds['front']['z'] - bounds['back']['z']) * scale
        height = (bounds['top']['y'] - bounds['bottom']['y']) * scale

        # Apply Y rotation effect
        y_rotation = rotation.get('y', 0) % 360
        if y_rotation in [90, 270]:
            width, depth = depth, width
        elif y_rotation not in [0, 180]:
            angle_rad = math.radians(y_rotation)
            new_width = abs(width * math.cos(angle_rad)) + abs(depth * math.sin(angle_rad))
            new_depth = abs(width * math.sin(angle_rad)) + abs(depth * math.cos(angle_rad))
            width, depth = new_width, new_depth

        return (width, depth, height)
    except Exception as e:
        print(f"Failed to get bounds for {model_name}: {e}")
        return (1.0 * scale, 1.0 * scale, 1.0 * scale)


def is_object_bbox_in_view(self, obj, cam_pos: Dict[str, float], look_direction: Tuple[float, float], fov_deg: float = 90) -> bool:
    """Check if object bounding box is within camera field of view using geometric intersection"""
    try:
        from shapely.geometry import Polygon
        import numpy as np

        # Get object position and bounds (apply overall_scale)
        obj_final_pos = obj.get_final_position()
        obj_final_rot = obj.get_final_rotation()
        x0, z0 = obj_final_pos["x"], obj_final_pos["z"]
        obj_scale = getattr(obj, 'scale', 1.0) * self.overall_scale
        width, depth, height = self.get_object_bounds(obj.model, obj_final_rot, obj_scale)
        hx, hz = width/2, depth/2

        # Object bounding box corners
        obj_corners = [
            (x0-hx, z0-hz), (x0+hx, z0-hz), (x0+hx, z0+hz), (x0-hx, z0+hz)
        ]

        # Camera position and look direction
        cx, cz = cam_pos["x"], cam_pos["z"]
        look_dir = np.array(look_direction, dtype=float)
        look_dir = look_dir / np.linalg.norm(look_dir)

        half_fov = math.radians(fov_deg / 2)

        def rotate_vector(v, angle):
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            return np.array([v[0]*cos_a - v[1]*sin_a, v[0]*sin_a + v[1]*cos_a])

        # Check if any corner is within FOV
        for corner_x, corner_z in obj_corners:
            to_corner = np.array([corner_x - cx, corner_z - cz])
            if np.linalg.norm(to_corner) < 1e-6:
                return True  # Corner is at camera position

            to_corner_norm = to_corner / np.linalg.norm(to_corner)
            cos_angle = np.clip(np.dot(look_dir, to_corner_norm), -1.0, 1.0)
            angle = math.acos(cos_angle)

            if angle <= half_fov:
                return True  # At least one corner is in view

        # Also check if FOV intersects with object edges (more complex check)
        # Create FOV polygon and check intersection with object polygon
        obj_polygon = Polygon(obj_corners)

        # Create FOV sector polygon
        radius = max(20, np.linalg.norm([x0-cx, z0-cz]) * 2)  # Large enough radius
        sector_points = [(cx, cz)]
        for i in range(32 + 1):  # 32 samples for smoother sector
            angle = -half_fov + i * (2 * half_fov / 32)
            direction = rotate_vector(look_dir, angle)
            point = (cx + direction[0] * radius, cz + direction[1] * radius)
            sector_points.append(point)

        fov_polygon = Polygon(sector_points)

        # Check if polygons intersect
        return obj_polygon.intersects(fov_polygon)

    except Exception as e:
        print(f"Error in bbox view check: {e}")
        # Fallback to simple center-based check
        return self.is_object_center_in_view(obj, cam_pos, look_direction, fov_deg)


def is_object_center_in_view(self, obj, cam_pos: Dict[str, float], look_direction: Tuple[float, float], fov_deg: float = 90) -> bool:
    """Check if object center is within camera field of view"""
    import numpy as np

    # Vector from camera to object center
    obj_final_pos = obj.get_final_position()
    obj_x, obj_z = obj_final_pos["x"], obj_final_pos["z"]
    cam_x, cam_z = cam_pos["x"], cam_pos["z"]

    to_obj = np.array([obj_x - cam_x, obj_z - cam_z])
    if np.linalg.norm(to_obj) < 1e-6:
        return True  # Object is at camera position

    to_obj_norm = to_obj / np.linalg.norm(to_obj)
    look_dir_norm = np.array(look_direction) / np.linalg.norm(look_direction)

    # Calculate angle between look direction and direction to object
    cos_angle = np.clip(np.dot(look_dir_norm, to_obj_norm), -1.0, 1.0)
    angle_deg = math.degrees(math.acos(cos_angle))

    return angle_deg <= fov_deg / 2


def compute_visibility_ratio(self, target_obj, cam_pos, look_direction, fov_deg=90, samples=64):
    """Compute visibility ratio based on top-down view geometry using bounding box intersection"""
    try:
        from shapely.geometry import Polygon
        import numpy as np

        target_obj_final_pos = target_obj.get_final_position()
        target_obj_final_rot = target_obj.get_final_rotation()
        x0, z0 = target_obj_final_pos["x"], target_obj_final_pos["z"]
        obj_scale = getattr(target_obj, 'scale', 1.0) * self.overall_scale
        width, depth, height = self.get_object_bounds(target_obj.model, target_obj_final_rot, obj_scale)
        hx, hz = width/2, depth/2

        # Target object bounding box
        B = Polygon([
            (x0-hx, z0-hz), (x0+hx, z0-hz), (x0+hx, z0+hz), (x0-hx, z0+hz)
        ])

        # Camera field of view sector
        cx, cz = cam_pos["x"], cam_pos["z"]
        look_dir = np.array(look_direction, dtype=float)
        look_dir /= np.linalg.norm(look_dir)

        half_fov = math.radians(fov_deg / 2)

        def rotate_vector(v, angle):
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            return np.array([v[0]*cos_a - v[1]*sin_a, v[0]*sin_a + v[1]*cos_a])

        # Calculate sector radius
        corners = np.array(B.exterior.coords[:-1])
        distances = np.linalg.norm(corners - np.array([cx, cz]), axis=1)
        radius = distances.max() * 1.5

        # Construct sector polygon
        sector_points = [(cx, cz)]
        for i in range(samples + 1):
            angle = -half_fov + i * (2 * half_fov / samples)
            direction = rotate_vector(look_dir, angle)
            point = (cx + direction[0] * radius, cz + direction[1] * radius)
            sector_points.append(point)

        F = Polygon(sector_points)
        intersection = B.intersection(F)

        if B.area > 0:
            return intersection.area / B.area
        return 0.0

    except Exception as e:
        print(f"Error in visibility ratio calculation: {e}")
        return 0.0


def calculate_object_ratios(self, camera_pos, camera_deg, cam_id, room_filter=None):
    """
    Calculate object visibility ratios using proper bounding box intersection
    For door cameras, room_filter specifies which room's objects to include
    Returns list of object ratio dicts with object_id, model, visibility_ratio, occlusion_ratio
    """
    object_ratios = []

    if not self.object_generator:
        return object_ratios

    # Get camera's room (for agent cam_id="agent", for objects use their room_id)
    camera_room_id = None

    # For door cameras, use room_filter if provided
    if room_filter is not None:
        camera_room_id = room_filter
    elif cam_id == "agent":
        # Agent's room is the main room
        agent = self.object_generator.get_agent()
        if agent:
            camera_room_id = agent.room_id
    else:
        # Object camera - find the object's room
        try:
            obj_id = int(cam_id)
            all_objects = self.object_generator.get_all_objects()
            for obj in all_objects:
                if obj.object_id == obj_id:
                    camera_room_id = obj.room_id
                    break
        except ValueError:
            # Could be a door ID, try to find it
            if hasattr(self, 'door_id_mapping'):
                for orig_door_id, door_obj_id in self.door_id_mapping.items():
                    if str(door_obj_id) == cam_id:
                        # This is a door camera, but room_filter should have been provided
                        if self.door_handler and orig_door_id in self.door_handler.processed_doors:
                            connected_rooms = self.door_handler.processed_doors[orig_door_id].door_info.connected_rooms
                            if connected_rooms:
                                camera_room_id = connected_rooms[0]  # Default to first room
                        break

    if camera_room_id is None:
        return object_ratios

    # Get all objects in the same room (excluding the camera object itself)
    all_objects = self.object_generator.get_all_objects()
    room_objects = []
    for obj in all_objects:
        if obj.room_id == camera_room_id and str(obj.object_id) != cam_id:
            room_objects.append(obj)

    # Calculate look direction from camera degree
    rad = math.radians(camera_deg)
    look_direction = (math.sin(rad), math.cos(rad))

    print(f"[INFO] Calculating object ratios for {cam_id} in room {camera_room_id} facing {camera_deg}°")

    # Calculate visibility for each object in the room using proper bounding box method
    for obj in room_objects:
        try:
            # Use bbox-based checking for consistency with other pipelines
            bbox_in_view = self.is_object_bbox_in_view(obj, camera_pos, look_direction)
            visibility_ratio = self.compute_visibility_ratio(obj, camera_pos, look_direction)

            # Simple occlusion check (could be enhanced with actual ray casting)
            # For now, assume no occlusion if object is clearly visible
            is_occluded = False
            if bbox_in_view and visibility_ratio > 0.1:
                # Simple distance-based occlusion estimation
                obj_final_pos = obj.get_final_position()
                distance = ((obj_final_pos["x"] - camera_pos["x"])**2 + (obj_final_pos["z"] - camera_pos["z"])**2)**0.5
                # Assume objects are occluded if they are very far (>8 units) or visibility is very low
                is_occluded = distance > 8.0 or visibility_ratio < 0.3

            # Only include objects that are actually in view
            if bbox_in_view and visibility_ratio > 0.05:  # Minimum threshold for inclusion
                object_ratios.append({
                    "object_id": obj.object_id,
                    "model": obj.model,
                    "visibility_ratio": round(float(visibility_ratio), 3),
                    "occlusion_ratio": 1.0 if is_occluded else 0.0  # 0.0 = not occluded, 1.0 = occluded
                })

                print(f"  ✓ {obj.name}: bbox in view, visibility {visibility_ratio:.3f}, {'occluded' if is_occluded else 'not occluded'}")
            else:
                if not bbox_in_view:
                    print(f"  - {obj.name}: bbox not in view")
                else:
                    print(f"  - {obj.name}: visibility too low ({visibility_ratio:.3f})")

        except Exception as e:
            print(f"  ERROR calculating ratios for {obj.name}: {e}")
            continue

    return object_ratios


def create_direction_ray(self, pos, deg):
    """
    Create a dashed red direction ray on the ground pointing in the specified direction.
    Return a list of all segment IDs so they can be destroyed later.
    Based exactly on pipeline_partial_label_only.py implementation
    """
    try:
        # Use fixed ray length similar to pipeline_partial_label_only.py
        ray_length = 6  # Fixed length, much simpler

        # Dashed line parameters (exactly from pipeline)
        dash_len = 0.1  # From pipeline
        gap_len = 0.05  # From pipeline

        # Direction vector
        rad = math.radians(deg)
        dir_x, dir_z = math.sin(rad), math.cos(rad)

        # Calculate number of segments (exactly like pipeline)
        num_segments = int(ray_length // (dash_len + gap_len))
        segmentids = []
        commands = []

        for i in range(num_segments):
            # Position of each segment center relative to camera (exactly like pipeline)
            t = (gap_len/2) + i * (dash_len + gap_len) + (dash_len/2)
            cx = pos["x"] + dir_x * t
            cz = pos["z"] + dir_z * t
            segid = self.controller.get_unique_id()
            segmentids.append(segid)

            # Load small cube (exactly like pipeline)
            commands.append({
                "$type": "load_primitive_from_resources",
                "primitive_type": "Cube",
                "id": segid,
                "position": {"x": cx, "y": 0.02, "z": cz}
            })

            # Scale to appropriate dash_len size (exactly like pipeline)
            # Adjust scale based on direction: east/west uses x-axis along ray; north/south uses z-axis
            if deg in (90, 270):
                scale = {"x": dash_len, "y": 0.01, "z": 0.02}
            else:
                scale = {"x": 0.02, "y": 0.005, "z": dash_len}
            commands.append({
                "$type": "scale_object",
                "id": segid,
                "scale_factor": scale
            })

            # Set to red color (exactly like pipeline)
            commands.append({
                "$type": "set_color",
                "id": segid,
                "color": {"r": 1.0, "g": 0.0, "b": 0.0, "a": 1.0}
            })

        # Send all commands to TDW at once (exactly like pipeline)
        self.controller.communicate(commands)
        print(f"[DEBUG] Created dashed ray segments for {deg}° direction")
        return segmentids

    except Exception as e:
        print(f"[WARN] Failed to create dashed direction ray: {e}")
        return []
