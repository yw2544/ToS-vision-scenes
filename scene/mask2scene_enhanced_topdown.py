#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Top-down annotation and pixel-mapping utilities.
Extracted from mask2scene_enhanced.py to reduce file length.
"""

import json
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# Annotation helper functions (adapted from pipeline_partial_label_only.py)
def _compute_object_id_array(id_img: np.ndarray):
    """Convert TDW _id pass RGB image to object id int array."""
    if id_img.ndim == 3 and id_img.shape[2] >= 3:
        # TDW _id pass format: R + (G << 8) + (B << 16)
        return (
            id_img[..., 0].astype("int32") +              # R
            (id_img[..., 1].astype("int32") << 8) +       # G
            (id_img[..., 2].astype("int32") << 16)        # B
        )
    else:
        return id_img.astype("int32")


def build_seg_to_object_map_topdown(controller, image_capture, objects, agent_marker_id=None):
    """
    Display objects individually, capture top_down _id frame to get seg_id -> object_id mapping.
    For agent marker, use difference method: capture with/without agent, difference is agent location.
    """
    import numpy as np

    print("[INFO] Building segmentation ID to object ID mapping...")

    # Record original position of each object
    original_pos = {o.object_id: dict(o.pos) for o in objects}
    mapping = {}

    def hide_cmd(obj_id):
        return {"$type": "teleport_object", "id": obj_id,
                "position": {"x": 999, "y": -999, "z": 999}}

    def show_cmd(obj_id, pos):
        return {"$type": "teleport_object", "id": obj_id, "position": pos}

    # ------------ 1) Capture "background" (without all objects) ------------
    if objects:
        controller.communicate([hide_cmd(o.object_id) for o in objects])
        controller.communicate([])

    image_capture.set(frequency="once", avatar_ids=["top_down"], pass_masks=["_id"], save=False)
    controller.communicate([])
    bg_img = image_capture.get_pil_images()["top_down"]["_id"]
    bg_arr = _compute_object_id_array(np.array(bg_img))
    bg_ids = set(np.unique(bg_arr).tolist()); bg_ids.discard(0)
    print(f"[DEBUG] Background has {len(bg_ids)} non-zero seg ids")

    # ------------ 2) Detect objects individually ------------
    try:
        for idx, o in enumerate(objects, 1):
            print(f"[DEBUG] Detecting seg id for object {idx}/{len(objects)}: {o.name} ({o.object_id})")

            # Show only current object
            controller.communicate([show_cmd(o.object_id, original_pos[o.object_id])])
            controller.communicate([])

            # Capture one _id frame
            image_capture.set(frequency="once", avatar_ids=["top_down"], pass_masks=["_id"], save=False)
            controller.communicate([])
            cur_img = image_capture.get_pil_images()["top_down"]["_id"]
            cur_arr = _compute_object_id_array(np.array(cur_img))
            cur_ids = set(np.unique(cur_arr).tolist()); cur_ids.discard(0)

            # Difference with background: new IDs are the object's segmentation IDs
            new_ids = [sid for sid in cur_ids if sid not in bg_ids]
            if not new_ids:
                print(f"[WARN] {o.name}({o.object_id}) no new seg (possibly occluded/too small), skipping.")
            else:
                # For multiple segments, take the one with most pixels
                best_seg = max(new_ids, key=lambda sid: int((cur_arr == sid).sum()))
                mapping[best_seg] = o.object_id
                print(f"[MAP] seg_id {best_seg}  ← object_id {o.object_id} ({o.name})")

            # Hide it back and continue to next
            controller.communicate([hide_cmd(o.object_id)])
            controller.communicate([])

    finally:
        # Restore all objects
        if objects:
            # First restore all objects to their original positions
            controller.communicate([show_cmd(oid, pos) for oid, pos in original_pos.items()])
            controller.communicate([])

            # Enable physics and gravity for all objects to settle naturally
            physics_commands = []
            for obj in objects:
                physics_commands.extend([
                    {
                        "$type": "set_kinematic_state",
                        "id": obj.object_id,
                        "is_kinematic": True,
                        "use_gravity": False  # Explicitly enable gravity
                    },
                    # {
                    #     "$type": "apply_force_to_object",
                    #     "id": obj.object_id,
                    #     "force": {"x": 0, "y": -0.1, "z": 0}  # Small downward force
                    # }
                ])

            controller.communicate(physics_commands)
            controller.communicate([])

            # Wait for physics to settle
            import time
            time.sleep(0.5)  # Increased wait time for multiple objects to settle completely

            # Set all objects back to kinematic to prevent further movement
            kinematic_commands = []
            for obj in objects:
                kinematic_commands.append({
                    "$type": "set_kinematic_state",
                    "id": obj.object_id,
                    "is_kinematic": True,
                    "use_gravity": False  # Disable gravity to keep objects stable
                })

            controller.communicate(kinematic_commands)
            controller.communicate([])

        print("[INFO] All objects restored to original positions with gravity settlement")

    print(f"[INFO] Built segmentation mapping for {len(mapping)} objects")
    return mapping


def annotate_topdown_with_segmentation(rgbimg, id_img, objects_data, doors_data, agent_data, save_path, scene_bounds=None, seg_to_obj_map=None, door_id_mapping=None, agent_pixel_position=None, door_pixel_map=None, grid_pixel_map=None):
    """
    Annotate top-down view using segmentation data (similar to oblique annotation)
    - rgbimg: PIL RGB Image
    - id_img: PIL Image from id pass
    - objects_data: list of PlacedObj with objectid, name, attributes
    - doors_data: dict of door information
    - agent_data: agent position information
    - save_path: file path to save
    """
    try:
        rgb = rgbimg.convert("RGBA")
        id_arr = _compute_object_id_array(np.array(id_img))
        draw = ImageDraw.Draw(rgb)

        # Debug: print unique IDs in segmentation
        unique_ids = np.unique(id_arr)
        print(f"[DEBUG] Unique IDs in top-down segmentation: {unique_ids}")

        # Try to load font
        font = None
        font_size = 15
        font_paths = [
            "arialbd.ttf",
            "/System/Library/Fonts/Arial Bold.ttf",  # macOS
            "/System/Library/Fonts/Helvetica.ttc",   # macOS backup
            "/Windows/Fonts/arialbd.ttf",             # Windows
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux
        ]

        try:
            for font_path in font_paths:
                try:
                    font = ImageFont.truetype(font_path, font_size)
                    print(f"[DEBUG] Successfully loaded font: {font_path}")
                    break
                except (OSError, IOError):
                    continue

            if font is None:
                try:
                    font = ImageFont.load_default()
                    print("[DEBUG] Using default font")
                except:
                    font = None
                    print("[DEBUG] No font available")
        except ImportError:
            font = None
            print("[DEBUG] PIL ImageFont not available")

        annotated_count = 0

        # Find all non-background objects by unique IDs (excluding 0 which is background)
        non_zero_ids = unique_ids[unique_ids != 0]
        print(f"[DEBUG] Non-background object IDs found: {non_zero_ids}")

        # Step 1: Calculate 2D positions and pixel counts for all objects in segmentation
        # Filter out very large objects (walls, floors) and very small objects (debris)
        seg_objects = []
        for obj_id in non_zero_ids:
            mask = (id_arr == obj_id)
            if not mask.any():
                continue

            # Calculate object center from mask and pixel count
            ys, xs = np.nonzero(mask)
            cy, cx = int(ys.mean()), int(xs.mean())
            pixel_count = len(ys)

            # Filter logic: keep only reasonably sized objects
            # Very small (agent marker): 1-100 pixels
            # Furniture objects: 100-10000 pixels
            # Skip very large objects (walls/floors): >20000 pixels
            if 1 <= pixel_count <= 20000:
                seg_objects.append({
                    'id': obj_id,
                    'center': (cx, cy),
                    'mask': mask,
                    'pixel_count': pixel_count
                })
            else:
                print(f"[DEBUG] Filtered out seg ID {obj_id}: {pixel_count} pixels (too {'small' if pixel_count < 1 else 'large'})")

        # Sort by pixel count to identify different object types
        seg_objects.sort(key=lambda x: x['pixel_count'])
        print(f"[DEBUG] Found {len(seg_objects)} valid objects in segmentation (after filtering)")
        for i, obj in enumerate(seg_objects):
            print(f"[DEBUG] Seg object {i}: ID {obj['id']}, pixels: {obj['pixel_count']}, center: {obj['center']}")

        # Step 2: Use pre-computed agent pixel position
        agent_seg = None

        if agent_pixel_position is not None:
            agent_x, agent_y = agent_pixel_position
            print(f"[DEBUG] Using pre-computed agent position at ({agent_x}, {agent_y})")

            # Create a fake seg object for agent at this position
            agent_seg = {
                'id': -1,  # Special ID for agent
                'center': (agent_x, agent_y),
                'mask': None,  # We don't need the mask for drawing
                'pixel_count': 1  # Placeholder
            }
        else:
            print(f"[WARN] No agent pixel position provided")

        # Step 3: Match objects with their correct camera labels using mapping or fallback to proximity
        # Create a mapping from object_id to camera label based on objects_data
        object_to_label = {}
        for i, obj in enumerate(objects_data, 1):
            object_to_label[obj.object_id] = str(i)
            print(f"[DEBUG] Object {obj.name} (ID: {obj.object_id}) should have label: {i}")

        # Build seg_id -> center mapping
        id_to_center = {s['id']: s['center'] for s in seg_objects}

        # Now try to match segmentation objects to real objects
        matched_objects = []
        used_seg_indices = set()

        if seg_to_obj_map:
            print("[INFO] Using segmentation mapping for object annotation")
            matched_objects = []
            for i, obj in enumerate(objects_data, 1):
                # 1) Find seg_id for this object_id
                seg_id = next((sid for sid, oid in seg_to_obj_map.items()
                            if oid == obj.object_id), None)
                if seg_id is None:
                    print(f"[WARN] No mapping: object {obj.object_id}")
                    continue

                # 2) Compute centroid directly from the id map (no seg_objects dependency)
                mask = (id_arr == seg_id)
                if not mask.any():
                    print(f"[WARN] seg_id {seg_id} not found in topdown (possibly occluded), skip.")
                    continue

                ys, xs = np.nonzero(mask)
                cy, cx = int(ys.mean()), int(xs.mean())

                label = object_to_label.get(obj.object_id, str(i))
                matched_objects.append((obj, {"center": (cx, cy)}, label))

        else:
            # ---- Legacy nearest-neighbor logic (fallback) ----
            print("[INFO] Using spatial proximity matching for object annotation")

            # Get image dimensions for coordinate conversion
            h, w = id_arr.shape[:2]

            # Calculate scene bounds for coordinate conversion
            if scene_bounds:
                min_x, max_x, min_z, max_z = scene_bounds
                scene_width = max_x - min_x
                scene_depth = max_z - min_z
                print(f"[DEBUG] Using scene bounds: x[{min_x:.1f}, {max_x:.1f}], z[{min_z:.1f}, {max_z:.1f}]")
            else:
                print("[DEBUG] No scene bounds available, using default conversion")
                min_x, max_x, min_z, max_z = -10, 10, -10, 10
                scene_width = scene_depth = 20

            # Match objects using spatial proximity
            for obj in objects_data:
                obj_final_pos = obj.get_final_position()
                obj_x, obj_z = obj_final_pos["x"], obj_final_pos["z"]
                print(f"[DEBUG] Object {obj.name} (ID: {obj.object_id}) at 3D position ({obj_x:.1f}, {obj_z:.1f})")

                # Convert to normalized coordinates [0, 1]
                norm_x = (obj_x - min_x) / scene_width if scene_width > 0 else 0.5
                norm_z = (obj_z - min_z) / scene_depth if scene_depth > 0 else 0.5

                # Convert to image coordinates
                img_x = int(norm_x * w)
                img_y = int((1.0 - norm_z) * h)  # Y is flipped in image coordinates
                print(f"[DEBUG] Converted to 2D image position ({img_x}, {img_y})")

                # Find the closest segmentation object by distance (excluding agent and already used)
                best_match = None
                best_distance = float('inf')
                best_idx = -1

                for i, seg_obj in enumerate(seg_objects):
                    if i in used_seg_indices or seg_obj == agent_seg:
                        continue  # Skip agent and already used objects

                    seg_x, seg_y = seg_obj['center']
                    distance = ((img_x - seg_x)**2 + (img_y - seg_y)**2)**0.5

                    if distance < best_distance:
                        best_distance = distance
                        best_match = seg_obj
                        best_idx = i

                # Only match if distance is reasonable (within 200 pixels, increased for better matching)
                if best_match and best_distance < 200:
                    # Get the correct label for this object
                    correct_label = object_to_label.get(obj.object_id, str(len(matched_objects) + 1))
                    matched_objects.append((obj, best_match, correct_label))
                    used_seg_indices.add(best_idx)
                    print(f"[DEBUG] Matched object {obj.name} (ID: {obj.object_id}) with seg ID {best_match['id']} (distance: {best_distance:.1f}, label: {correct_label})")
                else:
                    print(f"[DEBUG] No close match found for object {obj.name} (ID: {obj.object_id}, best distance: {best_distance:.1f})")

        # Step 4: Draw annotations for matched objects
        for obj, seg_obj, label in matched_objects:
            cx, cy = seg_obj['center']

            # Draw object label with red background (use correct camera label)
            label_text = label

            # Position label at bottom-right of center
            r = 5
            label_x = cx + r + 8
            label_y = cy + r + 8

            # Draw semi-transparent background for label
            if font:
                bbox = draw.textbbox((label_x, label_y), label_text, font=font)
                padding = 2
                expanded_bbox = (bbox[0] - padding, bbox[1] - padding,
                               bbox[2] + padding, bbox[3] + padding)
                draw.rectangle(expanded_bbox, fill=(255, 0, 0, 255))
                draw.text((label_x, label_y), label_text, fill=(255, 255, 255, 255), font=font)
            else:
                draw.rectangle((label_x, label_y, label_x+20, label_y+10), fill=(255, 0, 0, 255))
                draw.text((label_x+2, label_y+2), label_text, fill=(255, 255, 255, 255))

            annotated_count += 1
            print(f"[DEBUG] Annotated object {obj.name} with label {label_text} at ({cx}, {cy})")

        # Step 5: Draw agent position with blue dot and red north arrow
        # Use the pre-identified agent segmentation object
        if agent_seg and agent_data:
            img_x, img_y = agent_seg['center']

            # Draw blue center dot
            agent_r = 8
            draw.ellipse((img_x-agent_r, img_y-agent_r, img_x+agent_r, img_y+agent_r),
                       fill=(0, 0, 255, 255))

            # Draw red arrow pointing north
            arrow_length = 30
            arrow_end_x = img_x
            arrow_end_y = img_y - arrow_length  # North is up (negative Y)

            # Arrow shaft
            draw.line([(img_x, img_y), (arrow_end_x, arrow_end_y)], fill=(255, 0, 0, 255), width=3)

            # Arrow head (triangle)
            head_size = 8
            arrow_head = [
                (arrow_end_x, arrow_end_y),  # tip
                (arrow_end_x - head_size//2, arrow_end_y + head_size),  # left
                (arrow_end_x + head_size//2, arrow_end_y + head_size)   # right
            ]
            draw.polygon(arrow_head, fill=(255, 0, 0, 255))

            print(f"[INFO] Agent position marked with blue dot and red north arrow at ({img_x}, {img_y}) using segmentation")
        else:
            print("[WARN] No agent segmentation object found for marking")

        # Step 6: Draw door positions (if available)
        if doors_data:
            h, w = id_arr.shape[:2]
            min_x = max_x = min_z = max_z = None
            scene_width = scene_depth = None
            if scene_bounds:
                min_x, max_x, min_z, max_z = scene_bounds
                scene_width = max_x - min_x
                scene_depth = max_z - min_z

            for door_id, door_info in doors_data.items():
                if door_pixel_map and door_id in door_pixel_map:
                    door_x, door_y = door_pixel_map[door_id]
                elif scene_bounds:
                    door_center = door_info.get('center', [0, 0])
                    norm_x = (door_center[0] - min_x) / scene_width if scene_width and scene_width > 0 else 0.5
                    norm_z = (door_center[1] - min_z) / scene_depth if scene_depth and scene_depth > 0 else 0.5
                    door_x = int(norm_x * w)
                    door_y = int((1.0 - norm_z) * h)
                else:
                    continue

                door_r = 6
                draw.rectangle((door_x-door_r, door_y-door_r, door_x+door_r, door_y+door_r),
                             fill=(255, 165, 0, 255))

                door_label = f"D{door_id}"
                if 'color' in door_info:
                    door_label = door_info['color'] + " door"
                elif door_id_mapping and door_id in door_id_mapping:
                    coord_id = door_id_mapping[door_id]
                    door_label = f"D{coord_id}"
                label_x = door_x + door_r + 5
                label_y = door_y + door_r + 5

                if font:
                    bbox = draw.textbbox((label_x, label_y), door_label, font=font)
                    padding = 2
                    expanded_bbox = (bbox[0] - padding, bbox[1] - padding,
                                   bbox[2] + padding, bbox[3] + padding)
                    draw.rectangle(expanded_bbox, fill=(255, 165, 0, 255))
                    draw.text((label_x, label_y), door_label, fill=(255, 255, 255, 255), font=font)
                else:
                    draw.rectangle((label_x, label_y, label_x+15, label_y+10), fill=(255, 165, 0, 255))
                    draw.text((label_x+2, label_y+2), door_label, fill=(255, 255, 255, 255))

                print(f"[DEBUG] Annotated door {door_id} at ({door_x}, {door_y}) using segmentation" if door_pixel_map and door_id in door_pixel_map else f"[DEBUG] Annotated door {door_id} with fallback coords")

        print(f"[INFO] Successfully annotated {annotated_count} objects in top-down view")
        rgb.save(save_path)

    except Exception as e:
        print(f"[ERROR] Failed to annotate top-down view: {e}")
        import traceback
        traceback.print_exc()
        # Save the original image as fallback
        rgbimg.save(save_path)


def capture_topdown_empty(self):
    """Capture top-down image of the empty scene (no objects/agent placed)."""
    if not self.controller or not self.image_capture:
        print("[WARN] Controller or image_capture not initialized for top_down_empty")
        return None

    self.image_capture.set(frequency="once", avatar_ids=["top_down"], pass_masks=["_img", "_id"], save=False)
    self.controller.communicate([])
    imgs = self.image_capture.get_pil_images()
    if "top_down" in imgs and "_img" in imgs["top_down"]:
        empty_path = self.output_dir / "top_down_empty.png"
        imgs["top_down"]["_img"].save(empty_path)
        print(f"[INFO] Saved top_down_empty.png (512x512) to {empty_path}")
        return imgs["top_down"].get("_id")
    else:
        print("[WARN] Failed to capture top_down_empty (no image returned)")
        return None


def build_door_segmentation_map_topdown(self):
    """Use segmentation to locate door centers in top-down view."""
    if not hasattr(self, "door_object_ids") or not hasattr(self, "door_positions"):
        return None
    tdw_door_ids = self.door_object_ids
    if not tdw_door_ids:
        return None

    # Map processed door_id -> nearest TDW door object id by position
    door_obj_map = {}
    if self.door_handler and hasattr(self.door_handler, "processed_doors"):
        for door_id, proc in self.door_handler.processed_doors.items():
            center = proc.door_info.center
            best_obj = None
            best_dist = 1e9
            for obj_id, pos in self.door_positions.items():
                dx = pos["x"] - center[0]
                dz = pos["z"] - center[1]
                dist = (dx * dx + dz * dz) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_obj = obj_id
            if best_obj is not None:
                door_obj_map[door_id] = best_obj
    if not door_obj_map:
        return None

    door_pixel_map = {}
    # Hide all doors
    hide_cmds = []
    for did in tdw_door_ids:
        hide_cmds.append({
            "$type": "teleport_object",
            "id": did,
            "position": {"x": 999, "y": -999, "z": 999}
        })
    self.controller.communicate(hide_cmds)

    # Capture background id map
    self.image_capture.set(frequency="once", avatar_ids=["top_down"], pass_masks=["_id"], save=False)
    self.controller.communicate([])
    bg_imgs = self.image_capture.get_pil_images()
    if "top_down" not in bg_imgs or "_id" not in bg_imgs["top_down"]:
        return None
    bg_arr = _compute_object_id_array(np.array(bg_imgs["top_down"]["_id"]))
    bg_ids = set(np.unique(bg_arr).tolist())

    # Process each door individually
    for logical_id, obj_id in door_obj_map.items():
        if obj_id not in self.door_positions:
            continue
        pos = self.door_positions[obj_id]
        self.controller.communicate([{
            "$type": "teleport_object",
            "id": obj_id,
            "position": pos
        }])

        self.image_capture.set(frequency="once", avatar_ids=["top_down"], pass_masks=["_id"], save=False)
        self.controller.communicate([])
        imgs = self.image_capture.get_pil_images()
        if "top_down" in imgs and "_id" in imgs["top_down"]:
            arr = _compute_object_id_array(np.array(imgs["top_down"]["_id"]))
            new_ids = set(np.unique(arr).tolist()) - bg_ids - {0}
            px = py = None
            if new_ids:
                best_seg = None
                best_count = 0
                for seg_id in new_ids:
                    count = int((arr == seg_id).sum())
                    if count > best_count:
                        best_count = count
                        best_seg = seg_id
                if best_seg is not None and best_count > 0:
                    ys, xs = np.nonzero(arr == best_seg)
                    px = float(xs.mean())
                    py = float(ys.mean())
            if px is not None and py is not None:
                door_pixel_map[logical_id] = (px, py)

        # hide again
        self.controller.communicate([{
            "$type": "teleport_object",
            "id": obj_id,
            "position": {"x": 999, "y": -999, "z": 999}
        }])

    # Restore all doors to original positions
    restore_cmds = []
    for did in tdw_door_ids:
        if did in self.door_positions:
            restore_cmds.append({
                "$type": "teleport_object",
                "id": did,
                "position": self.door_positions[did]
            })
    if restore_cmds:
        self.controller.communicate(restore_cmds)

    return door_pixel_map


def generate_topdown_pixel_map(self, id_img, mask, annotated_path=None):
    """Map every integer grid cell to its pixel center in top-down view and store into metadata."""
    import numpy as np
    print("[INFO] Building top-down grid pixel map...")
    bg_arr = _compute_object_id_array(np.array(id_img))
    bg_ids = set(np.unique(bg_arr).tolist())
    rows, cols = len(mask), len(mask[0])

    offset_x = -(rows // 2) * self.cell_size
    offset_z = -(cols // 2) * self.cell_size
    mapping = {}
    total_cells = rows * cols
    processed = 0

    # Debug mode disabled for production (set to 0 to disable, or >0 to enable)
    debug_save_count = 0
    debug_saved = 0
    debug_dir = None
    if debug_save_count > 0:
        debug_dir = self.output_dir / "grid_point_debug"
        debug_dir.mkdir(exist_ok=True)
        print(f"[DEBUG] Debug images will be saved to: {debug_dir}")

    for r in range(rows):
        for c in range(cols):
            if mask[r][c] < 0:
                continue
            cube_id = self.controller.get_unique_id()
            # Calculate world position using same formula as x_center/z_center in mask2scene.py
            # x = (r + offset_x_int) * cell_size where offset_x_int = -(rows // 2)
            offset_x_int = -(rows // 2)
            offset_z_int = -(cols // 2)
            wx = (r + offset_x_int) * self.cell_size
            wz = (c + offset_z_int) * self.cell_size

            # Save image BEFORE placing cube (if debug mode enabled)
            if debug_save_count > 0 and debug_saved < debug_save_count:
                self.image_capture.set(frequency="once", avatar_ids=["top_down"], pass_masks=["_img"], save=False)
                self.controller.communicate([])
                before_imgs = self.image_capture.get_pil_images()
                if "top_down" in before_imgs and "_img" in before_imgs["top_down"]:
                    before_img = before_imgs["top_down"]["_img"]
                    crop_size = 100
                    img_w, img_h = before_img.size
                    center_x = img_w // 2
                    center_y = img_h // 2
                    crop_x0 = max(0, center_x - crop_size // 2)
                    crop_y0 = max(0, center_y - crop_size // 2)
                    crop_x1 = min(img_w, center_x + crop_size // 2)
                    crop_y1 = min(img_h, center_y + crop_size // 2)
                    before_crop = before_img.crop((crop_x0, crop_y0, crop_x1, crop_y1))
                    before_crop.save(debug_dir / f"r{r:02d}_c{c:02d}_before.png")
                    print(f"[DEBUG] Saved before image for ({r},{c}) at world ({wx:.2f},{wz:.2f})")

            cmds = [
                {
                    "$type": "load_primitive_from_resources",
                    "primitive_type": "Cube",
                    "id": cube_id,
                    "position": {"x": wx, "y": 0.05, "z": wz},
                    "rotation": {"x": 0, "y": 0, "z": 0}
                },
                {
                    "$type": "scale_object",
                    "id": cube_id,
                    "scale_factor": {"x": 0.08, "y": 0.08, "z": 0.08}
                },
                {
                    "$type": "set_kinematic_state",
                    "id": cube_id,
                    "is_kinematic": True,
                    "use_gravity": False
                },
                {
                    "$type": "set_color",
                    "id": cube_id,
                    "color": {"r": 0.0, "g": 0.0, "b": 1.0, "a": 1.0}
                }
            ]
            self.controller.communicate(cmds)

            self.image_capture.set(frequency="once", avatar_ids=["top_down"], pass_masks=["_img", "_id"], save=False)
            self.controller.communicate([])
            imgs = self.image_capture.get_pil_images()

            # Save image AFTER placing cube (if debug mode enabled)
            if debug_save_count > 0 and debug_saved < debug_save_count and "top_down" in imgs and "_img" in imgs["top_down"]:
                after_img = imgs["top_down"]["_img"]
                px = None
                py = None
                if "_id" in imgs["top_down"]:
                    arr = _compute_object_id_array(np.array(imgs["top_down"]["_id"]))
                    new_ids = set(np.unique(arr).tolist()) - bg_ids - {0}
                    if new_ids:
                        best_seg = max(new_ids, key=lambda sid: int((arr == sid).sum()))
                        ys, xs = np.nonzero(arr == best_seg)
                        px = float(xs.mean())
                        py = float(ys.mean())

                        # Crop around detected pixel position
                        crop_size = 100
                        img_w, img_h = after_img.size
                        crop_x0 = max(0, int(px - crop_size // 2))
                        crop_y0 = max(0, int(py - crop_size // 2))
                        crop_x1 = min(img_w, int(px + crop_size // 2))
                        crop_y1 = min(img_h, int(py + crop_size // 2))
                        after_crop = after_img.crop((crop_x0, crop_y0, crop_x1, crop_y1))
                        after_crop.save(debug_dir / f"r{r:02d}_c{c:02d}_after.png")

                        # Save larger crop with red cross marking detected position
                        large_crop_size = 200
                        large_crop_x0 = max(0, int(px - large_crop_size // 2))
                        large_crop_y0 = max(0, int(py - large_crop_size // 2))
                        large_crop_x1 = min(img_w, int(px + large_crop_size // 2))
                        large_crop_y1 = min(img_h, int(py + large_crop_size // 2))
                        large_crop = after_img.crop((large_crop_x0, large_crop_y0, large_crop_x1, large_crop_y1))

                        # Draw red cross at detected position
                        from PIL import ImageDraw
                        draw = ImageDraw.Draw(large_crop)
                        cross_x = px - large_crop_x0
                        cross_y = py - large_crop_y0
                        cross_size = 10
                        draw.line([(cross_x - cross_size, cross_y), (cross_x + cross_size, cross_y)], fill=(255, 0, 0), width=3)
                        draw.line([(cross_x, cross_y - cross_size), (cross_x, cross_y + cross_size)], fill=(255, 0, 0), width=3)
                        large_crop.save(debug_dir / f"r{r:02d}_c{c:02d}_after_marked.png")

                        # Save debug info
                        with open(debug_dir / f"r{r:02d}_c{c:02d}_info.txt", "w") as f:
                            f.write(f"Grid: row={r}, col={c}\n")
                            f.write(f"World: x={wx:.6f}, z={wz:.6f}\n")
                            f.write(f"Pixel: x={px:.6f}, y={py:.6f}\n")
                            f.write(f"Offset (int): offset_x={offset_x_int}, offset_z={offset_z_int}\n")
                            f.write(f"Cell size: {self.cell_size}\n")
                            f.write(f"Calculation: wx = ({r} + {offset_x_int}) * {self.cell_size} = {wx:.6f}\n")
                            f.write(f"Calculation: wz = ({c} + {offset_z_int}) * {self.cell_size} = {wz:.6f}\n")
                            f.write(f"Note: Using same offset calculation as mask2scene.py x_center/z_center\n")

                        debug_saved += 1
                        print(f"[DEBUG] Saved after image for ({r},{c}) at pixel ({px:.2f},{py:.2f}), world ({wx:.2f},{wz:.2f})")

            if "top_down" in imgs and "_id" in imgs["top_down"]:
                arr = _compute_object_id_array(np.array(imgs["top_down"]["_id"]))
                new_ids = set(np.unique(arr).tolist()) - bg_ids - {0}
                px = None
                py = None
                if new_ids:
                    # Choose largest new segment
                    best_seg = None
                    best_count = 0
                    for seg_id in new_ids:
                        count = int((arr == seg_id).sum())
                        if count > best_count:
                            best_count = count
                            best_seg = seg_id
                    if best_seg is not None and best_count > 0:
                        ys, xs = np.nonzero(arr == best_seg)
                        px = float(xs.mean())
                        py = float(ys.mean())

                if px is not None and py is not None:
                    mapping[(r, c)] = (px, py)
                else:
                    # Log warning for failed detection
                    if processed < 50 or processed % 100 == 0:  # Only log first 50 or every 100th to avoid spam
                        print(f"[WARN] Failed to detect pixel position for grid ({r},{c}) at world ({wx:.2f},{wz:.2f})")
                        if not new_ids:
                            print(f"       Reason: No new segmentation IDs found (cube may be too small or occluded)")
                        elif best_seg is None:
                            print(f"       Reason: Found {len(new_ids)} new IDs but none had pixels")
            else:
                if processed < 50 or processed % 100 == 0:
                    print(f"[WARN] Failed to capture _id image for grid ({r},{c}) at world ({wx:.2f},{wz:.2f})")

            self.controller.communicate([{"$type": "destroy_object", "id": cube_id}])
            processed += 1
            if processed % 200 == 0:
                print(f"[INFO] Grid mapping progress: {processed}/{total_cells}, mapped: {len(mapping)}")

    # Calculate offset for consistent world coordinate calculation
    offset_x_int = -(rows // 2)
    offset_z_int = -(cols // 2)

    # Count total valid grid cells (excluding walls/invalid)
    total_valid_cells = sum(1 for r in range(rows) for c in range(cols) if mask[r][c] >= 0)
    mapped_count = len(mapping)
    skipped_count = total_valid_cells - mapped_count

    print(f"[INFO] Grid mapping summary:")
    print(f"  Total valid grid cells: {total_valid_cells}")
    print(f"  Successfully mapped: {mapped_count}")
    print(f"  Failed/Skipped: {skipped_count}")
    if skipped_count > 0:
        print(f"  ⚠️  Warning: {skipped_count} grid points were not mapped (may be due to occlusion or detection failure)")

    mapping_json = [{
        "row": r,
        "col": c,
        "world": {"x": (r + offset_x_int) * self.cell_size, "z": (c + offset_z_int) * self.cell_size},
        "pixel": {"x": px, "y": py}
    } for (r, c), (px, py) in mapping.items()]

    # cache
    self.topdown_pixel_map = {(item["row"], item["col"]): (item["pixel"]["x"], item["pixel"]["y"]) for item in mapping_json}

    # write into metadata
    if self._should_save_topdown_map():
        meta_path = self.output_dir / "meta_data.json"
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                meta["topdown_map"] = {
                    "rows": rows,
                    "cols": cols,
                    "cell_size": self.cell_size,
                    "mapping": mapping_json
                }
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2, ensure_ascii=False)
                print("[INFO] topdown_map written into metadata")
            except Exception as e:
                print(f"[WARN] Failed to write topdown_map into metadata: {e}")

    self.image_capture.set(frequency="never")
    print(f"[INFO] Completed top-down pixel map generation")
