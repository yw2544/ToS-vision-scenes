"""
Orientation Instruction Generator Module
=======================================

Generates orientation instruction images for multi-room environments:
- Extracts objects from metadata (excluding doors)
- Matches object model names to instruction images
- Adds sequence labels to each object image
- Combines all object images into a single instruction sheet
"""

import json
import os
import re
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import math

class OrientationInstructionGenerator:
    """Generates orientation instruction images for objects in the scene"""
    
    def __init__(self, instruction_images_dir: str, output_dir: Path):
        """
        Initialize orientation instruction generator
        
        Args:
            instruction_images_dir: Directory containing object instruction images
            output_dir: Output directory for generated instruction images
        """
        self.instruction_images_dir = Path(instruction_images_dir)
        self.output_dir = Path(output_dir)
        
        # Side images directory (with _left suffix) - kept for backward compatibility
        self.side_images_dir = Path(str(instruction_images_dir) + "_side")
        
        # Dual images directory (with _dual suffix)
        self.dual_images_dir = Path(str(instruction_images_dir) + "_dual")
        
        # Load available instruction images
        self.available_images = self._load_available_images()
        self.available_side_images = self._load_available_side_images()
        self.available_dual_images = self._load_available_dual_images()
        # print(f"[INFO] Loaded {len(self.available_images)} front images")
        # print(f"[INFO] Loaded {len(self.available_side_images)} side images")
        # print(f"[INFO] Loaded {len(self.available_dual_images)} dual-view images")
        
        # Font settings for labels
        self.font_size = 48  # Larger label font size
        self.label_color = (255, 0, 0)  # Red color for labels
        self.label_bg_color = (255, 255, 255, 230)  # More opaque white background
        
    def _load_available_images(self) -> Dict[str, Path]:
        """Load available instruction images and create model name mapping"""
        available_images = {}
        
        if not self.instruction_images_dir.exists():
            print(f"[WARNING] Instruction images directory not found: {self.instruction_images_dir}")
            return available_images
        
        # Scan for PNG files
        for img_file in self.instruction_images_dir.glob("*.png"):
            # Use filename without extension as the key
            model_name = img_file.stem
            available_images[model_name] = img_file
            
        return available_images
    
    def _load_available_side_images(self) -> Dict[str, Path]:
        """Load available side instruction images (with _left suffix)"""
        available_side_images = {}
        
        if not self.side_images_dir.exists():
            print(f"[WARNING] Side images directory not found: {self.side_images_dir}")
            return available_side_images
        
        # Scan for PNG files with _left suffix
        for img_file in self.side_images_dir.glob("*_left.png"):
            # Remove _left suffix to get the base model name
            model_name = img_file.stem.replace("_left", "")
            available_side_images[model_name] = img_file
            
        return available_side_images
    
    def _load_available_dual_images(self) -> Dict[str, Path]:
        """Load available dual instruction images (with _dual suffix)"""
        available_dual_images = {}
        
        if not self.dual_images_dir.exists():
            print(f"[WARNING] Dual images directory not found: {self.dual_images_dir}")
            return available_dual_images
        
        # Scan for PNG files with _dual suffix
        for img_file in self.dual_images_dir.glob("*_dual.png"):
            # Remove _dual suffix to get the base model name
            model_name = img_file.stem.replace("_dual", "")
            available_dual_images[model_name] = img_file
            
        return available_dual_images
    
    def _normalize_model_name(self, model_name: str) -> str:
        """
        Normalize model name for matching with instruction images
        
        Args:
            model_name: Original model name from metadata
            
        Returns:
            Normalized model name for matching
        """
        # Remove common prefixes and suffixes
        normalized = model_name.lower()
        
        # Remove version numbers and common suffixes
        normalized = re.sub(r'_v\d+$', '', normalized)
        normalized = re.sub(r'_\d+$', '', normalized) 
        normalized = re.sub(r'_low$', '', normalized)
        normalized = re.sub(r'_high$', '', normalized)
        normalized = re.sub(r'_medium$', '', normalized)
        
        return normalized
    
    def _extract_color_from_attributes(self, attributes: Dict[str, Any]) -> Optional[str]:
        """
        Extract color name from attributes dictionary
        
        Args:
            attributes: Object attributes dictionary
            
        Returns:
            Color name string, or None if no color found
        """
        if not attributes or "color" not in attributes:
            return None
        
        color_info = attributes["color"]
        
        # Handle string format: "color": "yellow"
        if isinstance(color_info, str):
            return color_info
        
        # Handle object format: "color": {"name": "blue", "r": 0.0, "g": 0.0, "b": 1.0, "a": 1.0}
        if isinstance(color_info, dict) and "name" in color_info:
            return color_info["name"]
        
        return None

    def _find_matching_image(self, model_name: str, color: Optional[str] = None, image_type: str = "front") -> Optional[Path]:
        """
        Find matching instruction image for a model name, considering color if available
        
        Args:
            model_name: Model name from metadata
            color: Color name from attributes (optional)
            image_type: Type of image to search for ("front", "side", "dual")
            
        Returns:
            Path to matching image file, or None if no match found
        """
        # Choose the appropriate image collection
        if image_type == "side":
            images_dict = self.available_side_images
        elif image_type == "dual":
            images_dict = self.available_dual_images
        else:
            images_dict = self.available_images
        
        # If color is available, try model_name + "_" + color first
        if color:
            color_variant = f"{model_name}_{color}"
            if color_variant in images_dict:
                return images_dict[color_variant]
            
            # Try normalized name with color
            normalized_base = self._normalize_model_name(model_name)
            normalized_color_variant = f"{normalized_base}_{color}"
            if normalized_color_variant in images_dict:
                return images_dict[normalized_color_variant]
        
        # Direct match first
        if model_name in images_dict:
            return images_dict[model_name]
        
        # Try normalized name
        normalized = self._normalize_model_name(model_name)
        if normalized in images_dict:
            return images_dict[normalized]
        
        # Try partial matching
        for available_name, image_path in images_dict.items():
            # Check if model name contains available name or vice versa
            if normalized in available_name or available_name in normalized:
                return image_path
            
            # Check for common words
            model_words = set(normalized.split('_'))
            available_words = set(available_name.split('_'))
            if model_words & available_words:  # If there's any common word
                return image_path
        
        return None
    
    def _add_label_to_image(self, image: Image.Image, label: str) -> Image.Image:
        """
        Add a label to the top-left corner of an image
        
        Args:
            image: PIL Image object
            label: Label text to add
            
        Returns:
            Image with label added
        """
        # Create a copy to avoid modifying original
        labeled_image = image.copy()
        draw = ImageDraw.Draw(labeled_image)
        
        # Try to load a font, fall back to default if not available
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", self.font_size)
        except:
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", self.font_size)
            except:
                font = ImageFont.load_default()
        
        # Get text size
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Position for label (top-left corner with some padding)
        padding = 10  # Increased padding
        label_x = padding
        label_y = padding
        
        # Draw background rectangle for better readability
        bg_padding = 8  # Increased background padding
        bg_rect = [
            label_x - bg_padding,
            label_y - bg_padding,
            label_x + text_width + bg_padding,
            label_y + text_height + bg_padding
        ]
        draw.rectangle(bg_rect, fill=self.label_bg_color)
        
        # Draw the label text
        draw.text((label_x, label_y), label, fill=self.label_color, font=font)
        
        return labeled_image
    
    def _resize_image_proportional(self, image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        """
        Resize image proportionally to fit within target size
        
        Args:
            image: PIL Image object
            target_size: (width, height) tuple for target size
            
        Returns:
            Resized image
        """
        target_width, target_height = target_size
        img_width, img_height = image.size
        
        # Calculate scaling factor to fit within target size
        scale_x = target_width / img_width
        scale_y = target_height / img_height
        scale = min(scale_x, scale_y)
        
        # Calculate new size
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        # Resize using high-quality resampling
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    def _arrange_images_in_grid(self, labeled_images: List[Image.Image], 
                                max_cols: int = 4) -> Image.Image:
        """
        Arrange multiple images in a grid layout
        
        Args:
            labeled_images: List of PIL Image objects with labels
            max_cols: Maximum number of columns in the grid
            
        Returns:
            Combined grid image
        """
        if not labeled_images:
            # Create a placeholder image if no images
            placeholder = Image.new('RGB', (400, 300), color=(240, 240, 240))
            draw = ImageDraw.Draw(placeholder)
            draw.text((150, 140), "No objects found", fill=(100, 100, 100))
            return placeholder
        
        # Determine grid dimensions
        num_images = len(labeled_images)
        cols = min(num_images, max_cols)
        rows = math.ceil(num_images / cols)
        
        # Find the maximum dimensions
        max_width = max(img.size[0] for img in labeled_images)
        max_height = max(img.size[1] for img in labeled_images)
        
        # Resize all images to the same size for consistent grid
        cell_size = (max_width, max_height)
        resized_images = []
        for img in labeled_images:
            resized = self._resize_image_proportional(img, cell_size)
            # Center the resized image in the cell
            centered = Image.new('RGB', cell_size, color=(255, 255, 255))
            paste_x = (cell_size[0] - resized.size[0]) // 2
            paste_y = (cell_size[1] - resized.size[1]) // 2
            centered.paste(resized, (paste_x, paste_y))
            resized_images.append(centered)
        
        # Create the grid image
        grid_width = cols * max_width
        grid_height = rows * max_height
        grid_image = Image.new('RGB', (grid_width, grid_height), color=(255, 255, 255))
        
        # Paste images into grid
        for i, img in enumerate(resized_images):
            row = i // cols
            col = i % cols
            x = col * max_width
            y = row * max_height
            grid_image.paste(img, (x, y))
        
        return grid_image
    
    def generate_instruction_image(self, metadata_path: Path, 
                                   output_filename: str = "orientation_instruction.png") -> Optional[Path]:
        """
        Generate orientation instruction image from metadata
        
        Args:
            metadata_path: Path to metadata JSON file
            output_filename: Name for output instruction image
            
        Returns:
            Path to generated instruction image, or None if failed
        """
        try:
            # Load metadata
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            print("[INFO] Starting orientation instruction generation...")
            
            # Extract objects from metadata (excluding doors)
            objects_data = []
            
            # Check if using direct objects array format (current format)
            if "objects" in metadata and isinstance(metadata["objects"], list):
                # Direct objects array format
                for obj_data in metadata["objects"]:
                    obj_name = obj_data.get("name", "").lower()
                    obj_model = obj_data.get("model", "").lower()
                    # Exclude doors by checking both name and model
                    if ("door" not in obj_name and "door" not in obj_model):
                        # Extract color from attributes
                        color = None
                        if "attributes" in obj_data:
                            color = self._extract_color_from_attributes(obj_data["attributes"])
                        
                        objects_data.append({
                            "object_id": obj_data["object_id"],
                            "name": obj_data["name"],
                            "model": obj_data["model"],
                            "color": color,
                            "attributes": obj_data.get("attributes", {})
                        })
            # Check if using nested objects format (from metadata_manager)
            elif "objects" in metadata and isinstance(metadata["objects"], dict) and "objects" in metadata["objects"]:
                # New format from metadata_manager
                for obj_id, obj_data in metadata["objects"]["objects"].items():
                    obj_name = obj_data.get("name", "").lower()
                    obj_model = obj_data.get("model", "").lower()
                    # Exclude doors by checking both name and model
                    if ("door" not in obj_name and "door" not in obj_model):
                        # Extract color from attributes
                        color = None
                        if "attributes" in obj_data:
                            color = self._extract_color_from_attributes(obj_data["attributes"])
                        
                        objects_data.append({
                            "object_id": obj_data["object_id"],
                            "name": obj_data["name"],
                            "model": obj_data["model"],
                            "color": color,
                            "attributes": obj_data.get("attributes", {})
                        })
            else:
                # Old format - extract from rooms
                if "rooms" in metadata:
                    for room_id, room_data in metadata["rooms"].items():
                        if "objects" in room_data:
                            for obj in room_data["objects"]:
                                obj_name = obj.get("name", "").lower()
                                obj_model = obj.get("model", "").lower()
                                # Exclude doors by checking both name and model
                                if ("door" not in obj_name and "door" not in obj_model):
                                    # Extract color from attributes
                                    color = None
                                    if "attributes" in obj:
                                        color = self._extract_color_from_attributes(obj["attributes"])
                                    
                                    objects_data.append({
                                        "object_id": obj.get("object_id", len(objects_data) + 1),
                                        "name": obj["name"],
                                        "model": obj["model"],
                                        "color": color
                                    })
            
            print(f"[INFO] Found {len(objects_data)} objects (excluding doors)")
            
            if not objects_data:
                print("[WARNING] No objects found for instruction generation")
                return None
            
            # Extract camera labels mapping from metadata
            camera_labels = {}
            if "cameras" in metadata:
                for camera in metadata["cameras"]:
                    camera_id = camera.get("id")
                    camera_label = camera.get("label")
                    if camera_id and camera_label and camera_id != "agent":
                        # Convert camera_id to int if it's a string representation of a number
                        try:
                            camera_id_int = int(camera_id)
                            camera_labels[camera_id_int] = camera_label
                        except ValueError:
                            camera_labels[camera_id] = camera_label
            
            print(f"[INFO] Found {len(camera_labels)} object camera labels")
            
            # Process each object and create labeled images
            labeled_images = []
            matched_count = 0
            
            for obj in objects_data:
                model_name = obj["model"]
                object_id = obj["object_id"]
                color = obj.get("color")

                # Get camera label for this object, fallback to object_id if not found
                label = camera_labels.get(object_id, str(object_id))

                # Check if object has orientation
                has_orientation = False
                if "attributes" in obj and "has_orientation" in obj["attributes"]:
                    has_orientation = obj["attributes"]["has_orientation"]

                color_info = f" (color:{color})" if color else ""

                # Choose image type based on has_orientation
                if has_orientation:
                    # Use dual image for objects with orientation
                    image_path = self._find_matching_image(model_name, color, image_type="dual")
                    image_type_desc = "dual-view image"
                else:
                    # Use single front image for objects without orientation
                    image_path = self._find_matching_image(model_name, color, image_type="front")
                    image_type_desc = "front image"
                
                if image_path:
                    try:
                        # Load and process the image
                        img = Image.open(image_path)
                        
                        # Convert to RGB if necessary
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        
                        # Add label with camera label (not object ID)
                        labeled_img = self._add_label_to_image(img, str(label))
                        labeled_images.append(labeled_img)
                        matched_count += 1
                        
                        print(f"[INFO] Matched {image_type_desc} for object {object_id} (label:{label}): {model_name}{color_info} -> {image_path.name}")
                        
                    except Exception as e:
                        print(f"[ERROR] Failed to process image {image_path}: {e}")
                else:
                    print(f"[WARNING] No matching {image_type_desc}: {model_name}{color_info} (object_id:{object_id})")
            
            print(f"[INFO] Matched {matched_count}/{len(objects_data)} objects")
            
            if not labeled_images:
                print("[WARNING] No object images were loaded")
                return None
            
            # Arrange images in grid
            instruction_image = self._arrange_images_in_grid(labeled_images)
            
            # Save the instruction image
            output_path = self.output_dir / output_filename
            instruction_image.save(output_path, 'PNG', quality=95)
            
            print(f"[INFO] Orientation instruction image generated: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"[ERROR] Failed to generate orientation instruction image: {e}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about available instruction images"""
        return {
            "total_front_images": len(self.available_images),
            "total_side_images": len(self.available_side_images),
            "total_dual_images": len(self.available_dual_images),
            "instruction_images_dir": str(self.instruction_images_dir),
            "side_images_dir": str(self.side_images_dir),
            "dual_images_dir": str(self.dual_images_dir),
            "available_front_models": list(self.available_images.keys()),
            "available_side_models": list(self.available_side_images.keys()),
            "available_dual_models": list(self.available_dual_images.keys())
        }
