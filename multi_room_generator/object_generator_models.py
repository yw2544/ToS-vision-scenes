"""
Object generator model loading and selection helpers.
"""

from typing import List, Dict, Optional, Any
from pathlib import Path

from .object_generator_types import COLORS


def _load_models_from_json(self):
    """Load all models from JSON files and group by category"""
    import json
    from pathlib import Path

    self.models_by_category = {}

    # 1. Load builtin models (models/builtin_models.json)
    if self.builtin_models_path and Path(self.builtin_models_path).exists():
        try:
            with open(self.builtin_models_path, 'r', encoding='utf-8') as f:
                builtin_models = json.load(f)

            for model_data in builtin_models:
                category = model_data.get("category", "unknown")

                # Build model record
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

                # Group by category
                if category not in self.models_by_category:
                    self.models_by_category[category] = []
                self.models_by_category[category].append(processed_model)

            builtin_count = sum(len(models) for models in self.models_by_category.values())
            print(f"[INFO] Loaded {builtin_count} builtin models from {self.builtin_models_path}")
        except Exception as e:
            print(f"[ERROR] Failed to load builtin models: {e}")

    # 2. Load custom models (models/custom_models.json)
    if self.custom_models_path and Path(self.custom_models_path).exists():
        try:
            with open(self.custom_models_path, 'r', encoding='utf-8') as f:
                custom_models = json.load(f)

            custom_count = 0
            for model_data in custom_models:
                category = model_data.get("category", model_data["name"])

                # Build model record
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

                # Group by category
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
    print(f"[INFO] Models per category: {[(cat, len(models)) for cat, models in sorted(self.models_by_category.items())]}")


def _map_color_name_to_value(self, color_name: Optional[str]) -> Optional[str]:
    """Map color names to object generator color definitions"""
    if color_name is None:
        return None

    # If already an RGB dict, return as-is
    if isinstance(color_name, dict):
        return color_name

    # If a string, map to COLORS definitions
    if isinstance(color_name, str):
        color_name_lower = color_name.lower()
        if color_name_lower in COLORS:
            return color_name_lower
    else:
            print(f"[WARN] Unknown color name: {color_name}, using None")
            return None

    return None


def get_models_by_category(self) -> Dict[str, List[Dict]]:
    """Get models grouped by category"""
    return self.models_by_category


def select_objects_by_categories(self, num_objects: int, preferred_categories: Optional[List[str]] = None, min_oriented_objects: int = 0) -> List[Dict]:
    """Select objects by category, ensuring uniqueness and minimum oriented count

    Args:
        num_objects: Number of objects to select
        preferred_categories: Preferred category list (e.g., chair for main room)
        min_oriented_objects: Minimum oriented objects (default 0, recommended 2 per room)

    Returns:
        Selected object configs
    """
    if num_objects <= 0:
        return []

    # Get available categories (exclude used) and sort for determinism
    available_categories = sorted([cat for cat in self.models_by_category.keys()
                          if cat not in self.used_categories])

    if len(available_categories) < num_objects:
        print(f"[WARN] Available categories ({len(available_categories)}) < required objects ({num_objects})")
        num_objects = len(available_categories)

    if num_objects == 0:
        return []

    # Split into oriented and non-oriented categories
    oriented_categories = []
    non_oriented_categories = []

    for cat in available_categories:
        models_in_cat = self.models_by_category[cat]
        # Check whether this category has oriented models
        has_oriented = any(model.get('has_orientation', False) for model in models_in_cat)
        if has_oriented:
            oriented_categories.append(cat)
        else:
            non_oriented_categories.append(cat)

    selected_categories = []

    # Prefer specified categories first (regardless of orientation)
    if preferred_categories:
        for pref_cat in preferred_categories:
            if pref_cat in available_categories and len(selected_categories) < num_objects:
                selected_categories.append(pref_cat)
                available_categories.remove(pref_cat)
                # Remove from corresponding list
                if pref_cat in oriented_categories:
                    oriented_categories.remove(pref_cat)
                elif pref_cat in non_oriented_categories:
                    non_oriented_categories.remove(pref_cat)

    # Ensure at least min_oriented_objects are oriented
    oriented_count = sum(1 for cat in selected_categories if cat in self.models_by_category and
                       any(m.get('has_orientation', False) for m in self.models_by_category[cat]))

    oriented_needed = max(0, min_oriented_objects - oriented_count)
    remaining_slots = num_objects - len(selected_categories)

    # Select required oriented objects first
    if oriented_needed > 0 and oriented_categories:
        num_to_select = min(oriented_needed, len(oriented_categories), remaining_slots)
        if num_to_select > 0:
            # Sort oriented_categories for deterministic order
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
            print(f"[INFO] Selected {num_to_select} oriented categories to meet minimum (min={min_oriented_objects})")

    # Randomly select remaining objects (prioritize oriented for diversity)
    if remaining_slots > 0:
        # Merge remaining categories, prioritize oriented, and sort for determinism
        remaining_categories = sorted(oriented_categories + non_oriented_categories)
        if remaining_categories:
            num_to_select = min(remaining_slots, len(remaining_categories))
            additional_categories = self.rng.choice(
                remaining_categories,
                size=num_to_select,
                replace=False
            )
            selected_categories.extend(additional_categories)

    # Randomly select one model per selected category
    selected_objects = []
    for category in selected_categories:
        models_in_category = self.models_by_category[category]
        # If category is oriented, prefer models with has_orientation=True
        oriented_models = [m for m in models_in_category if m.get('has_orientation', False)]
        if oriented_models and category in (oriented_categories or []):
            selected_model = self.rng.choice(oriented_models)
        else:
            selected_model = self.rng.choice(models_in_category)
        selected_objects.append(selected_model)

    # Count actual oriented objects selected
    actual_oriented = sum(1 for obj in selected_objects if obj.get('has_orientation', False))
    print(f"[INFO] Selected {len(selected_objects)} objects, categories: {[obj['category'] for obj in selected_objects]}")
    print(f"[INFO] Oriented objects: {actual_oriented} (minimum: {min_oriented_objects})")

    return selected_objects


def _place_object_in_room_new(self, room, obj_config: Dict) -> Any:
    """Place object using the new config format"""
    # Reuse legacy placement with new config
    return self._place_object_in_room(room, obj_config, f"new_{obj_config['category']}_{obj_config['model']}")


def get_current_object_pool(self) -> Dict[str, Any]:
    """Compatibility helper: convert models_by_category to legacy object_pool

    Note: kept for compatibility; new code should use select_objects_by_categories().
    """
    object_pool = {}
    # Sort categories for deterministic order
    for category in sorted(self.models_by_category.keys()):
        models = self.models_by_category[category]
        for i, model in enumerate(models):
            # Build unique key
            key = f"{model['source']}_{category}_{model['model']}_{i}"
            object_pool[key] = model
    return object_pool


def _debug_min_center_gap(self, room_id: int, min_gap: float = 1.1):
    """Check minimum center distance between objects in a room"""
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
            print(f"[OK][room {room_id}] Min distance {worst['d']:.3f} > {min_gap}")

        # Print all distance pairs
        print(f"[DEBUG][room {room_id}] All object pair distances:")
        for i in range(len(objs)):
            pi = objs[i].get_final_position()
            for j in range(i+1, len(objs)):
                pj = objs[j].get_final_position()
                d = ((pi["x"]-pj["x"])**2 + (pi["z"]-pj["z"])**2) ** 0.5
                status = "VIOLATION" if d <= min_gap else "OK"
                print(f"  [{status}] {objs[i].name}({pi['x']:.1f},{pi['z']:.1f}) <-> {objs[j].name}({pj['x']:.1f},{pj['z']:.1f}) = {d:.3f}")
