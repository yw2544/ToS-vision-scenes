"""
Export and accessors for ObjectGenerator.
"""

from typing import List, Dict, Optional

from .object_generator_types import PlacedObject, AgentInfo


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
    """Validate that all object categories are unique"""
    all_categories = []
    duplicate_categories = []

    for obj in self.all_objects:
        # Read category from model_config
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
        print(f"[ERROR] Duplicate categories found: {duplicate_categories}")
        print(f"[ERROR] All object categories: {all_categories}")

        # Print details for each object
        for obj in self.all_objects:
            category = obj.model_config.get('category', 'unknown') if obj.model_config else 'unknown'
            print(f"[ERROR] Object: {obj.name}, category: {category}, room: {obj.room_id}")

        return False
    else:
        print(f"[INFO] Category uniqueness validated: {len(all_categories)} unique categories")
        return True


def export_summary(self) -> Dict:
    """Export generation summary"""
    # Validate category uniqueness before exporting
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
                    "rot": obj.rot,  # Store base rotation without default_rotation
                    "attributes": obj.attributes
                }
                for obj in objs
            ]
            for room_id, objs in self.rooms_objects.items()
        }
    }
