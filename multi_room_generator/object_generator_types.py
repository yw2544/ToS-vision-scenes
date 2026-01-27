"""
Object generator shared types and constants.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any

# Standard color definitions
COLORS = {
    "red": {"r": 0.549, "g": 0.063, "b": 0.027, "a": 1.0},
    "yellow": {"r": 0.988, "g": 0.776, "b": 0.114, "a": 1.0},
    "blue": {"r": 0.0, "g": 0.0, "b": 1.0, "a": 1.0},
    "green": {"r": 0.541, "g": 0.651, "b": 0.141, "a": 1.0},
    "black": {"r": 0.0, "g": 0.0, "b": 0.0, "a": 1.0},
    "white": {"r": 1.0, "g": 1.0, "b": 1.0, "a": 1.0},
    "pink": {"r": 0.961, "g": 0.729, "b": 0.733, "a": 1.0},
    "brown":  {"r": 0.54, "g": 0.23, "b": 0.18, "a": 1.0}
}


@dataclass
class PlacedObject:
    """Data class representing a placed object in a room"""
    object_id: int
    model: str
    name: str
    pos: Dict[str, float]
    rot: Dict[str, float]
    size: Tuple[float, float]
    scale: float
    color: Optional[str]
    room_id: int  # Which room this object is in
    has_orientation: bool = False
    orientation: Optional[str] = None
    is_custom_model: bool = False  # Whether this is a custom model
    custom_config: Optional[Any] = None  # Custom model config
    model_config: Optional[Dict] = None  # Model config (builtin and custom)

    @property
    def attributes(self) -> Dict:
        """Get object attributes for JSON output"""
        attr = {
            "scale": self.scale,
            "has_orientation": self.has_orientation,
            "room_id": self.room_id
        }
        if self.color:
            # Check if custom model color (dict) or library color (string key)
            if isinstance(self.color, dict):
                # Custom models: color is an RGB dict
                attr["color"] = {
                    "name": "custom",
                    "r": self.color.get("r", 1.0),
                    "g": self.color.get("g", 1.0),
                    "b": self.color.get("b", 1.0),
                    "a": self.color.get("a", 1.0)
                }
            elif isinstance(self.color, str) and self.color in COLORS:
                # Library models: color is a string key
                color_values = COLORS[self.color]
                attr["color"] = {
                    "name": self.color,
                    "r": color_values["r"],
                    "g": color_values["g"],
                    "b": color_values["b"],
                    "a": color_values["a"]
                }
            else:
                # Fallback to default white
                attr["color"] = {
                    "name": "default",
                    "r": 1.0,
                    "g": 1.0,
                    "b": 1.0,
                    "a": 1.0
                }
        if self.has_orientation and self.orientation:
            attr["orientation"] = self.orientation
        return attr

    def get_final_position(self, base_pos: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Get final position with default_position offset applied

        Args:
            base_pos: Base position (defaults to self.pos)

        Returns:
            Final position including default_position offset
        """
        if base_pos is None:
            base_pos = self.pos

        final_pos = {
            "x": base_pos["x"],
            "y": base_pos["y"],
            "z": base_pos["z"]
        }

        # Apply default_position offset (custom and builtin)
        default_position = None

        if self.is_custom_model and self.custom_config:
            # Custom models: default_position from custom_config
            if isinstance(self.custom_config, dict):
                default_position = self.custom_config.get("default_position")
            elif hasattr(self.custom_config, 'default_position'):
                default_position = self.custom_config.default_position
        elif not self.is_custom_model and hasattr(self, 'model_config') and self.model_config:
            # Builtin models: default_position from model_config
            default_position = self.model_config.get("default_position")

        if default_position:
            final_pos["x"] += default_position.get("x", 0)
            final_pos["y"] += default_position.get("y", 0)
            final_pos["z"] += default_position.get("z", 0)

        return final_pos

    def get_final_rotation(self, base_rot: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Get final rotation with default_rotation offset applied

        Args:
            base_rot: Base rotation (defaults to self.rot)

        Returns:
            Final rotation including default_rotation offset
        """
        if base_rot is None:
            base_rot = self.rot

        final_rot = {
            "x": base_rot["x"],
            "y": base_rot["y"],
            "z": base_rot["z"]
        }

        # Apply default_rotation offset (custom and builtin)
        default_rotation = None

        if self.is_custom_model and self.custom_config:
            # Custom models: default_rotation from custom_config
            if isinstance(self.custom_config, dict):
                default_rotation = self.custom_config.get("default_rotation")
            elif hasattr(self.custom_config, 'default_rotation'):
                default_rotation = self.custom_config.default_rotation
        elif not self.is_custom_model and hasattr(self, 'model_config') and self.model_config:
            # Builtin models: default_rotation from model_config
            default_rotation = self.model_config.get("default_rotation")

        if default_rotation:
            final_rot["x"] = (final_rot["x"] + default_rotation.get("x", 0)) % 360
            final_rot["y"] = (final_rot["y"] + default_rotation.get("y", 0)) % 360
            final_rot["z"] = (final_rot["z"] + default_rotation.get("z", 0)) % 360

        return final_rot


@dataclass
class AgentInfo:
    """Data class for agent placement"""
    pos: Dict[str, float]
    room_id: int
    rotation: Dict[str, float]
