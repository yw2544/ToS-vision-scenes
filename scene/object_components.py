from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any, List, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .room_analyzer import RoomAnalyzer, RoomInfo


COLORS = {
    "red": {"r": 0.549, "g": 0.063, "b": 0.027, "a": 1.0},
    "yellow": {"r": 0.988, "g": 0.776, "b": 0.114, "a": 1.0},
    "blue": {"r": 0.0, "g": 0.0, "b": 1.0, "a": 1.0},
    "green": {"r": 0.541, "g": 0.651, "b": 0.141, "a": 1.0},
    "black": {"r": 0.0, "g": 0.0, "b": 0.0, "a": 1.0},
    "white": {"r": 1.0, "g": 1.0, "b": 1.0, "a": 1.0},
    "pink": {"r": 0.961, "g": 0.729, "b": 0.733, "a": 1.0},
    "brown": {"r": 0.54, "g": 0.23, "b": 0.18, "a": 1.0},
}


@dataclass
class PlacedObject:
    object_id: int
    model: str
    name: str
    pos: Dict[str, float]
    rot: Dict[str, float]
    size: Tuple[float, float]
    scale: float
    color: Optional[str]
    room_id: int
    has_orientation: bool = False
    orientation: Optional[str] = None
    is_custom_model: bool = False
    custom_config: Optional[Any] = None
    model_config: Optional[Dict] = None

    @property
    def attributes(self) -> Dict:
        attr = {
            "scale": self.scale,
            "has_orientation": self.has_orientation,
            "room_id": self.room_id,
        }
        if self.color:
            if isinstance(self.color, dict):
                attr["color"] = {
                    "name": "custom",
                    "r": self.color.get("r", 1.0),
                    "g": self.color.get("g", 1.0),
                    "b": self.color.get("b", 1.0),
                    "a": self.color.get("a", 1.0),
                }
            elif isinstance(self.color, str) and self.color in COLORS:
                color_values = COLORS[self.color]
                attr["color"] = {
                    "name": self.color,
                    "r": color_values["r"],
                    "g": color_values["g"],
                    "b": color_values["b"],
                    "a": color_values["a"],
                }
            else:
                attr["color"] = {
                    "name": "default",
                    "r": 1.0,
                    "g": 1.0,
                    "b": 1.0,
                    "a": 1.0,
                }
        if self.has_orientation and self.orientation:
            attr["orientation"] = self.orientation
        return attr

    def get_final_position(self, base_pos: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        if base_pos is None:
            base_pos = self.pos
        final_pos = {"x": base_pos["x"], "y": base_pos["y"], "z": base_pos["z"]}
        default_position = None
        if self.is_custom_model and self.custom_config:
            if isinstance(self.custom_config, dict):
                default_position = self.custom_config.get("default_position")
            elif hasattr(self.custom_config, "default_position"):
                default_position = self.custom_config.default_position
        elif not self.is_custom_model and self.model_config:
            default_position = self.model_config.get("default_position")
        if default_position:
            final_pos["x"] += default_position.get("x", 0)
            final_pos["y"] += default_position.get("y", 0)
            final_pos["z"] += default_position.get("z", 0)
        return final_pos

    def get_final_rotation(self, base_rot: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        if base_rot is None:
            base_rot = self.rot
        final_rot = {"x": base_rot["x"], "y": base_rot["y"], "z": base_rot["z"]}
        default_rotation = None
        if self.is_custom_model and self.custom_config:
            if isinstance(self.custom_config, dict):
                default_rotation = self.custom_config.get("default_rotation")
            elif hasattr(self.custom_config, "default_rotation"):
                default_rotation = self.custom_config.default_rotation
        elif not self.is_custom_model and self.model_config:
            default_rotation = self.model_config.get("default_rotation")
        if default_rotation:
            final_rot["x"] = (final_rot["x"] + default_rotation.get("x", 0)) % 360
            final_rot["y"] = (final_rot["y"] + default_rotation.get("y", 0)) % 360
            final_rot["z"] = (final_rot["z"] + default_rotation.get("z", 0)) % 360
        return final_rot


@dataclass
class AgentInfo:
    pos: Dict[str, float]
    room_id: int
    rotation: Dict[str, float]


@dataclass(frozen=True)
class RoomPlacementCache:
    parity_buckets: Dict[Tuple[int, int], Tuple[Tuple[int, int], ...]]
    all_positions: Tuple[Tuple[int, int], ...]


def build_room_placement_cache(room_analyzer: "RoomAnalyzer", room: "RoomInfo") -> RoomPlacementCache:
    integer_positions: List[Tuple[int, int]] = []
    for row, col in room.cells:
        world_x, world_z = room_analyzer._cell_to_world(row, col)
        int_x = int(round(world_x))
        int_z = int(round(world_z))
        if room_analyzer.is_position_valid_for_placement(int_x, int_z, room.room_id):
            integer_positions.append((int_x, int_z))
    deduped_positions = list(set(integer_positions))
    from collections import defaultdict

    bucket = defaultdict(list)
    for ix, iz in deduped_positions:
        bucket[(ix & 1, iz & 1)].append((ix, iz))
    parity_buckets = {key: tuple(vals) for key, vals in bucket.items()}
    return RoomPlacementCache(parity_buckets=parity_buckets, all_positions=tuple(deduped_positions))

