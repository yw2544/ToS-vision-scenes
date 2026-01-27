"""
Room Analyzer Module
===================

Analyzes mask data to identify:
- Individual rooms and their boundaries
- Door positions and connections between rooms
- Room sizes and usable areas for object placement
"""

from typing import List, Dict, Tuple, Optional, Set
import numpy as np
from dataclasses import dataclass
from collections import defaultdict
import json

def is_room(v: int) -> bool: 
    """Check if value represents a room (1-99)"""
    return 1 <= v <= 99

def is_door(v: int) -> bool: 
    """Check if value represents a door (>=100)"""
    return v >= 100

@dataclass
class RoomInfo:
    """Information about a single room"""
    room_id: int
    name: str
    cells: List[Tuple[int, int]]  # List of (row, col) coordinates
    center: Tuple[float, float]   # Center position in world coordinates
    area: int                     # Number of cells
    bounds: Tuple[int, int, int, int]  # min_row, max_row, min_col, max_col
    usable_area: float           # Area available for object placement (considering margins)
    
    @property
    def width(self) -> int:
        """Room width in cells"""
        return self.bounds[3] - self.bounds[2] + 1
    
    @property 
    def height(self) -> int:
        """Room height in cells"""
        return self.bounds[1] - self.bounds[0] + 1

@dataclass
class DoorInfo:
    """Information about a door"""
    door_id: int
    name: str
    cells: List[Tuple[int, int]]  # List of (row, col) coordinates
    center: Tuple[float, float]   # Center position in world coordinates
    connected_rooms: List[int]    # Room IDs that this door connects
    orientation: str              # "horizontal" or "vertical"
    width: int                    # Door width in cells
    color: Optional[Dict[str, float]] = None  # Door color {r, g, b, a}

class RoomAnalyzer:
    """Analyzes mask data to extract room and door information"""
    
    def __init__(self, mask: List[List[int]], cell_size: float = 1.0):
        """
        Initialize room analyzer
        
        Args:
            mask: 2D mask array where 1-99 = rooms, >=100 = doors, others = walls/void
            cell_size: Size of each cell in world units
        """
        self.mask = mask
        self.rows = len(mask) if mask is not None and len(mask) > 0 else 0
        self.cols = len(mask[0]) if mask is not None and len(mask) > 0 else 0
        self.cell_size = cell_size
        
        # Analysis results
        self.rooms: Dict[int, RoomInfo] = {}
        self.doors: Dict[int, DoorInfo] = {}
        self.room_connections: Dict[int, Set[int]] = defaultdict(set)  # room_id -> connected room_ids
        
        # Perform analysis
        self._analyze_mask()
    
    def _analyze_mask(self):
        """Main analysis method"""
        print("[INFO] Starting mask analysis...")
        
        # Find all unique room and door IDs
        unique_values = set()
        for row in self.mask:
            unique_values.update(row)
        
        room_ids = [v for v in unique_values if is_room(v)]
        door_ids = [v for v in unique_values if is_door(v)]
        
        print(f"[INFO] Found {len(room_ids)} rooms: {sorted(room_ids)}")
        print(f"[INFO] Found {len(door_ids)} doors: {sorted(door_ids)}")
        
        # Analyze rooms
        for room_id in room_ids:
            self._analyze_room(room_id)
        
        # Analyze doors
        for door_id in door_ids:
            self._analyze_door(door_id)
        
        print(f"[INFO] Analysis complete: {len(self.rooms)} rooms, {len(self.doors)} doors")
    
    def _analyze_room(self, room_id: int):
        """Analyze a specific room"""
        # Find all cells belonging to this room
        cells = []
        for r in range(self.rows):
            for c in range(self.cols):
                if self.mask[r][c] == room_id:
                    cells.append((r, c))
        
        if not cells:
            return
        
        # Calculate bounds
        rows = [cell[0] for cell in cells]
        cols = [cell[1] for cell in cells]
        bounds = (min(rows), max(rows), min(cols), max(cols))
        
        # Calculate center in world coordinates
        center_row = sum(rows) / len(rows)
        center_col = sum(cols) / len(cols)
        center_world = self._cell_to_world(center_row, center_col)
        
        # Calculate usable area (considering wall margins)
        area = len(cells)
        # Estimate usable area as 60-80% of total area to account for walls and margins
        room_width = bounds[3] - bounds[2] + 1
        room_height = bounds[1] - bounds[0] + 1
        usable_area = min(area * 0.7, (room_width - 2) * (room_height - 2) * 0.8)
        usable_area = max(0, usable_area)  # Ensure non-negative
        
        room_info = RoomInfo(
            room_id=room_id,
            name=f"room_{room_id}",
            cells=cells,
            center=center_world,
            area=area,
            bounds=bounds,
            usable_area=usable_area
        )
        
        self.rooms[room_id] = room_info
        print(f"[INFO] Room {room_id}: {area} cells, center at {center_world}, usable area: {usable_area:.1f}")
    
    def _analyze_door(self, door_id: int):
        """Analyze a specific door, separating disconnected door segments with same ID"""
        # Find all cells belonging to this door
        all_door_cells = []
        for r in range(self.rows):
            for c in range(self.cols):
                if self.mask[r][c] == door_id:
                    all_door_cells.append((r, c))
        
        if not all_door_cells:
            return
        
        # Group connected door cells (same ID but different physical doors)
        door_segments = self._group_connected_cells(all_door_cells)
        
        # Process each door segment as a separate door
        for segment_idx, cells in enumerate(door_segments):
            # Create unique door ID for each segment
            if len(door_segments) == 1:
                # Only one segment, use original door_id
                unique_door_id = door_id
            else:
                # Multiple segments, create IDs based on position
                center_r = sum(cell[0] for cell in cells) / len(cells)
                center_c = sum(cell[1] for cell in cells) / len(cells)
                # Create base ID: original_id * 1000 + position_hash
                position_hash = int(abs(hash((round(center_r, 1), round(center_c, 1)))) % 1000)
                unique_door_id = door_id * 1000 + position_hash

                # Only adjust when a collision would drop a door segment.
                if unique_door_id in self.doors:
                    # Deterministic linear probe based on segment index.
                    base_id = door_id * 1000
                    candidate = base_id + 100 + segment_idx
                    while candidate in self.doors:
                        candidate += 1
                    print(
                        f"[WARN] Door ID collision for {door_id} segment {segment_idx}; "
                        f"using fallback id {candidate} instead of {unique_door_id}"
                    )
                    unique_door_id = candidate
            
            # Calculate center in world coordinates
            rows = [cell[0] for cell in cells]
            cols = [cell[1] for cell in cells]
            center_row = sum(rows) / len(rows)
            center_col = sum(cols) / len(cols)
            center_world = self._cell_to_world(center_row, center_col)
            
            # Determine orientation and width
            orientation, width = self._determine_door_orientation(cells)
            
            # Find connected rooms
            connected_rooms = self._find_connected_rooms(cells)
            
            # Update room connections
            for i, room1 in enumerate(connected_rooms):
                for room2 in connected_rooms[i+1:]:
                    self.room_connections[room1].add(room2)
                    self.room_connections[room2].add(room1)
            
            door_info = DoorInfo(
                door_id=unique_door_id,
                name=f"door_{door_id}_{segment_idx}" if len(door_segments) > 1 else f"door_{door_id}",
                cells=cells,
                center=center_world,
                connected_rooms=connected_rooms,
                orientation=orientation,
                width=width
            )
            
            self.doors[unique_door_id] = door_info
            print(f"[INFO] Door {unique_door_id}: {orientation}, width {width}, connects rooms {connected_rooms}")
    
    def _determine_door_orientation(self, cells: List[Tuple[int, int]]) -> Tuple[str, int]:
        """Determine if door is horizontal or vertical and its width"""
        if not cells:
            return "horizontal", 1
        
        rows = [cell[0] for cell in cells]
        cols = [cell[1] for cell in cells]
        
        row_span = max(rows) - min(rows) + 1
        col_span = max(cols) - min(cols) + 1
        
        # If door spans multiple cells, use span to determine orientation
        if row_span > col_span:
            return "horizontal", row_span  # Door spans rows = horizontal
        elif col_span > row_span:
            return "vertical", col_span  # Door spans cols = vertical
        else:
            # For single-cell doors, determine orientation by checking room positions
            connected_rooms = self._find_connected_rooms(cells)
            if len(connected_rooms) >= 2:
                # Find representative positions for each connected room
                room_positions = []
                for room_id in connected_rooms[:2]:  # Check first two rooms
                    # Find a representative cell for this room
                    found = False
                    for r in range(self.rows):
                        for c in range(self.cols):
                            if self.mask[r][c] == room_id:
                                room_positions.append((r, c))
                                found = True
                                break
                        if found:
                            break
                
                if len(room_positions) >= 2:
                    # Calculate relative positions
                    pos1, pos2 = room_positions[0], room_positions[1]
                    
                    # Check if rooms are separated more horizontally or vertically
                    row_diff = abs(pos1[0] - pos2[0])
                    col_diff = abs(pos1[1] - pos2[1])
                    
                    if col_diff > row_diff:
                        return "horizontal", 1  # Rooms are left-right (different cols), door is horizontal
                    else:
                        return "vertical", 1  # Rooms are up-down (different rows), door is vertical
            
            # Fallback to horizontal for single-cell doors
            return "horizontal", 1
    
    def _group_connected_cells(self, cells: List[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
        """Group cells into connected components (adjacent cells form one group)"""
        if not cells:
            return []
        
        cell_set = set(cells)
        visited = set()
        groups = []
        
        def get_neighbors(r, c):
            """Get adjacent cells (4-connectivity)"""
            return [(r+dr, c+dc) for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]]
        
        def dfs(start_cell):
            """Depth-first search to find connected component"""
            stack = [start_cell]
            component = []
            
            while stack:
                r, c = stack.pop()
                if (r, c) in visited:
                    continue
                
                visited.add((r, c))
                component.append((r, c))
                
                # Add unvisited neighbors that are also door cells
                for nr, nc in get_neighbors(r, c):
                    if (nr, nc) in cell_set and (nr, nc) not in visited:
                        stack.append((nr, nc))
            
            return component
        
        # Find all connected components
        for cell in cells:
            if cell not in visited:
                component = dfs(cell)
                if component:
                    groups.append(component)
        
        return groups
    
    def _find_connected_rooms(self, door_cells: List[Tuple[int, int]]) -> List[int]:
        """Find which rooms this door connects"""
        connected_rooms = set()
        
        # Check adjacent cells to door cells
        for r, c in door_cells:
            # Check 4-connected neighbors
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    neighbor_val = self.mask[nr][nc]
                    if is_room(neighbor_val):
                        connected_rooms.add(neighbor_val)
        
        return list(connected_rooms)
    
    def _cell_to_world(self, row: float, col: float) -> Tuple[float, float]:
        """Convert mask coordinates to world coordinates
        
        New mapping: x = row, z = col (then apply offset)
        
        mask[row][col] where:
        - mask[row][col] maps to world coordinate (row, col) + offset
        - row corresponds to X axis 
        - col corresponds to Z axis
        
        Offset ensures integer values and centers the world appropriately.
        """
        # Calculate INTEGER offset to center the world
        # For mask of size (rows x cols), we want center at (0, 0)
        offset_x = -(self.rows // 2)  # offset for row (X axis)
        offset_z = -(self.cols // 2)  # offset for col (Z axis)
        
        # Direct mapping: mask(row,col) + offset = world(x,z)
        # mask row → world X, mask col → world Z
        x = (row + offset_x) * self.cell_size
        z = (col + offset_z) * self.cell_size
        
        return (x, z)
    
    def get_offset(self) -> Tuple[int, int]:
        """Get the integer offset used for coordinate conversion
        Returns (offset_x, offset_z) where x=row, z=col
        """
        offset_x = -(self.rows // 2)  # offset for row (X axis)
        offset_z = -(self.cols // 2)  # offset for col (Z axis)
        return (offset_x, offset_z)
    
    def get_largest_room(self) -> Optional[RoomInfo]:
        """Get the largest room (for main agent placement)"""
        if not self.rooms:
            return None
        return max(self.rooms.values(), key=lambda r: r.usable_area)
    
    def get_max_objects_for_room(self, room_id: int) -> int:
        """Calculate maximum number of objects that can fit in a room"""
        if room_id not in self.rooms:
            return 0
        
        room = self.rooms[room_id]
        usable_area = room.usable_area
        
        # New calculation: each object needs 1.5 cells of space (including clearance)
        space_per_object = 1.5
        
        # Calculate theoretical maximum based on area
        max_objects_by_area = int(usable_area / space_per_object)
        
        # Consider room dimensions to avoid overcrowding
        min_dimension = min(room.width, room.height)
        
        # Minimum usable area threshold
        if usable_area < 1.5:  # Too small for any object
            return 0
        
        # Dimension-based constraints (avoid linear arrangements in narrow rooms)
        if min_dimension < 2:  # Very narrow room
            max_objects_by_dimension = 1
        elif min_dimension < 3:  # Narrow room
            max_objects_by_dimension = 2
        else:  # Normal room
            max_objects_by_dimension = max_objects_by_area
        
        # Take the minimum of area-based and dimension-based limits
        max_objects = min(max_objects_by_area, max_objects_by_dimension)
        
        # Cap at reasonable maximum to avoid extreme cases  
        max_objects = min(max_objects, 12)  # Increased from 8 to 12
        
        return max_objects
    
    def get_room_bounds_world(self, room_id: int) -> Optional[Tuple[float, float, float, float]]:
        """Get room bounds in world coordinates (min_x, max_x, min_z, max_z)"""
        if room_id not in self.rooms:
            return None
        
        room = self.rooms[room_id]
        min_row, max_row, min_col, max_col = room.bounds
        
        # Convert corners to world coordinates
        min_x, max_z = self._cell_to_world(min_row, min_col)
        max_x, min_z = self._cell_to_world(max_row, max_col)
        
        # Ensure proper ordering since coordinate system is centered
        if min_x > max_x:
            min_x, max_x = max_x, min_x
        if min_z > max_z:
            min_z, max_z = max_z, min_z
        
        # Add cell_size/2 to account for cell centers vs boundaries
        margin = self.cell_size / 2
        return (min_x - margin, max_x + margin, min_z - margin, max_z + margin)
    
    def is_position_in_room(self, x: float, z: float, room_id: int) -> bool:
        """Check if a world position is inside a specific room (precise check)"""
        # Convert world coordinates back to cell coordinates
        row, col = self._world_to_cell(x, z)
        
        # Check if cell coordinates are valid
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return False
        
        # Check if the cell actually contains the specified room
        cell_value = self.mask[row][col]
        return cell_value == room_id
    
    def _world_to_cell(self, x: float, z: float) -> Tuple[int, int]:
        """Convert world coordinates back to mask coordinates
        Where x=row, z=col after applying offset
        """
        # Calculate offset (same as in _cell_to_world)
        offset_x = -(self.rows // 2)  # offset for row (X axis)
        offset_z = -(self.cols // 2)  # offset for col (Z axis)
        
        # Reverse conversion: world → mask
        # world(x,z) - offset = mask(row,col) where x=row, z=col
        row = round(x / self.cell_size - offset_x)
        col = round(z / self.cell_size - offset_z)
        return (row, col)
    
    def is_position_valid_for_placement(self, x: float, z: float, room_id: int) -> bool:
        """Check if a position is valid for object/agent placement (not on walls/doors)"""
        row, col = self._world_to_cell(x, z)
        
        # Check if cell coordinates are valid
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return False
        
        # Check if the cell is exactly the target room (not wall, door, or other room)
        cell_value = self.mask[row][col]
        if cell_value != room_id:
            return False
        
        # Additional check: make sure this cell is not a door
        if is_door(cell_value):
            return False
        
        # Relaxed safety check: only reject if directly on walls/doors or void
        # Allow positions near walls but not ON walls - this is much more reasonable
        # The real safety distance should be handled by the min_distance parameter in object placement
        
        return True
    
    def get_room_at_position(self, x: float, z: float) -> Optional[int]:
        """Get room ID at a world position"""
        for room_id, room in self.rooms.items():
            if self.is_position_in_room(x, z, room_id):
                return room_id
        return None
    
    def export_summary(self) -> Dict:
        """Export analysis summary for debugging/logging"""
        return {
            "mask_size": [self.rows, self.cols],
            "cell_size": self.cell_size,
            "rooms": {
                room_id: {
                    "name": room.name,
                    "area": room.area,
                    "usable_area": room.usable_area,
                    "center": room.center,
                    "bounds": room.bounds,
                    "max_objects": self.get_max_objects_for_room(room_id)
                }
                for room_id, room in self.rooms.items()
            },
            "doors": {
                door_id: {
                    "name": door.name,
                    "center": door.center,
                    "connected_rooms": door.connected_rooms,
                    "orientation": door.orientation,
                    "width": door.width
                }
                for door_id, door in self.doors.items()
            },
            "room_connections": {
                room_id: list(connections) 
                for room_id, connections in self.room_connections.items()
            }
        } 