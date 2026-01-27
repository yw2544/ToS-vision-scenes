"""Generate room layout masks using the VAGEN RoomGenerator."""

import sys
import os
from typing import Tuple, Optional
import numpy as np

vagen_root = os.environ.get("VAGEN_ROOT")
if vagen_root and vagen_root not in sys.path:
    sys.path.insert(0, vagen_root)

from vagen.env.spatial.Base.tos_base.utils.room_utils import RoomGenerator, RoomPlotter


def generate_room_layout_vagen(
    room_size: Tuple[int, int],
    room_num: int,
    n_objects: int,
    topology: int = 1,
    seed: int = 42,
    save_plot: Optional[str] = None,
    return_room: bool = False
):
    """
    Generate room layout using VAGEN RoomGenerator
    
    Args:
        room_size: Tuple (width, height) for each room, e.g., (6, 6)
        room_num: Number of rooms to generate
        n_objects: Objects per room (not total). Total = n_objects * room_num
        topology: Topology parameter (0 or 1, default: 1)
                 - 0: Main room connects to only 1 other room
                 - 1: Main room connects to 2 other rooms
        seed: Random seed for reproducibility
        save_plot: Optional path to save visualization plot
        return_room: If True, also return the Room object (for hash collision detection)
        
    Returns:
        np.ndarray: Room mask where each value represents room ID or wall/door markers
        (or tuple of (mask, room) if return_room=True)
    """
    try:
        np_random = np.random.default_rng(seed)
        
        print(f"[INFO] Generating {room_num} room(s) with size {room_size} and {n_objects} objects per room")
        print(f"[INFO]   Total objects: {n_objects * room_num}")
        
        # Generate multi-room layout with only required parameters
        room, agent = RoomGenerator.generate_multi_room(
            room_size=room_size,
            n_objects=n_objects,
            np_random=np_random,
            room_num=room_num,
            topology=topology
        )
        
        print(f"[INFO] Room layout generated successfully")
        print(f"[DEBUG] Mask shape: {room.mask.shape}")
        print(f"[DEBUG] Unique values in mask: {np.unique(room.mask)}")
        
        # Optionally save visualization
        if save_plot:
            print(f"[INFO] Saving visualization to {save_plot}")
            RoomPlotter.plot(room, agent, mode='img', save_path=save_plot)
        
        if return_room:
            return room.mask, room
        return room.mask
        
    except Exception as e:
        print(f"[ERROR] Failed to generate room layout: {e}")
        import traceback
        traceback.print_exc()
        raise


# Test/Demo function
if __name__ == "__main__":
    # Example usage
    mask = generate_room_layout_vagen(
        room_size=(6, 6),
        room_num=3,
        n_objects=4,
        topology=1,
        seed=42,
        save_plot='room_mask.png'
    )
    print(f"Room layout generated successfully")
    print(f"Mask:\n{mask}")

