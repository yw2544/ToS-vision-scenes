import sys
import numpy as np
from typing import Tuple, Optional

# Try to import VAGEN, if not found, try adding known paths
try:
    from vagen.env.spatial.Base.tos_base.utils.room_utils import RoomGenerator, RoomPlotter
except ImportError:
    # Add VAGEN path if not in python path
    # This path is specific to the user's environment, ideally should be configured or installed
    # sys.path.insert(0, "/Users/songshe/objaverse_import/ToS2/VAGEN")
    try:
        from vagen.env.spatial.Base.tos_base.utils.room_utils import RoomGenerator, RoomPlotter
    except ImportError:
        print("Warning: VAGEN package not found. Room generation using VAGEN will fail.")
        RoomGenerator = None
        RoomPlotter = None

class VagenRoomGenerator:
    """Generator for room layouts using VAGEN."""
    
    @staticmethod
    def generate(
        room_size: Tuple[int, int],
        room_num: int,
        n_objects: int,
        topology: int = 1,
        seed: int = 42,
        save_plot: Optional[str] = None
    ) -> np.ndarray:
        """
        Generate room layout using VAGEN RoomGenerator.
        
        Args:
            room_size: (width, height) for each room
            room_num: Number of rooms
            n_objects: Objects per room
            topology: Topology type (0 or 1)
            seed: Random seed
            save_plot: Path to save plot image (optional)
            
        Returns:
            np.ndarray: Room mask
        """
        if RoomGenerator is None:
            raise ImportError("VAGEN RoomGenerator is not available.")

        np_random = np.random.default_rng(seed)
        
        print(f"[INFO] Generating {room_num} room(s) with size {room_size} and {n_objects} objects per room")
        
        room, agent = RoomGenerator.generate_multi_room(
            room_size=room_size,
            n_objects=n_objects,
            np_random=np_random,
            room_num=room_num,
            topology=topology
        )
        
        if save_plot and RoomPlotter is not None:
            print(f"[INFO] Saving visualization to {save_plot}")
            RoomPlotter.plot(room, agent, mode='img', save_path=save_plot)
        elif save_plot:
            print("[WARN] RoomPlotter not available, skipping visualization.")
        
        return room.mask
