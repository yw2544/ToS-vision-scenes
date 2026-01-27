"""Generate bwd_loc task viewpoints and camera metadata."""

import sys
import os
import json
import numpy as np
import math
from typing import List, Dict, Tuple, Optional
from pathlib import Path

# Add VAGEN to path
vagen_root = os.environ.get("VAGEN_ROOT")
if vagen_root and vagen_root not in sys.path:
    sys.path.insert(0, vagen_root)

from vagen.env.spatial.Base.tos_base.utils.room_utils import initialize_room_from_json
from vagen.env.spatial.Base.tos_base.evaluation.task_types import EvalTaskType


# ============================================================================
# Coordinate and Direction Conversion Utilities
# ============================================================================

def ori_to_yaw(ori: np.ndarray) -> float:
    """Convert VAGEN orientation vector [dx, dz] to TDW yaw angle in degrees.
    
    Args:
        ori: Orientation vector [dx, dz] from VAGEN
        
    Returns:
        Yaw angle in degrees (0-360)
    """
    yaw = math.degrees(math.atan2(ori[0], ori[1]))
    return yaw % 360


def mask_pos_to_tdw(pos_mask: np.ndarray, offset: Tuple[int, int]) -> np.ndarray:
    """Convert mask coordinates to TDW world coordinates."""
    return pos_mask - np.array(offset)


def tdw_pos_to_mask(pos_tdw: np.ndarray, offset: Tuple[int, int]) -> np.ndarray:
    """Convert TDW world coordinates back to mask/global coordinates."""
    return pos_tdw + np.array(offset)


def ori_to_name(ori: np.ndarray) -> str:
    """Map VAGEN orientation vector to cardinal string."""
    mapping = {(0, 1): "north", (1, 0): "east", (0, -1): "south", (-1, 0): "west"}
    return mapping.get(tuple(int(x) for x in ori), "north")


def extract_bwd_loc_viewpoint(task, offset: tuple) -> Optional[Dict]:
    """Extract viewpoint from BackwardLocEvaluationTask.
    
    The viewpoint is at the agent's position after generating the question.
    
    Args:
        task: Already instantiated VAGEN task object (will call generate_question)
        offset: Coordinate offset
        
    Returns:
        Viewpoint dict or None
    """
    try:
        task.generate_question()
        
        # Agent has been moved to a new position
        pos_mask = task.agent.pos.copy()
        ori_mask = task.agent.ori.copy()
        pos_tdw = mask_pos_to_tdw(pos_mask, offset)
        yaw = ori_to_yaw(ori_mask)
        dir_name = ori_to_name(ori_mask)
        coord_name = f"{int(pos_mask[0])}_{int(pos_mask[1])}"
        image_name = f"{coord_name}_facing_{dir_name}"
        
        return {
            'task_type': 'bwd_loc',
            'pos_tdw': pos_tdw,
            'yaw': yaw,
            'task_id': task.eval_data.id,
            'extraction_method': 'from_agent',
            'agent_pos_mask': pos_mask,
            'agent_ori': ori_mask,
            'direction_label': dir_name,
            'cam_id': coord_name,
            'image_name': image_name
        }
    except Exception as e:
        print(f"[ERROR] Failed to extract bwd_loc viewpoint: {e}")
    return None


# ============================================================================
# Main Viewpoint Generation Function
# ============================================================================

def generate_task_viewpoints(metadata: Dict, run_seed: int = None,
                            num_questions_per_task: int = 3) -> List[Dict]:
    """Generate bwd_loc task viewpoints from metadata.

    This function matches VAGEN's generation behavior:
    - Create one bwd_loc task object (with run_seed)
    - Call generate_question() multiple times on the same task object
    - np_random state accumulates between calls

    Args:
        metadata: Scene metadata dict (must contain objects, offset, etc.)
        run_seed: Seed for reproducibility (defaults to metadata's run_seed or seed)
        num_questions_per_task: Number of questions/viewpoints (default: 3)

    Returns:
        List of viewpoint dicts for bwd_loc.
    """
    print("\n" + "=" * 70)
    print("Generating bwd_loc viewpoints (VAGEN strategy)")
    print("=" * 70)
    
    # Create Room and Agent from metadata
    try:
        room, agent = initialize_room_from_json(metadata)
    except Exception as e:
        print(f"[ERROR] Failed to initialize room from metadata: {e}")
        return []
    
    offset = tuple(metadata.get('offset', [0, 0]))

    # 与 VAGEN 保持一致：只要 image_dir 非空，题干中就会使用 "<image>" 占位符
    image_dir = (
        metadata.get('image_dir')
        or metadata.get('task_image_dir')
        or metadata.get('task_images_dir')
        or "_placeholder_image_dir"
    )
    
    # Use run_seed if provided, otherwise fallback
    if run_seed is None:
        run_seed = metadata.get('run_seed', metadata.get('seed', 42))
    
    print(f"  Offset: {offset}")
    print(f"  Run seed: {run_seed}")
    print(f"  Questions per task: {num_questions_per_task}")
    
    # Task configuration
    def create_task_with_fallback(short_names, np_random, room, agent, task_kwargs, history_tracker):
        last_error = None
        for name in short_names:
            try:
                task = EvalTaskType.create_task(name, np_random, room, agent, task_kwargs, history_tracker)
                return task, name
            except ValueError as e:
                last_error = e
        raise last_error if last_error else ValueError("No task short names provided")
    
    task_type = 'bwd_loc'
    task_short_names = ['bwd_loc_text', 'bwd_loc']
    task_kwargs = {'image_dir': image_dir}
    extract_args = (offset,)
    all_viewpoints = []
    
    # Create a simple history tracker to mimic VAGEN's deduplication
    class SimpleHistoryTracker:
        def __init__(self):
            self.seen_ids = set()
        
        def has_question(self, task_id: str) -> bool:
            return task_id in self.seen_ids
        
        def add_question(self, task_id: str):
            self.seen_ids.add(task_id)
    
    print(f"\nExtracting viewpoints: {task_type}")
    print(f"Creating task object with seed={run_seed}")

    # Create history tracker for this task type
    history_tracker = SimpleHistoryTracker()

    # CRITICAL: Create task object ONCE per task type (just like VAGEN)
    np_random = np.random.default_rng(run_seed)
    task, used_short_name = create_task_with_fallback(task_short_names, np_random, room, agent, task_kwargs, history_tracker)

    task_viewpoints = []

    # Call generate_question() multiple times on SAME task object
    for i in range(num_questions_per_task):
        print(f"Calling generate_question() #{i+1} (np_random state accumulated)")

        viewpoint = extract_bwd_loc_viewpoint(task, *extract_args)

        if viewpoint:
            pos = viewpoint['pos_tdw']
            yaw = viewpoint['yaw']
            task_id = viewpoint['task_id']
            original_group = viewpoint.get('task_type')
            viewpoint['task_group'] = original_group
            viewpoint['task_type'] = used_short_name
            viewpoint['task_short_name'] = used_short_name
            viewpoint['task_class'] = task.__class__.__name__

            # Add to history tracker
            history_tracker.add_question(task_id)

            viewpoint['question_idx'] = i + 1
            if not viewpoint.get('image_name'):
                viewpoint['image_name'] = task_id
            task_viewpoints.append(viewpoint)

            print(f"{task_type}-{i+1}: task_id={task_id}, pos=({pos[0]:.2f}, {pos[1]:.2f}), yaw={yaw:.1f}°")
        else:
            print(f"Failed to generate viewpoint for {task_type}-{i+1}")

    all_viewpoints.extend(task_viewpoints)
    print(f"Generated {len(task_viewpoints)}/{num_questions_per_task} viewpoints")
    
    print("\n" + "=" * 70)
    print(f"Total viewpoints generated: {len(all_viewpoints)}/{num_questions_per_task}")
    print("=" * 70)
    
    return all_viewpoints


# ============================================================================
# Utility Functions
# ============================================================================

def viewpoints_to_camera_metadata(viewpoints: List[Dict]) -> List[Dict]:
    """Convert viewpoints to camera metadata format for storage.
    
    Args:
        viewpoints: List of viewpoint dicts
        
    Returns:
        List of camera metadata dicts
    """
    cameras = []
    for vp in viewpoints:
        cam_id = vp.get('cam_id', vp['image_name'])
        camera = {
            "id": f"task_{cam_id}",
            "label": f"T{vp['question_idx']}",
            "position": {
                "x": float(vp['pos_tdw'][0]),
                "y": 0.8,  # Standard camera height
                "z": float(vp['pos_tdw'][1])
            },
            "rotation": {"y": float(vp['yaw'])},
            "task_type": vp['task_type'],
            "task_group": vp.get('task_group', vp['task_type']),
            "task_short_name": vp.get('task_short_name'),
            "task_class": vp.get('task_class'),
            "image_name": vp['image_name']
        }
        
        # Add rotation info for false_belief
        if 'rotation_info' in vp:
            camera['rotation_info'] = vp['rotation_info']
        
        cameras.append(camera)
    
    return cameras


if __name__ == "__main__":
    # Test with metadata
    test_metadata_path = "/Users/songshe/objaverse_import/ToS2/multi_room_gen/aaa_dataset_1104_test_orientation/run00/meta_data.json"
    
    if Path(test_metadata_path).exists():
        with open(test_metadata_path, 'r') as f:
            metadata = json.load(f)
        
        viewpoints = generate_task_viewpoints(metadata)
        cameras = viewpoints_to_camera_metadata(viewpoints)
        
        print(f"\nGenerated {len(cameras)} camera configurations")
        print(json.dumps(cameras[0], indent=2))
    else:
        print(f"❌ Test metadata not found: {test_metadata_path}")

