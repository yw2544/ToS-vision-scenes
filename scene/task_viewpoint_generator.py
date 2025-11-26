#!/usr/bin/env python3
"""
Task Viewpoint Generator Module
================================
Extracts additional viewpoints from VAGEN evaluation tasks and generates 
camera configurations for TDW rendering.

Tasks supported:
- bwd_nav: BackwardNavEvaluationTask
- bwd_loc: BackwardLocEvaluationTask
- bwd_pov: BackwardPovEvaluationTask
- false_belief: FalseBeliefDirectionPov

Author: Auto-generated
Date: 2024-11-04
"""

import os
import sys
import json
import numpy as np
import math
from typing import List, Dict, Tuple, Optional
from pathlib import Path

from ..utils.vagen import ensure_vagen_path_from_env


def _import_vagen_modules():
    ensure_vagen_path_from_env()
    try:
        from vagen.env.spatial.Base.tos_base.utils.room_utils import initialize_room_from_json
        from vagen.env.spatial.Base.tos_base.evaluation.task_types import EvalTaskType
        return initialize_room_from_json, EvalTaskType
    except ImportError:
        vagen_path = os.environ.get("VAGEN_PATH")
        if vagen_path and vagen_path not in sys.path:
            sys.path.insert(0, vagen_path)
        try:
            from vagen.env.spatial.Base.tos_base.utils.room_utils import initialize_room_from_json  # type: ignore
            from vagen.env.spatial.Base.tos_base.evaluation.task_types import EvalTaskType  # type: ignore
            return initialize_room_from_json, EvalTaskType
        except ImportError as exc:
            raise ImportError("VAGEN package not found. Please set VAGEN_PATH or install the dependency.") from exc


_INITIALIZE_ROOM = None
_EVAL_TASK_TYPE = None


def _get_vagen_modules():
    global _INITIALIZE_ROOM, _EVAL_TASK_TYPE
    if _INITIALIZE_ROOM is None or _EVAL_TASK_TYPE is None:
        _INITIALIZE_ROOM, _EVAL_TASK_TYPE = _import_vagen_modules()
    return _INITIALIZE_ROOM, _EVAL_TASK_TYPE


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
    """Convert mask coordinates to TDW world coordinates.
    
    Args:
        pos_mask: Position in mask coordinates [x, z]
        offset: Offset tuple (offset_x, offset_z) from metadata
        
    Returns:
        Position in TDW coordinates [x, z]
    """
    return pos_mask - np.array(offset)


# ============================================================================
# Task-Specific Viewpoint Extraction Functions
# ============================================================================

def extract_bwd_nav_viewpoint(task, offset: tuple) -> Optional[Dict]:
    """Extract viewpoint from BackwardNavEvaluationTask.
    
    The viewpoint is at the final position/orientation after navigation.
    
    Args:
        task: Already instantiated VAGEN task object (will call generate_question)
        offset: Coordinate offset (offset_x, offset_z)
        
    Returns:
        Viewpoint dict or None if extraction failed
    """
    try:
        task.generate_question()
        
        answer = task.eval_data.answer
        if isinstance(answer, dict) and 'final_pos' in answer and 'final_ori' in answer:
            pos_mask = np.array(answer['final_pos'], dtype=float)
            ori_mask = np.array(answer['final_ori'])
            pos_tdw = mask_pos_to_tdw(pos_mask, offset)
            yaw = ori_to_yaw(ori_mask)
            
            return {
                'task_type': 'bwd_nav',
                'pos_tdw': pos_tdw,
                'yaw': yaw,
                'task_id': task.eval_data.id,
                'extraction_method': 'from_answer'
            }
    except Exception as e:
        print(f"[ERROR] Failed to extract bwd_nav viewpoint: {e}")
    return None


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
        
        return {
            'task_type': 'bwd_loc',
            'pos_tdw': pos_tdw,
            'yaw': yaw,
            'task_id': task.eval_data.id,
            'extraction_method': 'from_agent',
            'agent_pos_mask': pos_mask,
            'agent_ori': ori_mask
        }
    except Exception as e:
        print(f"[ERROR] Failed to extract bwd_loc viewpoint: {e}")
    return None


def extract_bwd_pov_viewpoint(task, room, offset: tuple) -> Optional[Dict]:
    """Extract viewpoint from BackwardPovEvaluationTask.
    
    The viewpoint is at the anchor object's position and orientation.
    
    Args:
        task: Already instantiated VAGEN task object (will call generate_question)
        room: VAGEN Room object (to get object reference)
        offset: Coordinate offset
        
    Returns:
        Viewpoint dict or None
    """
    try:
        task.generate_question()
        
        # Answer is the anchor object name
        anchor_name = task.eval_data.answer
        anchor = room.get_object_by_name(anchor_name)
        
        pos_mask = anchor.pos.copy()
        ori_mask = anchor.ori.copy()
        pos_tdw = mask_pos_to_tdw(pos_mask, offset)
        yaw = ori_to_yaw(ori_mask)
        
        return {
            'task_type': 'bwd_pov',
            'pos_tdw': pos_tdw,
            'yaw': yaw,
            'task_id': task.eval_data.id,
            'extraction_method': 'from_anchor',
            'anchor_name': anchor_name,
            'anchor_pos_mask': pos_mask,
            'anchor_ori': ori_mask
        }
    except Exception as e:
        print(f"[ERROR] Failed to extract bwd_pov viewpoint: {e}")
    return None


def extract_false_belief_viewpoint(task, room, agent, offset: tuple) -> Optional[Dict]:
    """Extract viewpoint from FalseBeliefDirectionPov task.
    
    In this task:
    - An anchor object is rotated (90/180/270 degrees)
    - The agent observes from their original position, facing north
    - The viewpoint is at agent.pos with yaw=0 (facing north)
    
    We extract:
    - Viewpoint: agent's position facing north
    - Rotation info: which object to rotate and by how many degrees
    
    NOTE: The rotation will be applied in TDW before capturing the image.
    
    Args:
        task: Already instantiated VAGEN task object (will call generate_question)
        room: VAGEN Room object (original, before rotation)
        agent: VAGEN Agent object
        offset: Coordinate offset
        
    Returns:
        Viewpoint dict with rotation_info or None
    """
    try:
        task.generate_question()
        
        # Extract rotation information from task
        kwargs = task.eval_data.kwargs
        rotated_obj_name = kwargs.get('rotated_object')
        rotation_deg = kwargs.get('rotation_degrees')
        
        # Get object from ORIGINAL room (not rotated)
        anchor_original = room.get_object_by_name(rotated_obj_name)
        
        # Viewpoint is at agent's position facing north
        pos_mask = agent.pos.copy()
        pos_tdw = mask_pos_to_tdw(pos_mask, offset)
        
        # Agent faces north (0, 1) for observation
        ori_mask_north = np.array([0, 1])
        yaw_north = ori_to_yaw(ori_mask_north)
        
        # Original orientation of anchor (for metadata)
        ori_mask_original = anchor_original.ori.copy()
        yaw_original = ori_to_yaw(ori_mask_original)
        
        return {
            'task_type': 'false_belief',
            'pos_tdw': pos_tdw,
            'yaw': yaw_north,
            'task_id': task.eval_data.id,
            'extraction_method': 'from_agent_facing_north',
            'anchor_name': rotated_obj_name,
            'agent_pos_mask': pos_mask,
            'rotation_info': {
                'object_to_rotate': rotated_obj_name,
                'rotation_degrees': rotation_deg,
                'original_yaw': yaw_original
            },
            'note': 'Agent at original position facing north; object rotated in TDW'
        }
    except Exception as e:
        print(f"[ERROR] Failed to extract false_belief viewpoint: {e}")
    return None


# ============================================================================
# Main Viewpoint Generation Function
# ============================================================================

def generate_task_viewpoints(metadata: Dict, run_seed: int = None, 
                            num_questions_per_task: int = 3) -> List[Dict]:
    """Generate all task viewpoints from metadata.
    
    This function EXACTLY replicates VAGEN's question generation process:
    1. Create ONE task object per task type (with run_seed)
    2. Call generate_question() MULTIPLE times on the SAME task object
    3. np_random state accumulates between calls
    
    This ensures viewpoints match VAGEN's generated questions perfectly.
    
    Args:
        metadata: Scene metadata dict (must contain objects, offset, etc.)
        run_seed: Seed for reproducibility (defaults to metadata's run_seed or seed)
        num_questions_per_task: Number of questions/viewpoints per task (default: 3)
        
    Returns:
        List of viewpoint dicts, each containing:
            - task_type: str
            - pos_tdw: np.ndarray [x, z]
            - yaw: float (degrees)
            - image_name: str (e.g., "bwd_nav-1")
            - question_idx: int
            - rotation_info: dict (only for false_belief)
    """
    print("\n" + "=" * 70)
    print("🎯 Generating Task Viewpoints (VAGEN-identical strategy)")
    print("=" * 70)
    
    try:
        initialize_room_from_json, EvalTaskType = _get_vagen_modules()
    except ImportError as e:
        print(f"[ERROR] VAGEN modules unavailable: {e}")
        return []

    # Create Room and Agent from metadata
    try:
        room, agent = initialize_room_from_json(metadata)
    except Exception as e:
        print(f"[ERROR] Failed to initialize room from metadata: {e}")
        return []
    
    offset = tuple(metadata.get('offset', [0, 0]))

    # Match VAGEN behavior: use the "<image>" placeholder whenever image_dir is set
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
    
    # Task configurations
    def create_task_with_fallback(short_names, np_random, room, agent, task_kwargs, history_tracker):
        last_error = None
        for name in short_names:
            try:
                task = EvalTaskType.create_task(name, np_random, room, agent, task_kwargs, history_tracker)
                return task, name
            except ValueError as e:
                last_error = e
        raise last_error if last_error else ValueError("No task short names provided")
    
    task_configs = [
        {
            'task_type': 'bwd_nav',
            'task_short_names': ['bwd_nav_vision', 'bwd_nav_text', 'bwd_nav'],
            'task_kwargs': {'steps': 3, 'num_choices': 4, 'image_dir': image_dir},
            'extract_args': (offset,)
        },
        {
            'task_type': 'bwd_loc',
            'task_short_names': ['bwd_loc_vision', 'bwd_loc_text', 'bwd_loc'],
            'task_kwargs': {'image_dir': image_dir},
            'extract_args': (offset,)
        },
        {
            'task_type': 'bwd_pov',
            'task_short_names': ['bwd_pov_vision', 'bwd_pov_text', 'bwd_pov'],
            'task_kwargs': {'image_dir': image_dir},
            'extract_args': (room, offset)
        }
    ]
    
    all_viewpoints = []
    
    # Create a simple history tracker to mimic VAGEN's deduplication
    class SimpleHistoryTracker:
        def __init__(self):
            self.seen_ids = set()
        
        def has_question(self, task_id: str) -> bool:
            return task_id in self.seen_ids
        
        def add_question(self, task_id: str):
            self.seen_ids.add(task_id)
    
    for config in task_configs:
        task_type = config['task_type']
        task_short_names = config['task_short_names']
        task_kwargs = config['task_kwargs']
        extract_args = config['extract_args']
        
        print(f"\n📍 Extracting viewpoints: {task_type}")
        print(f"  🔧 Creating task object with seed={run_seed}")
        
        # Create history tracker for this task type
        history_tracker = SimpleHistoryTracker()
        
        # CRITICAL: Create task object ONCE per task type (just like VAGEN)
        np_random = np.random.default_rng(run_seed)
        task, used_short_name = create_task_with_fallback(task_short_names, np_random, room, agent, task_kwargs, history_tracker)
        
        task_viewpoints = []
        
        # Call generate_question() multiple times on SAME task object
        for i in range(num_questions_per_task):
            print(f"  🔄 Calling generate_question() #{i+1} (np_random state accumulated)")
            
            # Extract viewpoint using appropriate function
            # The @retry_generate_question decorator will automatically retry if duplicate
            if task_type == 'bwd_nav':
                viewpoint = extract_bwd_nav_viewpoint(task, *extract_args)
            elif task_type == 'bwd_loc':
                viewpoint = extract_bwd_loc_viewpoint(task, *extract_args)
            elif task_type == 'bwd_pov':
                viewpoint = extract_bwd_pov_viewpoint(task, *extract_args)
            elif task_type == 'false_belief':
                viewpoint = extract_false_belief_viewpoint(task, *extract_args)
            else:
                print(f"  ⚠️  Unknown task type: {task_type}")
                continue
            
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
                viewpoint['image_name'] = task_id  # Use task_id as image name for VAGEN consistency
                task_viewpoints.append(viewpoint)
                
                print(f"  ✅ {task_type}-{i+1}: task_id={task_id}, pos=({pos[0]:.2f}, {pos[1]:.2f}), yaw={yaw:.1f}°")
            else:
                print(f"  ⚠️  Failed to generate viewpoint for {task_type}-{i+1}")
        
        all_viewpoints.extend(task_viewpoints)
        print(f"  📊 Generated {len(task_viewpoints)}/{num_questions_per_task} viewpoints")
    
    print("\n" + "=" * 70)
    print(f"✅ Total viewpoints generated: {len(all_viewpoints)}/{num_questions_per_task * len(task_configs)}")
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
        camera = {
            "id": f"task_{vp['image_name']}",
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
    test_metadata_path = "test_metadata.json"
    
    if Path(test_metadata_path).exists():
        with open(test_metadata_path, 'r') as f:
            metadata = json.load(f)
        
        viewpoints = generate_task_viewpoints(metadata)
        cameras = viewpoints_to_camera_metadata(viewpoints)
        
        print(f"\n✅ Generated {len(cameras)} camera configurations")
        print(json.dumps(cameras[0], indent=2))
    else:
        print(f"❌ Test metadata not found: {test_metadata_path}")

