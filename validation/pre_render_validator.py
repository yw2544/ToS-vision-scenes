#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pre-Render Task Validator
==========================

Validates scene with placed objects BEFORE TDW rendering using Vagen EvaluationManager.
This validator runs after object placement optimization but before expensive TDW rendering,
saving time by rejecting invalid scenes early.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import sys
import os

# Add paths for VAGEN imports from config/env
vagen_root = os.environ.get("VAGEN_ROOT")
vagen_base = os.environ.get("VAGEN_BASE")
for p in (vagen_root, vagen_base):
    if p and p not in sys.path:
        sys.path.append(p)

# Import VAGEN components
try:
    from tos_base.managers.evaluation_manager import EvaluationManager
    from tos_base.evaluation.task_types import EvalTaskType
    from tos_base.core.room import Room
    from tos_base.core.object import Agent, Object
    VAGEN_AVAILABLE = True
except ImportError as e:
    print(f"Warning: VAGEN components not available: {e}")
    VAGEN_AVAILABLE = False

class PreRenderValidator:
    """
    Validates scene before TDW rendering using RAGEN EvaluationManager.
    
    This validator is designed to run AFTER object placement but BEFORE TDW rendering,
    to catch invalid scenes early and avoid wasting time on rendering unusable data.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.validation_config = config.get('text_based_validity', {})
        self.enabled = self.validation_config.get('enabled', False) and self.validation_config.get('pre_render_validation', False)
        self.eval_tasks = self.validation_config.get('eval_tasks', [])
        self.max_retries = self.validation_config.get('max_retries', 10)
        self.seed_increment = self.validation_config.get('seed_increment', 137)
        
        if self.enabled and not VAGEN_AVAILABLE:
            print("Warning: Pre-render validation enabled but VAGEN not available, disabling validation")
            self.enabled = False
            
        if self.enabled:
            print(f"Pre-render validator initialized with {len(self.eval_tasks)} tasks")
    
    def is_enabled(self) -> bool:
        """Check if validation is enabled."""
        return self.enabled and VAGEN_AVAILABLE
    
    def is_available(self) -> bool:
        """Check if VAGEN components are available for validation."""
        return VAGEN_AVAILABLE
    
    def validate_scene_before_render(self, mask: np.ndarray, 
                                   objects_data: List[Dict], 
                                   agent_data: Dict, 
                                   np_random: np.random.Generator,
                                   attempt_num: int = 0) -> Tuple[bool, Dict]:
        """
        Validate scene before TDW rendering.
        
        Args:
            mask: Room layout mask (multiroom format)
            objects_data: List of object dictionaries with name, position, etc.
            agent_data: Agent dictionary with position, orientation, etc.
            np_random: Random number generator
            attempt_num: Current attempt number (for logging)
            
        Returns:
            (is_valid, validation_summary)
        """
        if not self.enabled:
            return True, {"status": "disabled", "tasks_tested": 0}
            
        if not VAGEN_AVAILABLE:
            print("Warning: VAGEN not available, skipping pre-render validation")
            return True, {"status": "vagen_unavailable", "tasks_tested": 0}
        
        print(f"\nPre-render validation (attempt {attempt_num + 1})...")
        print(f"Testing {len(self.eval_tasks)} task types:")
        for task in self.eval_tasks:
            print(f"  - {task.get('task_type', 'unknown')}")
        
        # Debug: Print input data
        print(f"DEBUG - Input data:")
        print(f"   Mask shape: {mask.shape}")
        print(f"   Objects count: {len(objects_data)}")
        if objects_data:
            print(f"   First object data: {objects_data[0]}")
        print(f"   Agent data: {agent_data}")
            
        try:
            # Store mask shape for coordinate conversion
            self._current_mask_shape = mask.shape
            
            # Convert to RAGEN format
            ragen_mask = self._convert_mask_to_ragen_format(mask)
            ragen_objects = self._convert_objects_to_ragen_format(objects_data)
            ragen_agent = self._convert_agent_to_ragen_format(agent_data)
            
            print(f"Converted {len(ragen_objects)} objects and 1 agent to VAGEN format")
            
            # Create VAGEN Room object
            room = Room(objects=ragen_objects, name="pre_render_validation", mask=ragen_mask)
            
            # Test all tasks individually (new EvaluationManager only supports single task)
            all_valid = True
            task_results = []
            failed_tasks = []
            
            print(f"Testing {len(self.eval_tasks)} task types individually...")
            valid_task_types = set(EvalTaskType.get_short_names()) if hasattr(EvalTaskType, "get_short_names") else None
            
            for i, task_spec in enumerate(self.eval_tasks):
                try:
                    task_type = task_spec.get('task_type', 'unknown')
                    if valid_task_types is not None and task_type not in valid_task_types:
                        all_valid = False
                        failed_tasks.append(task_type)
                        task_results.append({
                            "task_index": i,
                            "task_name": task_type,
                            "success": False,
                            "error": f"Unknown task type: {task_type}"
                        })
                        print(f"  ❌ {task_type}: Unknown task type")
                        continue
                    
                    # Create single-task eval manager for each task
                    eval_manager = EvaluationManager([task_spec], np_random, room, ragen_agent, history_manager=None, seed=hash(str(task_spec)) % (2**31))
                    
                    # Try to get question for current task
                    question = self._get_question_from_manager(eval_manager)
                    
                    if question is None:
                        all_valid = False
                        failed_tasks.append(task_type)
                        task_results.append({
                            "task_index": i, 
                            "task_name": task_type,
                            "success": False, 
                            "error": "No question generated"
                        })
                        print(f"  ❌ {task_type}: Failed to generate question")
                    else:
                        task_results.append({
                            "task_index": i, 
                            "task_name": task_type,
                            "success": True, 
                            "question_preview": question[:100] + "..." if len(question) > 100 else question
                        })
                        print(f" {task_type}: Question generated successfully")
                        
                except Exception as e:
                    all_valid = False
                    task_type = task_spec.get('task_type', f"Task_{i}")
                    failed_tasks.append(task_type)
                    task_results.append({
                        "task_index": i, 
                        "task_name": task_type,
                        "success": False, 
                        "error": str(e)
                    })
                    print(f"  ❌ {task_type}: {str(e)}")
            
            # Create summary (compatible with old format)
            eval_summary = {
                "accuracy": 1.0 if all_valid else 0.0,
                "total_tasks": len(self.eval_tasks),
                "correct_count": len(self.eval_tasks) if all_valid else 0,
                "incorrect_count": 0 if all_valid else len(self.eval_tasks),
                "unanswered_count": 0
            }
            
            # Extract validated task types for successful validation
            validated_tasks = []
            if all_valid:
                for task_config in self.config.get('text_based_validity', {}).get('eval_tasks', []):
                    task_type = task_config.get('task_type')
                    if task_type:
                        validated_tasks.append(task_type)
            
            eval_summary.update({
                "status": "completed",
                "all_tasks_valid": all_valid,
                "tasks_tested": len(task_results),
                "failed_tasks": failed_tasks,
                "task_results": task_results,
                "validated_tasks": validated_tasks,  # Add validated task types
                "attempt_num": attempt_num
            })
            
            if all_valid:
                print(f"Pre-render validation PASSED! All {len(task_results)} tasks are valid")
                print(f"Validated tasks: {', '.join(validated_tasks)}")
            else:
                print(f"❌ Pre-render validation FAILED! {len(failed_tasks)} task(s) failed: {', '.join(failed_tasks)}")
            
            return all_valid, eval_summary
            
        except Exception as e:
            error_summary = {
                "status": "error",
                "error": str(e),
                "all_tasks_valid": False,
                "tasks_tested": 0,
                "attempt_num": attempt_num
            }
            print(f"❌ Pre-render validation ERROR: {e}")
            return False, error_summary
    
    def _convert_mask_to_ragen_format(self, mask: np.ndarray) -> np.ndarray:
        """Convert multiroom generator mask to VAGEN format."""
        ragen_mask = mask.copy()
        
        # Doors (100+) become walkable but maintain as room connection
        # For VAGEN, doors can be treated as walkable cells (1)
        ragen_mask[mask >= 100] = 1
        
        # Everything else stays the same:
        # - Rooms (1, 2, 3...) stay as room IDs
        # - Walls (0) stay as walls (0) 
        # - Impassable (-1) stay as impassable (-1)
        
        return ragen_mask

    def _get_question_from_manager(self, eval_manager) -> Optional[str]:
        """Get a question from EvaluationManager across VAGEN versions."""
        if hasattr(eval_manager, "get_current_question"):
            return eval_manager.get_current_question()
        task = None
        if hasattr(eval_manager, "_get_current_eval_task"):
            try:
                task = eval_manager._get_current_eval_task()
            except Exception:
                task = None
        if task is None and hasattr(eval_manager, "tasks") and eval_manager.tasks:
            task = eval_manager.tasks[0]
        if task is None:
            return None
        question = getattr(task, "question", None)
        if question:
            return question
        if hasattr(task, "generate_question"):
            return task.generate_question()
        return None
    
    def _world_to_mask_coords(self, world_x: float, world_z: float) -> Tuple[int, int]:
        """Convert TDW world coordinates back to mask coordinates.
        
        This reverses the _cell_to_world transformation:
        world(x,z) → mask(row,col) where x=row, z=col
        """
        # Get mask dimensions from stored values during validation
        if hasattr(self, '_current_mask_shape'):
            rows, cols = self._current_mask_shape
        else:
            # Fallback - assume 20x20 (most common)
            rows, cols = 20, 20
        
        # Cell size is usually 1.0
        cell_size = 1.0
        
        # Calculate offset (same as in room_analyzer._cell_to_world)
        offset_x = -(rows // 2)  # offset for row (X axis) 
        offset_z = -(cols // 2)  # offset for col (Z axis)
        
        # Reverse conversion: world → mask
        # world(x,z) - offset = mask(row,col) where x=row, z=col
        row = round(world_x / cell_size - offset_x)
        col = round(world_z / cell_size - offset_z)
        return (row, col)
    
    def _convert_objects_to_ragen_format(self, objects_data: List[Dict]) -> List[Object]:
        """Convert object data to VAGEN Object format."""
        ragen_objects = []
        
        for obj_data in objects_data:
            # Skip objects at default positions (not placed yet)
            if self._is_default_position(obj_data):
                continue
                
            # Extract TDW world position from object data  
            if 'position' in obj_data:
                if isinstance(obj_data['position'], dict):
                    world_x = obj_data['position']['x']
                    world_z = obj_data['position']['z'] 
                else:
                    # position is already a tuple/list (x, z)
                    world_x, world_z = obj_data['position'][:2]
            elif 'pos' in obj_data:
                world_x, world_z = obj_data['pos'][:2]
            elif 'final_position' in obj_data:
                final_pos = obj_data['final_position']
                world_x = final_pos['x']
                world_z = final_pos['z']
            else:
                continue  # Skip objects without position
            
            # Create VAGEN Object
            name = obj_data.get('name', f"object_{len(ragen_objects)}")
            
            # Convert TDW world coordinates back to mask coordinates for VAGEN
            # VAGEN expects positions in mask coordinate space, not TDW world space
            mask_row, mask_col = self._world_to_mask_coords(world_x, world_z)
            pos = np.array([mask_row, mask_col], dtype=float)                
            obj = Object(name=name, pos=pos)
            
            # Don't set room_id - let Room class determine it from mask and position
            ragen_objects.append(obj)
            
        return ragen_objects
    
    def _convert_agent_to_ragen_format(self, agent_data: Dict) -> Agent:
        """Convert agent data to VAGEN Agent format."""
        # Extract TDW world position from agent data
        if 'position' in agent_data:
            if isinstance(agent_data['position'], dict):
                world_x = agent_data['position']['x']
                world_z = agent_data['position']['z']
            else:
                # position is already a tuple/list (x, z)
                world_x, world_z = agent_data['position'][:2]
        elif 'pos' in agent_data:
            world_x, world_z = agent_data['pos'][:2]
        elif 'final_position' in agent_data:
            final_pos = agent_data['final_position']
            world_x = final_pos['x']
            world_z = final_pos['z']
        else:
            raise ValueError("Agent data must contain position information")
        
        # Convert TDW world coordinates back to mask coordinates for VAGEN
        mask_row, mask_col = self._world_to_mask_coords(world_x, world_z)
        pos = np.array([mask_row, mask_col], dtype=float)
        
        print(f"DEBUG - Agent: world({world_x}, {world_z}) → mask({mask_row}, {mask_col})")
            
        # Extract orientation if available
        ori = None
        if 'rotation' in agent_data:
            # Convert TDW rotation to orientation vector
            # This is a simplified conversion - may need adjustment based on TDW's coordinate system
            ori = np.array([0, 1], dtype=float)  # Default north-facing
        elif 'ori' in agent_data:
            ori = np.array(agent_data['ori'], dtype=float)
            
        # Create VAGEN Agent with initial state
        agent = Agent(name='agent', pos=pos, init_pos=pos.copy())
        if ori is not None:
            agent.ori = ori
            agent.init_ori = ori.copy()
        
        # Set room_id and init_room_id if available
        room_id = agent_data.get('room_id')
        if room_id is not None:
            agent.room_id = int(room_id)
            agent.init_room_id = int(room_id)
        
        return agent
    
    def _is_default_position(self, obj_data: Dict) -> bool:
        """Check if object is at a default/unplaced position."""
        # Check for common default position indicators
        if 'position' in obj_data:
            pos = obj_data['position']
            # Handle different position formats
            if isinstance(pos, dict):
                # Dict format: {'x': ..., 'z': ...}
                x, z = pos.get('x', 0), pos.get('z', 0)
            elif isinstance(pos, (list, tuple)) and len(pos) >= 2:
                # List/tuple format: [x, z] or (x, z)
                x, z = pos[0], pos[1]
            else:
                return True  # Unknown format, treat as default
                
            # Common default positions: origin, very large values, NaN, etc.
            if (x == 0 and z == 0) or abs(x) > 1000 or abs(z) > 1000:
                return True
        
        # Check if marked as unplaced
        if obj_data.get('is_placed', True) is False:
            return True
        
        # Check for specific default position markers
        if obj_data.get('at_default_position', False):
            return True
            
        return False

def extract_scene_data_for_validation(object_generator, room_analyzer):
    """
    Extract scene data from object generator for VAGEN validation.
    
    Args:
        object_generator: ObjectGenerator instance with placed objects
        room_analyzer: RoomAnalyzer instance with mask and room info
        
    Returns:
        (mask, objects_data, agent_data)
    """
    # Get mask from room analyzer and ensure it's a numpy array
    mask = room_analyzer.mask
    if not isinstance(mask, np.ndarray):
        mask = np.array(mask)
    
    # Extract objects data
    objects_data = []
    all_objects = object_generator.get_all_objects()
    
    for obj in all_objects:
        final_pos = obj.get_final_position()
        # Handle different position formats from ObjectGenerator
        if isinstance(final_pos, dict):
            # Dictionary format: {"x": ..., "y": ..., "z": ...}
            position = (float(final_pos.get('x', 0)), float(final_pos.get('z', 0)))
        elif isinstance(final_pos, (list, tuple)) and len(final_pos) >= 2:
            # List/tuple format: [x, z] or (x, z)
            position = (float(final_pos[0]), float(final_pos[1]))
        else:
            print(f"[WARNING] Invalid position format for {obj.name}: {final_pos}")
            continue
            
        obj_data = {
            'name': obj.name,
            'position': position,
            'model': obj.model,
            'room_id': obj.room_id,
            'is_placed': True
        }
        objects_data.append(obj_data)
    
    # Extract agent data  
    agent_data = {}
    agent = object_generator.get_agent()
    if agent:
        # Agent is AgentInfo dataclass with pos (dict), room_id, rotation (dict)
        if isinstance(agent.pos, dict):
            position = (float(agent.pos.get('x', 0)), float(agent.pos.get('z', 0)))
        else:
            position = (0.0, 0.0)
            
        agent_data = {
            'name': 'agent',
            'position': position,
            'rotation': agent.rotation.get('y', 0) if isinstance(agent.rotation, dict) else 0,
            'room_id': agent.room_id
        }
    elif len(objects_data) > 0:
        # Use first object's room as agent room
        first_obj = objects_data[0]
        agent_data = {
            'name': 'agent',
            'position': (0.0, 0.0),  # Default position
            'rotation': 0,
            'room_id': first_obj['room_id']
        }
    else:
        # Fallback
        agent_data = {
            'name': 'agent',
            'position': (0.0, 0.0),
            'rotation': 0,
            'room_id': 1
        }
    
    return mask, objects_data, agent_data

if __name__ == "__main__":
    # Test the pre-render validator
    print("Testing Pre-Render Validator...")
    
    config = {
        'text_based_validity': {
            'enabled': True,
            'pre_render_validation': True,
            'max_retries': 3,
            'eval_tasks': [
                {
                    'task_type': 'rot',
                    'task_kwargs': {
                        'angle_eps': 15.0,
                        'num_choices': 4
                    }
                },
                {
                    'task_type': 'dir',
                    'task_kwargs': {
                        'num_choices': 4
                    }
                }
            ]
        }
    }
    
    validator = PreRenderValidator(config)
    print(f"Validator enabled: {validator.is_enabled()}")
    print(f"Configured tasks: {len(validator.eval_tasks)}")
    
    if validator.is_enabled():
        print("Pre-render validator ready for integration!")
    else:
        print("❌ Pre-render validator not available")
