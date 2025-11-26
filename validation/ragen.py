import sys
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

from ..utils.vagen import ensure_vagen_path_from_env

EvaluationManager = None
EvalTaskType = None
Room = None
Agent = None
Object = None
VAGEN_AVAILABLE = False


def _load_vagen_dependencies():
    global EvaluationManager, EvalTaskType, Room, Agent, Object, VAGEN_AVAILABLE
    if EvaluationManager is not None:
        return
    ensure_vagen_path_from_env()
    try:
        from vagen.env.spatial.Base.tos_base.managers.evaluation_manager import EvaluationManager as VagenEvaluationManager
        from vagen.env.spatial.Base.tos_base.evaluation.task_types import EvalTaskType as VagenEvalTaskType
        from vagen.env.spatial.Base.tos_base.core.room import Room as VagenRoom
        from vagen.env.spatial.Base.tos_base.core.object import Agent as VagenAgent, Object as VagenObject
        EvaluationManager = VagenEvaluationManager
        EvalTaskType = VagenEvalTaskType
        Room = VagenRoom
        Agent = VagenAgent
        Object = VagenObject
        VAGEN_AVAILABLE = True
        return
    except ImportError:
        pass
    try:
        from tos_base.managers.evaluation_manager import EvaluationManager as LegacyEvaluationManager
        from tos_base.evaluation.task_types import EvalTaskType as LegacyEvalTaskType
        from tos_base.core.room import Room as LegacyRoom
        from tos_base.core.object import Agent as LegacyAgent, Object as LegacyObject
        EvaluationManager = LegacyEvaluationManager
        EvalTaskType = LegacyEvalTaskType
        Room = LegacyRoom
        Agent = LegacyAgent
        Object = LegacyObject
        VAGEN_AVAILABLE = True
        return
    except ImportError:
        VAGEN_AVAILABLE = False
        print("Warning: VAGEN components not available.")

def convert_mask_to_ragen_format(mask: np.ndarray) -> np.ndarray:
    ragen_mask = np.zeros_like(mask)
    ragen_mask[mask >= 1] = 1
    ragen_mask[mask >= 100] = 1
    ragen_mask[mask == 0] = 0
    ragen_mask[mask == -1] = -1
    return ragen_mask

class RagenValidator:
    def __init__(self, eval_tasks: List[Dict[str, Any]]):
        _load_vagen_dependencies()
        self.eval_tasks = eval_tasks
        self.enabled = VAGEN_AVAILABLE and len(eval_tasks) > 0

    def validate_scene(self, mask: np.ndarray, objects: List[Any], agent: Any, np_random: np.random.Generator) -> Tuple[bool, Dict]:
        if not self.enabled:
            return True, {"accuracy": 1.0, "message": "Validation disabled"}

        try:
            room = Room(objects=objects, mask=mask, name="validation_room")
            all_valid = True
            task_results = []

            for i, task_spec in enumerate(self.eval_tasks):
                try:
                    eval_manager = EvaluationManager([task_spec], np_random, room, agent, history_manager=None, seed=hash(str(task_spec)) % (2**31))
                    question = eval_manager.get_current_question()
                    if question is None:
                        all_valid = False
                        task_results.append({"task_index": i, "success": False, "error": "No question generated"})
                    else:
                        task_results.append({"task_index": i, "success": True})
                except Exception as e:
                    all_valid = False
                    task_results.append({"task_index": i, "success": False, "error": str(e)})

            return all_valid, {"all_tasks_valid": all_valid, "task_results": task_results}
        except Exception as e:
            return False, {"error": str(e), "all_tasks_valid": False}

class PreRenderValidator:
    def __init__(self, config: Dict[str, Any]):
        _load_vagen_dependencies()
        self.config = config
        self.validation_config = config.get('text_based_validity', {})
        self.enabled = self.validation_config.get('enabled', False) and self.validation_config.get('pre_render_validation', False)
        self.eval_tasks = self.validation_config.get('eval_tasks', [])
        if self.enabled and not VAGEN_AVAILABLE:
            print("Warning: Pre-render validation enabled but VAGEN not available.")
            self.enabled = False

    def validate(self, mask: np.ndarray, objects_data: List[Dict], agent_data: Dict, np_random: np.random.Generator) -> Tuple[bool, Dict]:
        if not self.enabled:
            return True, {"status": "disabled"}
        
        try:
            ragen_mask = convert_mask_to_ragen_format(mask)
            ragen_objects = self._convert_objects(objects_data)
            ragen_agent = self._convert_agent(agent_data)
            
            validator = RagenValidator(self.eval_tasks)
            is_valid, summary = validator.validate_scene(ragen_mask, ragen_objects, ragen_agent, np_random)
            summary['status'] = 'completed'
            return is_valid, summary
        except Exception as e:
            return False, {"status": "error", "error": str(e)}

    def _convert_objects(self, objects_data: List[Dict]) -> List[Any]:
        if not VAGEN_AVAILABLE:
            raise ImportError("VAGEN components not available; cannot convert objects")
        ragen_objects = []
        for obj_data in objects_data:
            if obj_data.get('at_default_position', False) or not obj_data.get('is_placed', True):
                continue
            
            pos_data = obj_data.get('position', obj_data.get('pos', (0,0)))
            if isinstance(pos_data, dict):
                x, z = pos_data.get('x', 0), pos_data.get('z', 0)
            else:
                x, z = pos_data[0], pos_data[1]
                            
            obj = Object(name=obj_data.get('name', 'obj'), pos=np.array([x, z], dtype=float))
            ragen_objects.append(obj)
        return ragen_objects

    def _convert_agent(self, agent_data: Dict) -> Any:
        if not VAGEN_AVAILABLE:
            raise ImportError("VAGEN components not available; cannot convert agent")
        pos_data = agent_data.get('position', agent_data.get('pos', (0,0)))
        if isinstance(pos_data, dict):
            x, z = pos_data.get('x', 0), pos_data.get('z', 0)
        else:
            x, z = pos_data[0], pos_data[1]
        
        agent = Agent(name='agent', pos=np.array([x, z], dtype=float))
        return agent

    def _world_to_mask_coords(self, world_x: float, world_z: float, mask_shape: Tuple[int, int] = (20, 20)) -> Tuple[int, int]:
         rows, cols = mask_shape
         offset_x = -(rows // 2)
         offset_z = -(cols // 2)
         row = round(world_x - offset_x)
         col = round(world_z - offset_z)
         return (row, col)

