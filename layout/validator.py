import numpy as np
from typing import List, Dict, Any, Optional
from ..validation.ragen import RagenValidator, convert_mask_to_ragen_format, VAGEN_AVAILABLE

try:
    from vagen.env.spatial.Base.tos_base.core.object import Agent, Object
except ImportError:
    # Fallback if VAGEN not available, handled by VAGEN_AVAILABLE check
    pass

class LayoutValidator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.validation_config = config.get('text_based_validity', {})
        self.enabled = self.validation_config.get('enabled', False)
        self.max_retries = self.validation_config.get('max_retries', 10)
        self.seed_increment = self.validation_config.get('seed_increment', 137)
        self.eval_tasks = self.validation_config.get('eval_tasks', [])
        
        if self.enabled and not VAGEN_AVAILABLE:
            print("Warning: Text-based validity enabled but VAGEN not available, disabling validation")
            self.enabled = False

    def is_enabled(self) -> bool:
        return self.enabled and VAGEN_AVAILABLE

    def validate_layout_with_ragen(self, mask: np.ndarray, objects: List, agent, np_random: np.random.Generator) -> bool:
        if not self.enabled:
            return True

        validator = RagenValidator(self.eval_tasks)
        # Objects and agent here are expected to be VAGEN compatible objects
        # If they are from the generator, they might need conversion or are already correct.
        # In original code, they are created using create_ragen_objects_from_object_generator
        
        valid, _ = validator.validate_scene(mask, objects, agent, np_random)
        return valid

    def generate_dependent_seed(self, base_seed: int, attempt: int) -> int:
        return (base_seed + attempt * self.seed_increment) % (2**32)

def create_ragen_objects_and_agent(mask: np.ndarray, n_objects: int, np_random: np.random.Generator) -> tuple:
    # Simplified mock object creation
    if not VAGEN_AVAILABLE:
        raise ImportError("VAGEN components not available")
        
    ragen_mask = convert_mask_to_ragen_format(mask)
    valid_positions = []
    for i in range(ragen_mask.shape[0]):
        for j in range(ragen_mask.shape[1]):
            if ragen_mask[i, j] == 1:
                valid_positions.append([i, j])
                
    if len(valid_positions) < n_objects + 1:
        raise ValueError(f"Not enough valid positions for {n_objects} objects + agent")
        
    selected = np_random.choice(len(valid_positions), size=n_objects + 1, replace=False)
    object_names = ["table", "chair", "lamp", "sofa", "desk", "bookshelf", "cabinet", "bed", "dresser"]
    
    objects = []
    for i in range(n_objects):
        pos = valid_positions[selected[i]]
        name = object_names[i % len(object_names)]
        objects.append(Object(name=name, pos=np.array(pos, dtype=float), ori=np.array([0, 1], dtype=float), room_id=1))
        
    agent_pos = valid_positions[selected[-1]]
    agent = Agent(name='agent', pos=np.array(agent_pos, dtype=float), ori=np.array([0, 1], dtype=float), room_id=1,
                  init_pos=np.array(agent_pos, dtype=float), init_ori=np.array([0, 1], dtype=float), init_room_id=1)
                  
    return objects, agent

