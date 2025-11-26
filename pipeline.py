import argparse
import yaml
import json
import sys
import os
import shutil
import random
from pathlib import Path
from typing import Dict, Optional
import numpy as np

from .layout.generator import VagenRoomGenerator
from .layout.legacy_generator import generate_room_layout as legacy_generate_room_layout
from .layout.validator import LayoutValidator
from .scene.generator import SceneGenerator
from .scene import object_generator # Access to ObjectGenerator if needed for validation
from .utils.vagen import configure_vagen_path

class Pipeline:
    def __init__(self, config_path: str, output_dir: Optional[str] = None):
        self.config_path = Path(config_path)
        self.config = self._load_config(self.config_path)
        self._configure_vagen_path()
        
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path(self.config.get('output', {}).get('base_dir', './output'))
            
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _configure_vagen_path(self):
        vagen_cfg = self.config.get('vagen', {})
        configure_vagen_path(vagen_cfg.get('path'))

    def _load_config(self, path: Path) -> Dict:
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def run(self, layout_only: bool = False):
        print(f"🏗️  Starting Pipeline with config: {self.config_path}")
        print(f"📁 Output directory: {self.output_dir}")
        
        batch_config = self.config.get('batch', {})
        num_runs = batch_config.get('num_runs', 1)
        seed_start = batch_config.get('seed_start', 42)
        seed_increment = batch_config.get('seed_increment', 1)
        run_offset = batch_config.get('run_offset', 0)
        
        for i in range(num_runs):
            run_seed = seed_start + i * seed_increment
            run_num = i + run_offset
            run_name = f"run{run_num:02d}"
            run_output_dir = self.output_dir / run_name
            
            print(f"\n===== Generating {run_name} (Seed: {run_seed}) =====")
            
            try:
                # 1. Generate Layout
                mask_path = self._generate_layout(run_seed, run_output_dir, run_name)
                
                if layout_only:
                    print(f"✅ Layout generated: {mask_path}")
                    continue
                
                # 2. Generate Scene
                self._generate_scene(run_seed, run_output_dir, mask_path)
                
                print(f"✅ {run_name} completed successfully.")
                
            except Exception as e:
                print(f"❌ {run_name} failed: {e}")
                import traceback
                traceback.print_exc()

    def _generate_layout(self, seed: int, output_dir: Path, run_name: str) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        layout_config = self.config['room_layout']
        use_vagen = layout_config.get('use_vagen', False)
        
        if use_vagen:
            room_size = tuple(layout_config.get('room_size_tuple', [6, 6]))
            room_num = layout_config.get('room_num', 3)
            n_objects = layout_config.get('n_objects', 4)
            topology = layout_config.get('topology', 1)
            
            mask = VagenRoomGenerator.generate(
                room_size=room_size,
                room_num=room_num,
                n_objects=n_objects,
                topology=topology,
                seed=seed,
                save_plot=str(output_dir / f"{run_name}_layout.png")
            )
        else:
            # Legacy support
            n = layout_config['room_size'][0]
            level = layout_config.get('level', 2)
            main_room_size = layout_config.get('main_room_size')
            np_random = np.random.default_rng(seed)
            
            mask = legacy_generate_room_layout(
                n=n,
                level=level,
                main=main_room_size,
                np_random=np_random
            )

        # Save mask
        mask_path = output_dir / f"{run_name}_mask.json"
        with open(mask_path, 'w') as f:
            json.dump(mask.tolist(), f, indent=2)
            
        return mask_path

    def _generate_scene(self, seed: int, output_dir: Path, mask_path: Path):
        random.seed(seed)
        np.random.seed(seed)
        generator = SceneGenerator(self.config, mask_path, output_dir, seed)
        generator.generate()

def main():
    parser = argparse.ArgumentParser(description="ToS Data Generation Pipeline")
    parser.add_argument("--config", type=str, default="tos_data_gen/config.yaml")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--layout-only", action="store_true")
    
    args = parser.parse_args()
    
    pipeline = Pipeline(args.config, args.output)
    pipeline.run(layout_only=args.layout_only)

if __name__ == "__main__":
    main()

