"""
Multiroom Scene Generation Pipeline
===================================

Complete multi-room scene generation pipeline including:
1. Room layout generation.
2. Object meta generation and saving.
3. Task validity validation.
3. TDW scene generation.

Usage:
    python generate_pipeline.py --config config.yaml
    python generate_pipeline.py --config config.yaml --output output_dir
"""

import argparse
import json
import yaml
import subprocess
import numpy as np
from pathlib import Path
import tempfile
import os
import sys
from datetime import datetime


def _check_door_hash_collision(room) -> tuple[bool, int, list]:
    """
    Check if any two doors have position hash collision.
    
    This uses the same hash logic as _generate_door_id_from_coordinates in mask2scene_enhanced.py:
        position_hash = abs(hash((round(x, 2), round(z, 2)))) % 100
    
    Args:
        room: VAGEN Room object with gates attribute
        
    Returns:
        (has_collision, num_doors, collisions)
        - has_collision: True if any two doors have the same position hash
        - num_doors: total number of doors
        - collisions: list of collision details [(hash_value, pos1, pos2), ...]
    """
    if not hasattr(room, 'gates') or not room.gates:
        return False, 0, []
    
    # Calculate position hash for each door
    hash_to_pos = {}
    collisions = []
    
    for gate in room.gates:
        x, z = float(gate.pos[0]), float(gate.pos[1])
        position_hash = abs(hash((round(x, 2), round(z, 2)))) % 100
        
        if position_hash in hash_to_pos:
            # Collision detected
            collisions.append((position_hash, (x, z), hash_to_pos[position_hash]))
        else:
            hash_to_pos[position_hash] = (x, z)
    
    return len(collisions) > 0, len(room.gates), collisions


def _get_vagen_paths(config: dict) -> tuple[str | None, str | None]:
    vagen_cfg = config.get("vagen_paths", {})
    return vagen_cfg.get("root"), vagen_cfg.get("base")


def _configure_vagen_paths(config: dict) -> None:
    vagen_root, vagen_base = _get_vagen_paths(config)
    for p in (vagen_root, vagen_base):
        if p and p not in sys.path:
            sys.path.insert(0, p)
    if vagen_root:
        os.environ.setdefault("VAGEN_ROOT", vagen_root)
    if vagen_base:
        os.environ.setdefault("VAGEN_BASE", vagen_base)


def load_config(config_path: str) -> dict:
    """Load YAML configuration file"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"Loaded config: {config_path}")
        _configure_vagen_paths(config)
        return config
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        sys.exit(1)

def validate_config(config: dict) -> bool:
    """Validate configuration file"""
    try:
        # Validate room layout config (VAGEN only)
        layout_config = config.get('room_layout', {})
        room_size_tuple = layout_config.get('room_size_tuple')
        room_num = layout_config.get('room_num')
        n_objects = layout_config.get('n_objects')
        topology = layout_config.get('topology')

        if not room_size_tuple or len(room_size_tuple) != 2:
            raise ValueError("room_size_tuple must be [width, height] for VAGEN")

        if room_num is None or room_num < 1:
            raise ValueError("room_num must be >= 1 for VAGEN")

        if n_objects is None or n_objects < 0:
            raise ValueError("n_objects must be >= 0 for VAGEN")

        if topology not in [0, 1, 2, 3]:
            raise ValueError("topology must be 0, 1, 2, or 3 for VAGEN (3=loop/ring structure)")

        print("Config validation passed (VAGEN mode)")
        return True
        
    except Exception as e:
        print(f"❌ Config validation failed: {e}")
        return False

def generate_room_layout_from_config(config: dict, output_dir: Path, run_name: str = None) -> Path:
    """Generate room layout from configuration with optional text-based validity checking"""
    print("\n" + "="*60)
    print("Now generating room layout...")
    print("="*60)
    
    layout_config = config['room_layout']
    obj_config = config.get('object_generation', {})
    
    print("Using VAGEN RoomGenerator for mask generation")
        
    try:
        from scene.room_layout_generator import generate_room_layout_vagen
    except ImportError as e:
        print(f"Failed to import VAGEN: {e}")
        print("Make sure vagen conda environment is activated")
        sys.exit(1)
        
    # Extract parameters
    room_size_tuple = tuple(layout_config.get('room_size_tuple', [6, 6]))
    room_num = layout_config.get('room_num', 3)
    n_objects = layout_config.get('n_objects', 1)
    topology = layout_config.get('topology', 1)
    seed = layout_config.get('seed', 42)
        
    print(f"VAGEN parameters:")
    print(f"  - room_size: {room_size_tuple}")
    print(f"  - room_num: {room_num}")
    print(f"  - n_objects: {n_objects} (per room)")
    print(f"  - topology: {topology}")
    print(f"  - seed: {seed}")
        
    # Generate mask using VAGEN (also get room object for hash collision detection)
    try:
        grid, room = generate_room_layout_vagen(
            room_size=room_size_tuple,
            room_num=room_num,
            n_objects=n_objects,
            topology=topology,
            seed=seed,
            return_room=True
        )
        print(f"VAGEN mask generated: shape={grid.shape}")
    except Exception as e:
        print(f"❌ VAGEN generation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    # Save mask and return
    if run_name:
        mask_filename = f"{run_name}_mask.json"
    else:
        mask_filename = "generated_mask.json"
        
    mask_path = output_dir / mask_filename
    mask_data = grid.tolist()
        
    with open(mask_path, 'w') as f:
        json.dump(mask_data, f, indent=2)
        
    print(f"Mask saved to: {mask_path}")
    return mask_path, room              

def run_scene_generation(config: dict, mask_path: Path, output_dir: Path, scene_seed: int, config_path: Path, run_seed: int = None):
    """Run scene generation directly"""
    try:
        import subprocess
        
        scene_config = config.get('scene_generation', {})
        obj_config = config.get('object_generation', {})
        # Fixed parameters (hidden from config)
        fixed_scene = {
            "cell_size": 1.0,
            "wall_thickness": 0.01,
            "wall_height": 2.0,
            "door_width": 0.6,
            "with_ray": False,
            "enable_gravity_fix": True,
            "physics_settle_time": 0.1,
        }
        fixed_object_mode = "fixed"
        vagen_root, vagen_base = _get_vagen_paths(config)
        
        # Build command arguments
        cmd = [
            'python', 'scene/mask2scene_enhanced.py',
            '--mask_path', str(mask_path),
            '--output', str(output_dir),
            '--seed', str(scene_seed),
            '--cell_size', str(fixed_scene["cell_size"]),
            '--wall_thickness', str(fixed_scene["wall_thickness"]),
            '--wall_height', str(fixed_scene["wall_height"]),
            '--door_width', str(fixed_scene["door_width"]),
            '--overall_scale', str(scene_config.get('overall_scale', 0.75)),
            '--port', str(scene_config.get('port', 1071)),
            '--physics_settle_time', str(fixed_scene["physics_settle_time"])
        ]
        
        # Add run_seed if provided
        if run_seed is not None:
            cmd.extend(['--run_seed', str(run_seed)])
        
        # Add object generation parameters
        total_objects = obj_config.get('total_objects', 10)
        cmd.extend(['--total_objects', str(total_objects)])
        
        mode = fixed_object_mode
        cmd.extend(['--object_mode', mode])

        if mode == 'fixed':
            fix_object_n = obj_config.get('fix_object_n', [])
            fix_object_str = ','.join(map(str, fix_object_n))
            cmd.extend(['--fix_object_n', fix_object_str])
        elif mode == 'proportional':
            if obj_config.get('proportional_to_area', False):
                cmd.append('--proportional_to_area')
        
        # Add optional parameters
        if fixed_scene["with_ray"]:
            cmd.append('--with_ray')

        if not fixed_scene["enable_gravity_fix"]:
            cmd.append('--disable_gravity_fix')
        
        # Add model system parameters (new version)
        if scene_config.get('builtin_models_path'):
            cmd.extend(['--builtin_models_path', scene_config.get('builtin_models_path')])
        
        if scene_config.get('custom_models_path'):
            cmd.extend(['--custom_models_path', scene_config.get('custom_models_path')])
        
        # Add custom models parameters (legacy, for backward compatibility)
        if scene_config.get('custom_models_config'):
            cmd.extend(['--custom_models_config', scene_config.get('custom_models_config')])
        
        if scene_config.get('disable_custom_models', False):
            cmd.append('--disable_custom_models')
        
        # Add configuration file path
        cmd.extend(['--config', str(config_path)])
        
        # Execute command
        env = os.environ.copy()
        if vagen_root:
            env["VAGEN_ROOT"] = vagen_root
        if vagen_base:
            env["VAGEN_BASE"] = vagen_base
        result = subprocess.run(cmd, check=True, cwd='.', env=env)
        
        if result.returncode == 0:
            print(f"Scene generation successful: {output_dir}")
        else:
            print(f"❌ Scene generation failed")
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Scene generation failed: {e}")
        raise
    except Exception as e:
        print(f"❌ Scene generation error: {e}")
        raise



def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Multiroom Scene Generation Pipeline")
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="Config file path (default: config.yaml)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output directory (default: read from config)")
    parser.add_argument("--layout-only", action="store_true",
                       help="Generate room layout only, skip scene generation")
    parser.add_argument("--skip-validation", action="store_true",
                       help="Skip config file validation")

    # Below are for False-belief scene generation (from existing runs)
    parser.add_argument("--falsebelief-exp", action="store_true",
                       help="Generate falsebelief_exp.json for existing runs (no rendering)")
    parser.add_argument("--fb-runs-root", type=str,
                       help="Root directory containing runXX folders (e.g., tos_dataset_1214_3room_100runs)")
    parser.add_argument("--fb-runs", type=str, default="0-24",
                       help="Run range 'start-end' (default: 0-24)")
    parser.add_argument("--fb-meta-name", type=str, default="falsebelief_exp.json",
                       help="Output meta filename inside each run (default: falsebelief_exp.json)")
    parser.add_argument("--fb-mod-type", type=str, default="auto", choices=["auto", "move", "rotate"],
                       help="ObjectModifier mode (default auto)")
    parser.add_argument("--fb-render", action="store_true",
                       help="After generating falsebelief_exp.json, render scenes with _fbexp suffix")
    parser.add_argument("--fb-suffix", type=str, default="_fbexp",
                       help="Suffix for rendered images (default _fbexp)")
    parser.add_argument("--fb-port", type=int, default=1071,
                       help="TDW port for fb rendering")
    # below false-belief arguments can be omitted if you pass --config                   
    parser.add_argument("--fb-builtin-models-path", type=str, default=None,
                       help="Builtin models path for fb rendering (optional)")
    parser.add_argument("--fb-custom-models-path", type=str, default=None,
                       help="Custom models path for fb rendering (optional)")
    parser.add_argument("--fb-door-record-path", type=str, default=None,
                       help="Door record json path for fb rendering (optional)")
    
    args = parser.parse_args()
    
    print("🏗️  Multiroom Scene Generation Pipeline")
    print("="*60)
    print(f"Config file: {args.config}")
    
    # Load and validate config
    config_path = Path(args.config)
    config = load_config(args.config)
    
    if not args.skip_validation:
        if not validate_config(config):
            sys.exit(1)

    # False-belief mode: only generate falsebelief_exp.json for existing runs
    if args.falsebelief_exp:
        if not args.fb_runs_root:
            print("❌ --fb-runs-root is required when --falsebelief-exp is set")
            sys.exit(1)
        fb_root = Path(args.fb_runs_root).expanduser().resolve()
        if not fb_root.exists():
            print(f"❌ fb_runs_root not found: {fb_root}")
            sys.exit(1)
        # parse range
        try:
            s, e = [int(x) for x in args.fb_runs.split("-")]
        except Exception:
            print(f"❌ Bad --fb-runs '{args.fb_runs}', expected 'start-end'")
            sys.exit(1)
        run_ids = list(range(s, e + 1))
        print(f"False-belief mode: runs {run_ids}, root={fb_root}")
        # lazy imports
        from scene.apply_false_belief_to_meta import load_tos_meta, to_vagen_room, apply_changes_to_meta
        from vagen.env.spatial.Base.tos_base.utils.room_modifier import ObjectModifier
        from scene.falsebelief_render import render_fb_runs
        fb_builtin = args.fb_builtin_models_path
        fb_custom = args.fb_custom_models_path
        fb_door = args.fb_door_record_path
        scene_cfg = config.get("scene_generation", {})
        if fb_builtin is None:
            fb_builtin = scene_cfg.get("builtin_models_path")
        if fb_custom is None:
            fb_custom = scene_cfg.get("custom_models_path")
        if fb_door is None:
            fb_door = scene_cfg.get("door_record_path")
        for rid in run_ids:
            run_name = f"run{rid:02d}"
            run_dir = fb_root / run_name
            meta_path = run_dir / "meta_data.json"
            if not meta_path.exists():
                print(f"Skip {run_name}: meta_data.json not found")
                continue
            out_path = run_dir / args.fb_meta_name
            print(f"{run_name}: applying false-belief changes -> {out_path}")
            meta = load_tos_meta(Path(meta_path))
            room, agent = to_vagen_room(meta)
            modifier = ObjectModifier(
                seed=rid,
                agent_pos=getattr(agent, "pos", None),
            )
            modified_room, changes = modifier.modify(room)
            meta_out = apply_changes_to_meta(meta, modified_room, changes)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(meta_out, indent=2))
            print(f"Saved {out_path} with changes: {[c.to_dict() for c in changes]}")
        if args.fb_render:
            render_fb_runs(
                root=fb_root,
                run_ids=run_ids,
                fb_meta_name=args.fb_meta_name,
                fb_suffix=args.fb_suffix,
                port=args.fb_port,
                config=config,
                builtin_models_path=fb_builtin,
                custom_models_path=fb_custom,
                door_record_path=fb_door,
            )
        return
    
    # Determine output directory
    if args.output:
        output_base = Path(args.output)
    else:
        output_base = Path(config.get('output', {}).get('base_dir', './output'))
    
    # Use base_dir directly as output directory, no nesting
    output_dir = output_base
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # Run batch generation with different layout seed for each run
    print("\n" + "="*60)
    print("Starting batch scene generation")
    print("="*60)
    
    batch_config = config.get('batch', {})
    num_runs = batch_config.get('num_runs', 3)
    seed_start = batch_config.get('seed_start', 60)
    seed_increment = batch_config.get('seed_increment', 100)
    run_offset = batch_config.get('run_offset', 0) 
    
    print(f"Batch generation parameters:")
    print(f"  - Number of runs: {num_runs}")
    print(f"  - Starting seed: {seed_start}")
    print(f"  - Seed increment: {seed_increment}")
    print(f"  - Run offset: {run_offset}")
    print()
    
    if args.layout_only:
        print("Layout-only mode: generating different layout previews for each run...")
        print("Note: Each run's layout seed will be consistent with scene seed in full generation")
        for i in range(num_runs):
            run_seed = seed_start + i * seed_increment
            run_num = i + run_offset
            print(f"\nGenerating run{run_num:02d} layout (seed={run_seed}, for full generation)")
            
            # Generate different layout for each run
            run_config = config.copy()
            run_config['room_layout']['seed'] = run_seed
            run_config['object_generation']['seed'] = run_seed
            
            # Generate and save layout
            temp_mask_path, room = generate_room_layout_from_config(run_config, output_dir, run_name=f"run{i:02d}")
            
            # Check for hash collision and warn
            has_collision, num_doors, collisions = _check_door_hash_collision(room)
            if has_collision:
                collision_details = "; ".join([f"hash={h} at {p1} and {p2}" for h, p1, p2 in collisions])
                print(f"  [WARN] Hash collision: {collision_details}")
            
            print(f"run{i:02d} layout generated: {temp_mask_path}")
        
        print(f"\nLayout-only mode completed")
        print(f"Output directory: {output_dir}")
        return
    
    # Full scene generation mode
    instruction_src = Path(__file__).resolve().parent / "models" / "instruction.png"
    if instruction_src.exists():
        import shutil
        shutil.copy2(instruction_src, output_dir / instruction_src.name)

    seed_offset = 0
    for i in range(num_runs):
        run_seed = seed_start + seed_offset + i * seed_increment
        run_num = i + run_offset
        run_name = f"run{run_num:02d}"
        run_output_dir = output_dir / run_name
        run_output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n🔧 Generating {run_name}: seed={run_seed} (layout + scene)")
        
        # Generate different layout for each run using unified seed
        run_config = config.copy()
        run_config['room_layout']['seed'] = run_seed
        run_config['object_generation']['seed'] = run_seed
        
        # Temporary directory for mask generation
        temp_dir = output_dir / f".temp_{run_name}"
        temp_dir.mkdir(exist_ok=True)
        
        try:
            bumped = False
            while True:
                # Generate layout for this run (using run_seed)
                mask_path, room = generate_room_layout_from_config(run_config, temp_dir, run_name=run_name)

                # Check for door position hash collision
                has_collision, num_doors, collisions = _check_door_hash_collision(room)
                if has_collision and not bumped:
                    collision_details = "; ".join([f"hash={h} at {p1} and {p2}" for h, p1, p2 in collisions])
                    print(
                        f"[WARN] Door hash collision detected for {run_name} ({num_doors} doors): "
                        f"{collision_details}. Bumping seed by 11."
                    )
                    seed_offset += 11
                    run_seed = seed_start + seed_offset + i * seed_increment
                    run_config['room_layout']['seed'] = run_seed
                    run_config['object_generation']['seed'] = run_seed
                    print(f"[INFO] {run_name} new seed={run_seed} (offset applied)")
                    bumped = True
                    continue
                if has_collision and bumped:
                    collision_details = "; ".join([f"hash={h} at {p1} and {p2}" for h, p1, p2 in collisions])
                    print(
                        f"[WARN] Door hash collision persists for {run_name} "
                        f"({collision_details}). Proceeding anyway."
                    )
                break

            # Run scene generation directly (using same run_seed)
            # Pass run_num as run_seed for metadata
            run_scene_generation(config, mask_path, run_output_dir, run_seed, config_path, run_seed=run_num)
            print(f"{run_name} completed (unified seed: {run_seed})")

        except Exception as e:
            print(f"❌ {run_name} failed: {e}")
        finally:
            # Clean up temporary directory
            import shutil
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
    
    print("\n" + "="*60)
    print("Pipeline execution completed!")
    print("="*60)
    print(f"All outputs saved to: {output_dir}")

if __name__ == "__main__":
    main()
