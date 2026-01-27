# ToS Data Generation Pipeline

This module provides a complete "VAGEN Mask → TDW Scene" pipeline for generating multi-room spatial reasoning datasets. It includes:

- **Room layout generation** via VAGEN RoomGenerator
- **Custom model support** (asset bundles built from `models/model_import`)
- **Scene validation** (orientation tasks, navigation tasks, etc.)
- **False-belief experiment generation** for existing runs

---

## 1. Prerequisites

| Requirement | Notes |
|-------------|-------|
| **Python 3.10+** | Recommended to use conda environment with VAGEN dependencies |
| **TDW** | TDW Python package plus a Unity build of TDW 1.12+ |
| **Unity Editor** | Required for asset bundle creation (**recommended: 2020.3.24f1c2**) |
| **`assimp` CLI** | Used to convert FBX to OBJ (`brew install assimp`) |
| **VAGEN repository** | Clone the VAGEN repo; set paths in `config.yaml` |
| **Git LFS** | Needed to download the model library from Hugging Face |

---

## 2. Model Import (First Step)

Before running the pipeline, you must download the model assets and build TDW asset bundles.

### 2.1 Download raw meshes

```bash
cd models/model_import

# Requires git-lfs
huggingface-cli login               # if needed
huggingface-cli download \
  yw12356/ToS_model_lib \
  --repo-type dataset \
  --local-dir model_lib
```

Dataset preview: https://huggingface.co/datasets/yw12356/ToS_model_lib

Each category has the structure:
```
model_lib/<category>/<model_name>/source/*   # .fbx or .obj/.mtl + textures
```

### 2.2 Configure Unity path

In `config.yaml`, set the Unity Editor path:

```yaml
model_import:
  unity_path: "/Applications/Unity/Unity.app/Contents/MacOS/Unity"
```

**Note**: Recommended Unity version is **2020.3.24f1c2** for best compatibility with TDW model import.

### 2.3 Build TDW asset bundles

```bash
cd models/model_import
chmod +x build_all_bundles.sh
./build_all_bundles.sh
```

This will:
1. Read `unity_path` from `config.yaml`
2. Build all custom objects under `model_lib/`
3. Update `models/custom_models.json` with record paths
4. Build the door asset and output to `models/door_record.json`

### 2.4 Advanced model import options

Rebuild specific models only:
```bash
python build_bundles.py --only-model chair7 --only-model lamp2
```

Rebuild door only:
```bash
python build_bundles.py --door-only --door-root model_lib/door --door-record-json ../door_record.json
```

For more details, see `models/model_import/README.md`.

---

## 3. Configure Paths

Edit `config.yaml` to set VAGEN and model paths:

```yaml
vagen_paths:
  root: "/path/to/VAGEN"
  base: "/path/to/VAGEN/vagen/env/spatial/Base"

model_import:
  unity_path: "/Applications/Unity/Unity.app/Contents/MacOS/Unity"

scene_generation:
  builtin_models_path: "./models/builtin_models.json"
  custom_models_path: "./models/custom_models.json"
  door_record_path: "./models/door_record.json"
```

---

## 4. Running the Pipeline

### Basic usage

```bash
# Full scene generation (layout + TDW rendering)
python generate_pipeline.py --config config.yaml

# Layout-only mode (no TDW rendering)
python generate_pipeline.py --config config.yaml --layout-only

# Override output directory
python generate_pipeline.py --config config.yaml --output ./my_dataset
```

### Pipeline arguments

| Argument | Description |
|----------|-------------|
| `--config` | Path to YAML config (default: `config.yaml`) |
| `--output` | Override `output.base_dir` from config |
| `--layout-only` | Generate layouts only, skip TDW rendering |
| `--skip-validation` | Skip config file validation |

---

## 5. False-Belief Experiment Mode

After generating normal scene data, you can use this mode to create false-belief experiment data for VAGEN's agent belief updating tasks. This mode modifies existing scenes by moving or rotating objects, simulating scenarios where an agent's belief about object locations becomes outdated.

### What it does

1. Reads existing `meta_data.json` from each run
2. Uses VAGEN's `ObjectModifier` to apply changes (move or rotate objects)
3. Generates `falsebelief_exp.json` with the modified scene metadata
4. Optionally renders the modified scenes with TDW

### Basic usage

```bash
python generate_pipeline.py --config config.yaml \
  --falsebelief-exp \
  --fb-runs-root /path/to/existing_dataset \
  --fb-runs 0-24 \
  --fb-render
```

### Full example with all options

```bash
python generate_pipeline.py --config config.yaml \
  --falsebelief-exp \
  --fb-runs-root ./tos_dataset_2room_25runs \
  --fb-runs 0-24 \
  --fb-meta-name falsebelief_exp.json \
  --fb-mod-type auto \
  --fb-render \
  --fb-suffix _fbexp \
  --fb-port 1073 \
  --skip-validation
```

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--falsebelief-exp` | Enable false-belief mode | (required flag) |
| `--fb-runs-root` | Root directory containing runXX folders | (required) |
| `--fb-runs` | Run range in format `start-end` | `0-24` |
| `--fb-meta-name` | Output filename for modified metadata | `falsebelief_exp.json` |
| `--fb-mod-type` | Modification type: `auto`, `move`, or `rotate` | `auto` |
| `--fb-render` | Render modified scenes after generating metadata | (flag) |
| `--fb-suffix` | Suffix for rendered images | `_fbexp` |
| `--fb-port` | TDW port for rendering | `1071` |
| `--fb-builtin-models-path` | Override builtin models path | (from config) |
| `--fb-custom-models-path` | Override custom models path | (from config) |
| `--fb-door-record-path` | Override door record path | (from config) |

### Modification types

| Type | Description |
|------|-------------|
| `auto` | Automatically choose between move and rotate based on object properties |
| `move` | Move objects to new positions within the room |
| `rotate` | Rotate objects to face different directions |

### Output

For each run, the mode generates:
- `falsebelief_exp.json` - Modified scene metadata with object changes
- If `--fb-render` is used: rendered images with `_fbexp` suffix (e.g., `agent_facing_north_fbexp.png`)

### Workflow

1. **First**: Generate normal scene data
   ```bash
   python generate_pipeline.py --config config.yaml
   ```

2. **Then**: Generate false-belief data for those scenes
   ```bash
   python generate_pipeline.py --config config.yaml \
     --falsebelief-exp \
     --fb-runs-root ./tos_dataset_output \
     --fb-runs 0-24 \
     --fb-render
   ```

## 6. Configuration (`config.yaml`)

### VAGEN Paths

```yaml
vagen_paths:
  root: "/path/to/VAGEN"           # VAGEN repo root
  base: "/path/to/VAGEN/vagen/env/spatial/Base"
```

### Room Layout (`room_layout`)

| Field | Description |
|-------|-------------|
| `room_size_tuple` | `[width, height]` per room (e.g., `[6, 6]`) |
| `room_num` | Number of rooms (e.g., 3 or 4) |
| `n_objects` | Objects per room |
| `topology` | Connection pattern: 0=linear, 1=two connections, 2=three connections |
| `seed` | Base random seed (overridden by batch settings) |

### Object Generation (`object_generation`)

| Field | Description |
|-------|-------------|
| `total_objects` | Total objects across all rooms |
| `fix_object_n` | Per-room object counts (e.g., `[4, 4, 4, 4]`) |

### Scene Generation (`scene_generation`)

| Field | Description |
|-------|-------------|
| `port` | TDW controller port |
| `overall_scale` | Scale factor for objects (e.g., `0.6`) |
| `builtin_models_path` | Path to built-in models JSON |
| `custom_models_path` | Path to custom models JSON |
| `door_record_path` | Path to door record JSON |

### Batch Settings (`batch`)

| Field | Description |
|-------|-------------|
| `num_runs` | Number of runs to generate |
| `seed_start` | Starting seed |
| `seed_increment` | Seed increment between runs |
| `run_offset` | Starting index for run folder naming |

**Note**: If a door hash collision is detected for a seed, the pipeline automatically bumps the seed by 11 and regenerates. This offset applies to all subsequent runs.

### Output Settings (`output`)

| Field | Description |
|-------|-------------|
| `base_dir` | Root directory for output |
| `save_mask` | Save generated mask files |
| `save_layout_debug` | Save layout debug info |
| `save_scene_metadata` | Save scene metadata |
| `save_topdown_map` | Save topdown coordinate mapping |

### Validation (`text_based_validity`)

Configure pre-render validation tasks:

```yaml
text_based_validity:
  eval_tasks:
    - task_type: "dir"
      num: 1
    - task_type: "pov"
      num: 1
    - task_type: "rot"
      num: 1
    # ... more tasks
```

Supported task types: `dir`, `pov`, `bwd_pov_text`, `fwd_fov`, `bwd_nav_text`, `e2a`, `rot`, `fwd_loc`, `bwd_loc_text`

---

## 7. Repository Structure

```
ToS-vision-scenes/
├── generate_pipeline.py          # Main entry point
├── config.yaml                   # Pipeline configuration
├── models/
│   ├── builtin_models.json       # TDW built-in model catalog
│   ├── custom_models.json        # Custom model catalog (updated by build script)
│   ├── door_record.json          # Door asset record (generated by build script)
│   └── model_import/             # Asset bundle build tools
│       ├── build_all_bundles.sh  # Build script (reads Unity path from config)
│       ├── build_bundles.py      # Core build logic
│       ├── model_lib/            # Downloaded source assets (not in git)
│       ├── model_record/         # Generated bundles (not in git)
│       └── README.md
├── scene/
│   ├── room_layout_generator.py  # VAGEN mask generation
│   ├── mask2scene.py             # TDW scene building
│   ├── mask2scene_enhanced.py    # Enhanced scene builder
│   ├── falsebelief_render.py     # False-belief rendering
│   └── ...
├── multi_room_generator/         # Object generation utilities
├── validation/                   # Pre-render validation
├── utils/                        # Shared utilities
└── README.md
```

---

## 8. Troubleshooting

| Symptom | Solution |
|---------|----------|
| `unity_path is missing in config.yaml` | Add `model_import.unity_path` to config.yaml |
| `record.json missing` during bundle creation | Check Unity Editor path; verify `assimp` produced OBJ/MTL correctly |
| `module tdw not found` | Install TDW or add to `PYTHONPATH` |
| Door mesh not appearing | Verify `door_record_path` in config points to valid file |
| VAGEN import errors | Check `vagen_paths.root` and `vagen_paths.base` in config |
| Door hash collision warnings | Normal behavior; pipeline auto-bumps seed by 11 |
| `FBX to OBJ failed` | Check `assimp` is installed (`brew install assimp`) |

---

## 9. Output Structure

Each run generates:

```
runXX/
├── meta_data.json           # Scene metadata (objects, cameras, doors, etc.)
├── top_down_annotated.png   # Annotated top-down view
├── top_down.png             # Raw top-down view
├── agent_facing_*.png       # Agent perspective images
└── *_facing_*.png           # Object/door camera images
```
