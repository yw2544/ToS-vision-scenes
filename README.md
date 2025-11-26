# ToS Data Generation Pipeline

This module wraps the full “Mask → Layout → TDW Scene” flow used by ToS. It includes:

- **Layout generation** via VAGEN or the legacy generator.
- **Custom model support** (asset bundles built from `models/model_import`).
- **Scene validation** (orientation tasks, navigation tasks, etc.).

The sections below describe the required preparation, how to run the pipeline, and which
configuration fields you are most likely to tweak.

---

## 1. Prerequisites

| Requirement | Notes |
|-------------|-------|
| **Python 3.10+** | Install dependencies from your preferred environment manager (conda/venv/poetry). |
| **TDW** | The TDW Python package plus a Unity build of TDW 1.12+ (or any version compatible with `tdw.controller`). |
| **Unity Editor** | Required for asset bundle creation. Supply the binary path when running the bundler scripts. |
| **`assimp` CLI** | Used to convert FBX to OBJ for mesh cleanup (`brew install assimp`). |
| **VAGEN repository** | Clone the official VAGEN repo somewhere on disk; the absolute path is referenced in `config.yaml` (see below). |
| **Git LFS** | Needed to download the model library dataset from Hugging Face. |

---

## 2. Preparing assets (model import + door import)

1. **Download the raw meshes**
   ```bash
   cd tos_data_gen/models/model_import
   huggingface-cli login                          # if not already logged in
   huggingface-cli download \
     yw12356/ToS_model_lib \
     --repo-type dataset \
     --local-dir model_lib
   ```
   The dataset preview is available at  
   `https://huggingface.co/datasets/yw12356/ToS_model_lib`.

2. **Build TDW asset bundles (custom models + door)**
   ```bash
   chmod +x build_all_bundles.sh
   ./build_all_bundles.sh \
     tos_data_gen/models/door_record.json \
     tos_data_gen/models/model_import/model_record \
     /Applications/Unity/Hub/Editor/2022.3.61f1c1/Unity.app/Contents/MacOS/Unity
   ```
   - The script calls `build_bundles.py` to process every entry in
     `models/custom_models.json`.
   - Door assets share the same build pipeline; the resulting `door_record.json` is
     consumed by `mask2scene` during TDW scene building.
   - If you need to rebuild only a subset of models, call
     `python build_bundles.py --only-model <model_name>` directly.

3. **Configure TDW + VAGEN paths**
   - Ensure your environment can import TDW (`pip install tdw`, or add TDW’s `Python`
     directory to `PYTHONPATH`).
   - Set `scene_generation.door_record_path`, `scene_generation.custom_models_path`,
     and `scene_generation.builtin_models_path` in `config.yaml` to match the files
     on disk.
   - Fill `vagen.path` with the absolute path to the VAGEN repo root if you plan to
     use the new VAGEN room generator.

---

## 3. Running the pipeline

From the project root:

```bash
python -m tos_data_gen.pipeline \
  --config tos_data_gen/config.yaml \
  --output tos_data_gen/aaa_dataset_test_seed \
  --layout-only   # optional flag; omit to run full TDW scenes
```

Pipeline arguments:

- `--config`: Path to the YAML config (defaults to `tos_data_gen/config.yaml`).
- `--output`: Overrides `output.base_dir` inside the YAML (optional).
- `--layout-only`: Stops after mask/layout generation (no TDW rendering).

The script prints progress for each batch run (seed, room generation, validation, etc.).

---

## 4. Key configuration fields (`config.yaml`)

### Layout generation (`room_layout`)

| Field | Meaning |
|-------|---------|
| `use_vagen` | Toggle between the new VAGEN generator and the legacy mask builder. |
| `room_size_tuple` | `(width, height)` per room (affects grid/mask size). |
| `room_num` | Number of rooms in each run. |
| `n_objects` | Objects per room (used by the VAGEN generator). |
| `topology` | Connection pattern (0 = main room connects to one room, 1 = two rooms). |

Legacy generator fields (used when `use_vagen=false`):
`room_size`, `level`, `main_room_size`, etc.

### Object generation (`object_generation`)

| Field | Meaning |
|-------|---------|
| `mode` | `"fixed"` uses `fix_object_n` per room; `"total"` or `"proportional"` distributes counts globally. |
| `fix_object_n` | List of per-room object counts (only when `mode="fixed"`). |
| `total_objects` | Global target across rooms (default for `"total"` mode). |
| `proportional_to_area` | When true, scales counts by room area in proportional mode. |

### Scene generation (`scene_generation`)

| Field | Meaning |
|-------|---------|
| `port` | TDW controller port. |
| `overall_scale` | Uniform scale factor applied to custom models. |
| `builtin_models_path`, `custom_models_path` | JSON catalogs consumed by `ObjectGenerator`. |
| `door_record_path` | Path to `door_record.json` produced by the bundler. |
| `disable_custom_models` | If true, only builtin TDW models will be used. |

### Batch settings (`batch`)

| Field | Meaning |
|-------|---------|
| `num_runs` | Number of seeds to generate per invocation. |
| `seed_start`, `seed_increment` | Control the RNG seeds for reproducibility. |
| `run_offset` | Starting index for run folder naming. |

### Output + debugging

| Field | Meaning |
|-------|---------|
| `output.base_dir` | Root directory where run folders are created. |
| `output.save_mask`, `save_layout_debug`, `save_scene_metadata` | Toggle intermediate artifacts. |
| `debug.object_generator_logs` | Enable verbose spatial debugging (door distances, etc.). |

### VAGEN integration (`vagen`)

| Field | Meaning |
|-------|---------|
| `path` | Absolute filesystem path to the VAGEN repo. Leave empty if you rely on legacy layout generation. |

### Validation (`text_based_validity` + `validation/ragen`)

The YAML already lists the supported tasks. Adjust `enabled`, `max_retries`, and
individual task counts as needed. Pre-render validation consumes the mask, placed objects,
and agent pose before TDW rendering.

---

## 5. Optional presets / environment tips

- **VAGEN path**: After cloning the repo, set `vagen.path` to the folder containing
  `multi_room_gen/...`. You can also export an environment variable (e.g., `VAGEN_HOME`)
  and reference it in your own wrapper scripts.
- **TDW build selection**: If you maintain multiple TDW builds, override the port and
  `launch_build` options when instantiating `SceneGenerator` or patch TDW’s config to
  pick the correct executable.
- **Model paths**: You may keep `model_lib/` outside the repo and symlink it into
  `models/model_import/model_lib` if disk space is a concern. The bundler only requires
  consistent folder names (`<category>/<model_name>/source`).

---

## 6. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `record.json missing ...` during bundle creation | Check that the Unity Editor path is valid and that `assimp` produced OBJ/MTL pairs correctly. |
| TDW import errors (`module tdw not found`) | Ensure TDW’s Python package is installed/added to `PYTHONPATH`. |
| Door mesh not appearing in TDW | Verify `scene_generation.door_record_path` points to the `door_record.json` copied by `build_all_bundles.sh`. |
| VAGEN path errors | Re-check `vagen.path` in `config.yaml` or disable VAGEN by setting `use_vagen=false`. |

---

## 7. Repository layout (abridged)

```
tos_data_gen/
├── config.yaml                    # Main pipeline configuration
├── pipeline.py                    # Entry point
├── layout/                        # VAGEN + legacy layout generators
├── models/
│   ├── builtin_models.json
│   ├── custom_models.json
│   └── model_import/              # Bundler scripts + (downloaded) assets
├── scene/                         # TDW scene construction, object placement, etc.
├── utils/                         # VAGEN helpers and shared utilities
├── validation/                    # Pre-render validation (RAGEN)
└── README.md                      # (this file)
```

Feel free to adjust the config to match your workflow—most fields can be tuned without
editing code.

