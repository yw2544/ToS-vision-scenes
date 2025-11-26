# Model Import Pipeline

This folder contains the tooling that converts raw meshes into TDW asset bundles and
updates `tos_data_gen/models/custom_models.json` with the corresponding record paths.

## Directory layout

```
tos_data_gen/models/model_import/
├── build_all_bundles.sh   # Convenience wrapper for batch builds
├── build_bundles.py       # Core pipeline logic
├── model_lib/             # Per-category source assets (downloaded separately)
└── model_record/          # Generated TDW bundles + records
```

The `model_lib/` directory is not tracked in Git because it contains ~800 meshes and
textures. Download the full asset pack from Hugging Face and unpack it here:

```bash
cd tos_data_gen/models/model_import
# Requires git-lfs; see https://huggingface.co/docs/hub/models-git-lfs
huggingface-cli login               # if needed
huggingface-cli download \
  yw12356/ToS_model_lib \
  --repo-type dataset \
  --local-dir model_lib
```

Each category has the structure `model_lib/<category>/<model_name>/source/*` with the
original `.fbx` or `.obj/.mtl` plus any textures. See the dataset preview for examples
of the expected assets [[source](https://huggingface.co/datasets/yw12356/ToS_model_lib)].

## Requirements

- Unity Editor (tested on 2022.3 LTS). Pass the binary via `--unity-path` or export it
  as `UNITY_PATH=/Applications/Unity/Hub/Editor/<version>/Unity.app/...`.
- `assimp` CLI (used for FBX→OBJ conversion). Install via `brew install assimp`.
- Python 3.9+ with TDW’s `ModelCreator` module available in `PYTHONPATH`.

*** For full guidance on setting up TDW’s model import environment, see the custom models tutorial and requirements in the TDW documentation.

## Batch usage

After downloading the models, run the provided shell script to generate every bundle
and refresh the JSON metadata:

```bash
cd /Users/songshe/objaverse_import
chmod +x tos_data_gen/models/model_import/build_all_bundles.sh
tos_data_gen/models/model_import/build_all_bundles.sh \
  tos_data_gen/models/door_record.json \
  tos_data_gen/models/model_import/model_record \
  /Applications/Unity/Hub/Editor/2022.3.61f1c1/Unity.app/Contents/MacOS/Unity
```

This command will:

1. Build all custom objects under `model_lib/`.
2. Update each `record` entry in `tos_data_gen/models/custom_models.json` to a path
   relative to `tos_data_gen/`.
3. Build the standalone door asset and copy its `record.json` to
   `tos_data_gen/models/door_record.json` (consumed by `mask2scene.py` via
   `scene_generation.door_record_path` in `config.yaml`).

## Advanced usage

- Rebuild only selected models:  
  `python build_bundles.py --only-model chair7 --only-model lamp2 ...`
- Rebuild only the door (e.g., after tweaking textures):  
  `python build_bundles.py --door-only --door-root model_lib/door --door-record-json ../door_record.json`
- Override directories (defaults shown):  
  `--models-json tos_data_gen/models/custom_models.json`  
  `--model-lib tos_data_gen/models/model_import/model_lib`  
  `--record-root tos_data_gen/models/model_import/model_record`

Refer to `python build_bundles.py --help` for the full list of flags.

