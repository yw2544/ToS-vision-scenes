# Model Import Pipeline

This folder contains the tooling that converts raw meshes into TDW asset bundles and updates `models/custom_models.json` with the corresponding record paths.

## Directory Layout

```
models/model_import/
├── build_all_bundles.sh   # Convenience wrapper (reads Unity path from config.yaml)
├── build_bundles.py       # Core pipeline logic
├── model_lib/             # Per-category source assets (downloaded separately)
└── model_record/          # Generated TDW bundles + records
```

The `model_lib/` directory is not tracked in Git because it contains ~800 meshes and textures. Download the full asset pack from Hugging Face.

## Requirements

| Requirement | Notes |
|-------------|-------|
| **Unity Editor** | Recommended: **2020.3.48** (via Unity Hub). Set path in `config.yaml` under `model_import.unity_path` |
| **`assimp` CLI** | For FBX→OBJ conversion. macOS: `brew install assimp`, Linux: `sudo apt install assimp-utils` |
| **Python 3.10+** | With TDW's `ModelCreator` module available |
| **Git LFS** | Required to download models from Hugging Face |

For detailed installation instructions (Unity Hub, Unity Editor, platform-specific dependencies), see the [ToS-vision-scenes README](../README.md#2-unity-setup-required-for-custom-models).

For full guidance on asset bundle creation, see:
- [TDW custom models documentation](https://github.com/threedworld-mit/tdw/blob/master/Documentation/lessons/custom_models/custom_models.md)
- [Asset Bundle Creator](https://github.com/alters-mit/asset_bundle_creator) - standalone tool for converting source files to Unity asset bundles

## Step 1: Download Models

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

## Step 2: Configure Unity Path

In `config.yaml` (at repo root), set the Unity Editor path. Below are **typical** installation paths (your actual path may vary):

| Platform | Typical Unity Path |
|----------|-------------------|
| **macOS** | `/Applications/Unity/Hub/Editor/2020.3.48f1/Unity.app/Contents/MacOS/Unity` |
| **Windows** | `C:/Program Files/Unity/Hub/Editor/2020.3.48f1/Editor/Unity.exe` |
| **Linux** | `$HOME/Unity/Hub/Editor/2020.3.48f1/Editor/Unity` |

Example `config.yaml`:

```yaml
model_import:
  # Choose ONE of the following based on your platform:
  unity_path: "/Applications/Unity/Hub/Editor/2020.3.48f1/Unity.app/Contents/MacOS/Unity"  # macOS
  # unity_path: "C:/Program Files/Unity/Hub/Editor/2020.3.48f1/Editor/Unity.exe"           # Windows
  # unity_path: "/home/youruser/Unity/Hub/Editor/2020.3.48f1/Editor/Unity"                 # Linux
```

**Note**: Recommended Unity version is **2020.3.48** for best compatibility.

## Step 3: Build Asset Bundles

Run the build script from the `model_import` directory:

```bash
cd models/model_import
chmod +x build_all_bundles.sh
./build_all_bundles.sh
```

The script will:
1. Read `unity_path` from `config.yaml`
2. Build all custom objects under `model_lib/`
3. Update `models/custom_models.json` with record paths
4. Build the door asset and output to `models/door_record.json`

## Advanced Usage

### Rebuild specific models only

```bash
python build_bundles.py --only-model chair7 --only-model lamp2
```

### Rebuild door only

```bash
python build_bundles.py \
  --door-only \
  --door-root model_lib/door \
  --door-record-json ../door_record.json
```

### Override directories

```bash
python build_bundles.py \
  --models-json ../custom_models.json \
  --model-lib ./model_lib \
  --record-root ./model_record \
  --unity-path /path/to/Unity
```

### All available flags

```bash
python build_bundles.py --help
```

| Flag | Description |
|------|-------------|
| `--unity-path` | Path to Unity editor binary |
| `--models-json` | Path to custom_models.json |
| `--model-lib` | Path to model_lib directory |
| `--record-root` | Path to model_record output directory |
| `--only-model` | Process only specified model(s) |
| `--build-door` | Also build the door asset |
| `--door-only` | Only build the door asset |
| `--door-root` | Door model source directory |
| `--door-record-json` | Output path for door record.json |
| `--door-scale` | Scale factor for door asset (default: 0.12) |

## Troubleshooting

| Symptom | Solution |
|---------|----------|
| `unity_path is missing in config.yaml` | Add `model_import.unity_path` to config.yaml |
| `unity_path is invalid or not executable` | Check the Unity binary path exists |
| `record.json missing` | Verify Unity Editor path is valid |
| `RuntimeError: scooter1: record.json missing for scooter1` | Check Unity Editor log at `~/.config/unity3d/Editor.log` for detailed error info |
| `FBX to OBJ failed` | Check `assimp` is installed (macOS: `brew install assimp`, Linux: `sudo apt install assimp-utils`) |
| `missing MTL file` | Some models may not have materials; this is a warning only |
