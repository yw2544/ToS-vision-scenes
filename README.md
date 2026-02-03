# ToS-vision-scenes

Visual scene generation module for [Theory of Space](https://github.com/williamzhangNU/Theory-of-Space). This module generates multi-room 3D environments using TDW (ThreeDWorld) for spatial reasoning experiments.

> **Note**: This module is available on the `scene_gen` branch of Theory-of-Space (not included in the main release).

> **Pre-generated Dataset**: If you ran `source setup.sh` in the main Theory-of-Space repo, the 3-room dataset (100 runs, including false-belief data) is already downloaded to `room_data/`. You only need this module if you want to generate **custom** scenes.

## Features

- **Room layout generation** via Theory-of-Space spatial environment
- **Custom 3D model support** (asset bundles built from `models/model_import`)
- **Scene validation** (orientation tasks, navigation tasks, etc.)
- **False-belief experiment generation** for belief updating tasks

---

## Linux Server Setup

This section provides a streamlined setup for Linux servers (Ubuntu 18.04+). Most users running on remote servers should follow this guide.

### Requirements

- **Non-headless server** with a graphical display attached (physical or virtual via Xvfb)
- TDW and Unity require X11 display for rendering

### Step 1: Clone and set up Theory-of-Space (skip if already done)
```bash
git clone --single-branch --branch release https://github.com/williamzhangNU/Theory-of-Space.git
cd Theory-of-Space
git checkout scene_gen
git submodule update --init --recursive
source setup.sh
```

This creates the `tos` conda environment and downloads the pre-generated dataset.

### Step 2: Install ToS-vision-scenes dependencies

> **Optional**: Set `HF_TOKEN` in `setup.sh` to avoid Hugging Face rate limits (429 errors). Get a token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

```bash
conda activate tos
cd ToS-vision-scenes
source setup.sh
```

### Step 3: Install Unity Hub & Editor

The `install.sh` script automatically installs Unity Hub, Unity Editor 2020.3.24, and all required system dependencies:

```bash
chmod +x install.sh
./install.sh
```

This script handles:
- Unity Hub installation
- Unity Editor 2020.3.24f1 with Linux build support
- System packages: `libgconf-2-4`, `assimp-utils`, `gcc-9`, `libstdc++6`

### Step 4: Activate Unity License

Set up display and open Unity Hub:

```bash
export DISPLAY=:0
unityhub --no-sandbox
```

In the Unity Hub window:
1. **Login** to your Unity account
2. **Activate** a Personal license (or your organization's license)
3. **Verify** that Unity Editor **2020.3.24f1** appears in the Editor list

> **Headless servers**: If no physical display is available, set up Xvfb first. See [TDW Linux setup guide](https://github.com/threedworld-mit/tdw/blob/master/Documentation/lessons/setup/server.md).

### Step 5: Test Unity Editor

```bash
$HOME/Unity/Hub/Editor/2020.3.24f1/Editor/Unity \
  -batchmode -nographics -logFile /tmp/unity_editor.log
```

Check for errors:

```bash
tail -n 200 /tmp/unity_editor.log
```

If successful (no errors), press `Ctrl+C` to exit.

### Step 6: Configure and build asset bundles

Edit `config.yaml` to set Unity path:

```yaml
model_import:
  unity_path: "$HOME/Unity/Hub/Editor/2020.3.24f1/Editor/Unity"
```

Build asset bundles:

```bash
cd models/model_import
chmod +x build_all_bundles.sh
./build_all_bundles.sh
```

### Step 7: Test scene generation

```bash
cd ../..  # back to ToS-vision-scenes root
python generate_pipeline.py --config config.yaml
```

If successful, generated scenes will appear in the output directory.

---

## macOS / Windows Setup

For local development on macOS or Windows.

### Step 1: Clone and set up Theory-of-Space

```bash
git clone https://github.com/williamzhangNU/Theory-of-Space.git
cd Theory-of-Space
git checkout scene_gen
git submodule update --init --recursive
source setup.sh   # or setup.bat on Windows
```

### Step 2: Install ToS-vision-scenes dependencies

> **Optional**: Set `HF_TOKEN` in `setup.sh` to avoid Hugging Face rate limits (429 errors). Get a token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

```bash
conda activate tos
cd ToS-vision-scenes
source setup.sh
```

### Step 3: Install Unity Hub & Editor

1. Download and install [Unity Hub](https://unity.com/download)
2. Install Unity **2020.3.24** via Unity Hub
3. Add build support for your target platform (macOS or Windows)

### Step 4: Platform-specific dependencies

<details>
<summary><b>macOS</b></summary>

```bash
# assimp - for FBX to OBJ conversion
brew install assimp
```

</details>

<details>
<summary><b>Windows</b></summary>

- Install [Visual C++ 2012 Redistributable](https://www.microsoft.com/en-us/download/details.aspx?id=30679)

</details>

### Step 5: Configure Unity path

Edit `config.yaml`:

```yaml
model_import:
  # macOS
  unity_path: "/Applications/Unity/Hub/Editor/2020.3.24f1/Unity.app/Contents/MacOS/Unity"
  # Windows
  # unity_path: "C:/Program Files/Unity/Hub/Editor/2020.3.24f1/Editor/Unity.exe"
```

### Step 6: Build asset bundles

```bash
cd models/model_import
chmod +x build_all_bundles.sh   # macOS only
./build_all_bundles.sh
```

**Troubleshooting**: If the build fails, check the Unity Editor log:
- macOS: `~/Library/Logs/Unity/Editor.log`
- Windows: `%LOCALAPPDATA%\Unity\Editor\Editor.log`

### Step 7: Test scene generation

```bash
cd ../..  # back to ToS-vision-scenes root
python generate_pipeline.py --config config.yaml
```

If successful, generated scenes will appear in the output directory.

---

## Model Assets

Model assets are automatically downloaded during `source setup.sh`. If you need to re-download manually:

```bash
huggingface-cli download yw12356/ToS_model_lib --repo-type dataset --local-dir models/model_import/model_lib
```

Dataset preview: https://huggingface.co/datasets/yw12356/ToS_model_lib

For more build options, see `models/model_import/README.md`.

---

## Configuration

Edit `config.yaml` to configure the pipeline:

```yaml
# Theory-of-Space paths (relative to this module)
tos_paths:
  root: ".."                           # Theory-of-Space repo root
  base: "../vagen/env/spatial/Base"    # Spatial environment base

# Room layout settings
room_layout:
  room_size_tuple: [6, 6]    # Room dimensions
  room_num: 4                # Number of rooms
  n_objects: 4               # Objects per room
  topology: 2                # Connection pattern

# Scene generation
scene_generation:
  port: 1071                 # TDW port
  overall_scale: 0.6         # Object scale

# Batch settings
batch:
  num_runs: 25
  seed_start: 0
  seed_increment: 1
```

---

## Running the Pipeline

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
| `--output` | Override output directory |
| `--layout-only` | Generate layouts only, skip TDW rendering |
| `--skip-validation` | Skip config validation |

---

## False-Belief Experiment Mode

Generate modified scenes for belief updating experiments. This mode modifies existing scenes by moving or rotating objects.

### Usage

```bash
python generate_pipeline.py --config config.yaml \
  --falsebelief-exp \
  --fb-runs-root ./tos_dataset_output \
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
| `--falsebelief-exp` | Enable false-belief mode | (flag) |
| `--fb-runs-root` | Root directory with runXX folders | (required) |
| `--fb-runs` | Run range (e.g., `0-24`) | `0-24` |
| `--fb-meta-name` | Output metadata filename | `falsebelief_exp.json` |
| `--fb-mod-type` | Modification: `auto`, `move`, `rotate` | `auto` |
| `--fb-render` | Render modified scenes | (flag) |
| `--fb-suffix` | Image suffix | `_fbexp` |
| `--fb-port` | TDW port | `1071` |

### Workflow

1. Generate normal scene data:
   ```bash
   python generate_pipeline.py --config config.yaml
   ```

2. Generate false-belief data:
   ```bash
   python generate_pipeline.py --config config.yaml \
     --falsebelief-exp \
     --fb-runs-root ./tos_dataset_output \
     --fb-runs 0-24 \
     --fb-render
   ```

---

## Repository Structure

```
ToS-vision-scenes/
├── generate_pipeline.py          # Main entry point
├── config.yaml                   # Configuration
├── setup.sh                      # Setup script
├── setup.py                      # Package setup
├── requirements.txt              # Additional dependencies
├── models/
│   ├── builtin_models.json       # TDW built-in models
│   ├── custom_models.json        # Custom model catalog
│   ├── door_record.json          # Door asset record
│   └── model_import/             # Asset bundle tools
├── scene/                        # Scene generation
├── multi_room_generator/         # Object generation
├── validation/                   # Pre-render validation
└── utils/                        # Utilities
```

---

## Output Structure

Each run generates:

```
runXX/
├── meta_data.json           # Scene metadata
├── top_down_annotated.png   # Annotated top-down view
├── top_down.png             # Raw top-down view
├── agent_facing_*.png       # Agent perspective images
└── *_facing_*.png           # Object/door camera images
```

---

## Troubleshooting

| Symptom | Solution |
|---------|----------|
| `unity_path is missing` | Set `model_import.unity_path` in config.yaml |
| `record.json missing` | Check Unity path; verify `assimp` installed |
| `RuntimeError: scooter1: record.json missing for scooter1` when running `build_all_bundles.sh` | Check Unity Editor log at `~/.config/unity3d/Editor.log` for detailed error info |
| `module tdw not found` | Run `pip install tdw` or `source setup.sh` |
| `tos_paths` errors | Verify Theory-of-Space is properly installed |
| Door hash collision | Normal; pipeline auto-adjusts seed |

---

## Related Resources

- [Theory-of-Space](https://github.com/williamzhangNU/Theory-of-Space) - Main benchmark repository
- [TDW Documentation](https://github.com/threedworld-mit/tdw) - ThreeDWorld simulator
- [TDW Custom Models](https://github.com/threedworld-mit/tdw/blob/master/Documentation/lessons/custom_models/custom_models.md) - Asset bundle creation guide
- [Asset Bundle Creator](https://github.com/alters-mit/asset_bundle_creator) - Standalone tool for converting source files to Unity asset bundles
- [Model Library](https://huggingface.co/datasets/yw12356/ToS_model_lib) - 3D model assets
