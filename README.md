# ToS-vision-scenes

Visual scene generation module for [Theory of Space](https://github.com/williamzhangNU/Theory-of-Space). This module generates multi-room 3D environments using TDW (ThreeDWorld) for spatial reasoning experiments.

> **Note**: This module is part of the Theory-of-Space repository, located at `Theory-of-Space/ToS-vision-scenes/`.

> **Remote Server Requirement**: If running on a remote server, make sure the server has a **graphical display** attached (not headless). TDW and Unity require a display for rendering. For headless servers, you may need to set up a virtual display (e.g., Xvfb) or use GPU-accelerated remote rendering.

## Features

- **Room layout generation** via Theory-of-Space spatial environment
- **Custom 3D model support** (asset bundles built from `models/model_import`)
- **Scene validation** (orientation tasks, navigation tasks, etc.)
- **False-belief experiment generation** for belief updating tasks

---

## 1. Installation

> **Note**: `ToS-vision-scenes` is a git submodule of Theory-of-Space. When cloning the main repository, use `--recursive` to include it:
> ```bash
> git clone --recursive https://github.com/williamzhangNU/Theory-of-Space.git
> ```
> Or if already cloned, run: `git submodule update --init --recursive`

### Step 0: Set up Theory-of-Space (prerequisite)

This step assumes you have already cloned and set up the main Theory-of-Space repository:

```bash
cd Theory-of-Space
source setup.sh
```

This creates the `tos` conda environment with all base dependencies.

### Step 1: Install ToS-vision-scenes dependencies

> **Important**: Before running `setup.sh`, add your Hugging Face token in `Theory-of-Space/setup.sh` to enable model downloads. You can get a token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

```bash
conda activate tos
cd ToS-vision-scenes
source setup.sh
```

This installs additional packages (TDW, huggingface_hub) required for visual scene generation.

---

## 2. Unity Setup (Required for Custom Models)

TDW requires Unity Editor to build custom model asset bundles.

### 2.1 Install Unity Hub & Editor

<details>
<summary><b>Linux /Linux Servers(Recommended: use install.sh)</b></summary>

The `install.sh` script automatically downloads and installs Unity Hub, Unity Editor 2020.3.48, and all required dependencies:

```bash
chmod +x install.sh
./install.sh
```

</details>

<details>
<summary><b>Linux/ macOS / Windows (Manual installation)</b></summary>

1. Download and install [Unity Hub](https://unity.com/download)
2. Install Unity **2020.3.48** via Unity Hub. When installing, add build support for your target platforms (Windows, macOS, Linux)

</details>

### 2.2 Platform-specific dependencies

<details>
<summary><b>macOS</b></summary>

```bash
# assimp - for FBX to OBJ conversion
brew install assimp
```

</details>

<details>
<summary><b>Linux</b></summary>

If you did NOT use `install.sh`, install these manually:

```bash
# Required packages
sudo apt install libgconf-2-4 assimp-utils

# From ubuntu-toolchain ppa
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt install gcc-9 libstdc++6
```

For headless Linux servers, see [TDW Linux setup guide](https://github.com/threedworld-mit/tdw/blob/master/Documentation/lessons/setup/server.md).

</details>

<details>
<summary><b>Windows</b></summary>

- [Visual C++ 2012 Redistributable](https://www.microsoft.com/en-us/download/details.aspx?id=30679)

</details>

### 2.3 Activate Unity Hub & License (Linux)

After installation, activate Unity Hub and your license:

```bash
export DISPLAY=:0
unityhub --no-sandbox
```

A Unity Hub window will appear. You need to:
1. **Login** to your Unity account
2. **Activate** a Personal license (or your organization's license)
3. **Verify** that Unity Editor **2020.3.48f1** appears in the Unity Hub Editor list

### 2.4 Test Unity Editor Launch

Before building asset bundles, verify Unity Editor works correctly:

```bash
$HOME/Unity/Hub/Editor/2020.3.48f1/Editor/Unity \
  -batchmode -nographics -logFile /tmp/unity_editor.log
```

Check the log for any errors:

```bash
tail -n 200 /tmp/unity_editor.log
```

If Unity launches successfully (no errors in log), press `Ctrl+C` to exit.

---

## 3. Model Import

Before running the pipeline, build TDW asset bundles from the downloaded model assets.

> **Note**: Model assets are automatically downloaded during `source setup.sh`. If you need to re-download manually:
> ```bash
> huggingface-cli download yw12356/ToS_model_lib --repo-type dataset --local-dir models/model_import/model_lib
> ```
> Dataset preview: https://huggingface.co/datasets/yw12356/ToS_model_lib

### 3.1 Configure Unity path

Edit `config.yaml` to set your Unity Editor path. Below are **typical** installation paths (your actual path may vary):

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
  # unity_path: "/home/Unity/Hub/Editor/2020.3.48f1/Editor/Unity"                 # Linux
```

### 3.2 Build asset bundles

```bash
cd models/model_import
chmod +x build_all_bundles.sh
./build_all_bundles.sh
```

**Troubleshooting**: If the build fails or produces unexpected results, check the Unity Editor log:

```bash
# Unity Editor log location (Linux)
cat ~/.config/unity3d/Editor.log
```

For more options, see `models/model_import/README.md`.

---

## 4. Configuration

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

## 5. Running the Pipeline

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

## 6. False-Belief Experiment Mode

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

## 7. Repository Structure

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

## 8. Output Structure

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

## 9. Troubleshooting

| Symptom | Solution |
|---------|----------|
| `unity_path is missing` | Set `model_import.unity_path` in config.yaml |
| `record.json missing` | Check Unity path; verify `assimp` installed |
| `RuntimeError: scooter1: record.json missing for scooter1` when running `build_all_bundles.sh` | Check Unity Editor log at `~/.config/unity3d/Editor.log` for detailed error info |
| `module tdw not found` | Run `pip install tdw` or `source setup.sh` |
| `tos_paths` errors | Verify Theory-of-Space is properly installed |
| Door hash collision | Normal; pipeline auto-adjusts seed |

---

## 10. Related Resources

- [Theory-of-Space](https://github.com/williamzhangNU/Theory-of-Space) - Main benchmark repository
- [TDW Documentation](https://github.com/threedworld-mit/tdw) - ThreeDWorld simulator
- [TDW Custom Models](https://github.com/threedworld-mit/tdw/blob/master/Documentation/lessons/custom_models/custom_models.md) - Asset bundle creation guide
- [Asset Bundle Creator](https://github.com/alters-mit/asset_bundle_creator) - Standalone tool for converting source files to Unity asset bundles
- [Model Library](https://huggingface.co/datasets/yw12356/ToS_model_lib) - 3D model assets
