#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 0 ]]; then
  echo "usage: $0" >&2
  exit 1
fi

resolve_path() {
  python3 - <<'PY' "$1"
import os
import sys
print(os.path.abspath(sys.argv[1]))
PY
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DOOR_RECORD_JSON="$(resolve_path "$MODELS_DIR/door_record.json")"
MODEL_RECORD_ROOT="$(resolve_path "$SCRIPT_DIR/model_record")"
CONFIG_PATH="$(resolve_path "$MODELS_DIR/../config.yaml")"

UNITY_PATH_CFG="$(python3 - <<'PY' "$CONFIG_PATH"
import sys
from pathlib import Path

cfg = Path(sys.argv[1])
if not cfg.exists():
    print("")
    sys.exit(0)

value = ""
in_block = False
for line in cfg.read_text(encoding="utf-8").splitlines():
    stripped = line.strip()
    if stripped.startswith("#") or not stripped:
        continue
    if not in_block and stripped == "model_import:":
        in_block = True
        continue
    if in_block and not line.startswith("  "):
        break
    if in_block and stripped.startswith("unity_path:"):
        value = stripped.split(":", 1)[1].strip().strip('"').strip("'")
        break
print(value)
PY
)"

if [[ -z "${UNITY_PATH_CFG}" ]]; then
  echo "error: model_import.unity_path is missing in $CONFIG_PATH" >&2
  exit 1
fi

UNITY_PATH_CFG="$(resolve_path "$UNITY_PATH_CFG")"
if [[ ! -x "$UNITY_PATH_CFG" ]]; then
  echo "error: unity_path is invalid or not executable: $UNITY_PATH_CFG" >&2
  exit 1
fi

UNITY_ARG=(--unity-path "$UNITY_PATH_CFG")

mkdir -p "$(dirname "$DOOR_RECORD_JSON")"
mkdir -p "$MODEL_RECORD_ROOT"

python3 "$SCRIPT_DIR/build_bundles.py" \
  --record-root "$MODEL_RECORD_ROOT" \
  --door-record-json "$DOOR_RECORD_JSON" \
  --build-door \
  "${UNITY_ARG[@]}"

