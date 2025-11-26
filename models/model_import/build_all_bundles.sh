#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 <door_record_json> <model_record_root> [unity_binary]" >&2
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
DOOR_RECORD_JSON="$(resolve_path "$1")"
MODEL_RECORD_ROOT="$(resolve_path "$2")"

UNITY_ARG=()
if [[ $# -eq 3 ]]; then
  UNITY_ARG=(--unity-path "$(resolve_path "$3")")
elif [[ -n "${UNITY_PATH:-}" ]]; then
  UNITY_ARG=(--unity-path "$(resolve_path "$UNITY_PATH")")
fi

mkdir -p "$(dirname "$DOOR_RECORD_JSON")"
mkdir -p "$MODEL_RECORD_ROOT"

python3 "$SCRIPT_DIR/build_bundles.py" \
  --record-root "$MODEL_RECORD_ROOT" \
  --door-record-json "$DOOR_RECORD_JSON" \
  --build-door \
  "${UNITY_ARG[@]}"

