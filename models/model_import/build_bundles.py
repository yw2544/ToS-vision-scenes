#!/usr/bin/env python3

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from tdw.asset_bundle_creator.model_creator import ModelCreator


TEXTURE_EXTENSIONS: Tuple[str, ...] = (
    ".png",
    ".jpg",
    ".jpeg",
    ".tga",
    ".bmp",
    ".tiff",
    ".tif",
    ".exr",
    ".hdr",
)


def prepare_model_files(model_path: Path, temp_dir: Path) -> Path:
    model_name = model_path.stem
    model_root = model_path.parent.parent

    work_dir = temp_dir / f"_work_{model_name}"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    if model_path.suffix.lower() == ".fbx":
        obj_path = work_dir / f"{model_name}.obj"
        try:
            result = subprocess.run(
                ["assimp", "export", str(model_path), str(obj_path)],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode == 0 and obj_path.exists():
                model_dst = obj_path
                mtl_path = obj_path.with_suffix(".mtl")
                if not mtl_path.exists() and result.stderr:
                    print(f"[warn] {model_name}: FBX converted but no MTL found ({result.stderr.strip()})")
            else:
                print(f"[warn] {model_name}: FBX to OBJ failed ({result.stderr.strip()})")
                model_dst = work_dir / model_path.name
                shutil.copy2(model_path, model_dst)
        except Exception as exc:
            print(f"[warn] {model_name}: FBX conversion error ({exc})")
            model_dst = work_dir / model_path.name
            shutil.copy2(model_path, model_dst)
    else:
        model_dst = work_dir / model_path.name
        shutil.copy2(model_path, model_dst)
        if model_path.suffix.lower() == ".obj":
            mtl_source = model_path.with_suffix(".mtl")
            if mtl_source.exists():
                shutil.copy2(mtl_source, work_dir / mtl_source.name)
            else:
                print(f"[warn] {model_name}: missing MTL file {mtl_source.name}")

    texture_count = 0
    texture_sources: List[Path] = []
    texture_sources.extend(_collect_textures(model_path.parent, recursive=False))

    source_textures_dir = model_path.parent / "textures"
    if source_textures_dir.exists():
        texture_sources.extend(_collect_textures(source_textures_dir, recursive=True))

    textures_dir = model_root / "textures"
    if textures_dir.exists():
        texture_sources.extend(_collect_textures(textures_dir, recursive=True))

    # Fallback: if nothing found, search source subdirs (e.g., Texture/)
    if not texture_sources:
        texture_sources.extend(_collect_textures(model_path.parent, recursive=True))

    texture_name_map: Dict[str, str] = {}
    used_texture_names: set[str] = set()

    if texture_sources:
        textures_subdir = work_dir / "textures"
        textures_subdir.mkdir(parents=True, exist_ok=True)
        for tex_file in texture_sources:
            target_name = _safe_texture_name(tex_file.name, used_texture_names)
            texture_name_map[tex_file.name] = target_name
            shutil.copy2(tex_file, work_dir / target_name)
            shutil.copy2(tex_file, textures_subdir / target_name)
            texture_count += 1

    if model_dst.suffix.lower() == ".obj":
        mtl_path = model_dst.with_suffix(".mtl")
        if mtl_path.exists():
            fix_mtl_texture_references(mtl_path, work_dir, texture_name_map)
        else:
            print(f"[warn] {model_name}: converted OBJ has no MTL file")

    return model_dst


def fix_mtl_texture_references(mtl_path: Path, texture_dir: Path, name_map: Optional[Dict[str, str]] = None) -> None:
    available_textures = _collect_textures(texture_dir, recursive=True)

    try:
        content = mtl_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        print(f"[warn] Unable to read {mtl_path}: {exc}")
        return

    has_texture_maps = any(
        line.strip().lower().startswith(
            ("map_kd", "map_ka", "map_ks", "map_bump", "map_d", "map_ns", "bump")
        )
        for line in content.splitlines()
    )

    if not has_texture_maps:
        auto_add_texture_references(mtl_path, available_textures, content, name_map=name_map)
        return

    lines = content.splitlines()
    modified = False

    current_material = ""
    for idx, line in enumerate(lines):
        stripped = line.strip()
        lowered = stripped.lower()
        if lowered.startswith("newmtl"):
            parts = stripped.split(maxsplit=1)
            current_material = parts[1] if len(parts) > 1 else ""
            continue
        if not lowered.startswith(
            ("map_kd", "map_ka", "map_ks", "map_bump", "map_d", "map_ns", "bump")
        ):
            continue

        parts = stripped.split()
        if len(parts) < 2:
            continue
        cmd = parts[0]
        option_tokens = parts[1:]
        texture_ref = _extract_texture_ref(option_tokens)
        matched_tex = None
        if not texture_ref:
            matched_tex = _find_texture_in_line(stripped, available_textures)
            if matched_tex:
                texture_ref = matched_tex.name
        replacement = None
        if not texture_ref:
            replacement = _match_texture_for_material(cmd.lower(), current_material, available_textures)
            if replacement is None:
                replacement = _match_texture_by_type(cmd.lower(), available_textures)
            if replacement is None:
                continue
            texture_name = replacement.name
        else:
            normalized_ref = texture_ref.replace("\\", "/").strip("\"'")
            texture_name = _basename_from_ref(normalized_ref)
        if replacement is None:
            for tex in available_textures:
                if tex.name == texture_name:
                    replacement = tex
                    break
        if replacement is None and texture_ref:
            replacement = _match_texture_by_ref(cmd.lower(), texture_ref, available_textures)
        if replacement is None:
            replacement = _match_texture_for_material(cmd.lower(), current_material, available_textures)
        if replacement is None:
            replacement = _match_texture_by_type(cmd.lower(), available_textures)

        if texture_name:
            new_name = replacement.name if replacement else texture_name
            new_name = _map_texture_name(new_name, name_map)
            new_name_escaped = _escape_texture_name(new_name)
            updated_line = _rewrite_map_line_with_options(stripped, cmd, option_tokens, new_name_escaped)
            if updated_line != stripped:
                lines[idx] = updated_line
                modified = True

    if modified:
        try:
            mtl_path.write_text("\n".join(lines), encoding="utf-8")
        except Exception as exc:
            print(f"[warn] Unable to write {mtl_path}: {exc}")


def auto_add_texture_references(
    mtl_path: Path,
    available_textures: List[Path],
    content: str,
    name_map: Optional[Dict[str, str]] = None,
) -> None:
    diffuse_tex = _first_matching_texture(available_textures, ["diffuse", "albedo", "basecolor", "color"])
    normal_tex = _first_matching_texture(available_textures, ["normal", "bump"])
    metallic_tex = _first_matching_texture(available_textures, ["metallic", "metal"])

    lines = content.splitlines()
    insert_index = len(lines)
    for i, line in enumerate(lines):
        if line.strip().startswith("newmtl"):
            insert_index = i + 1
            while insert_index < len(lines) and lines[insert_index].strip() and not lines[insert_index].strip().startswith("newmtl"):
                insert_index += 1
            break

    texture_lines = []
    if diffuse_tex:
        texture_lines.append(f"map_Kd {_escape_texture_name(_map_texture_name(diffuse_tex, name_map))}")
    if normal_tex:
        texture_lines.append(f"map_bump {_escape_texture_name(_map_texture_name(normal_tex, name_map))}")
    if metallic_tex:
        texture_lines.append(f"map_Ks {_escape_texture_name(_map_texture_name(metallic_tex, name_map))}")

    if not texture_lines:
        return

    updated = lines[:insert_index] + ["", "# Auto-added texture references"] + texture_lines + lines[insert_index:]
    try:
        mtl_path.write_text("\n".join(updated), encoding="utf-8")
    except Exception as exc:
        print(f"[warn] Unable to write {mtl_path}: {exc}")


def _first_matching_texture(textures: Sequence[Path], keywords: Sequence[str]) -> Optional[str]:
    for tex in textures:
        lower_name = tex.name.lower()
        if any(keyword in lower_name for keyword in keywords):
            return tex.name
    return None


def _match_texture_by_type(texture_type: str, textures: Sequence[Path]) -> Optional[Path]:
    keyword_map = {
        "map_kd": ["basecolor", "diffuse", "color", "albedo"],
        "map_bump": ["normal", "bump"],
        "map_ks": ["specular", "metallic"],
        "map_d": ["roughness", "smooth"],
        "map_ns": ["roughness", "smooth", "gloss"],
    }
    for keyword in keyword_map.get(texture_type, []):
        for tex in textures:
            if keyword in tex.name.lower():
                return tex
    return _pick_preferred_texture(texture_type, textures)


def _match_texture_by_ref(texture_type: str, ref: str, textures: Sequence[Path]) -> Optional[Path]:
    if not ref or not textures:
        return None
    ref_tokens = _tokenize_name(ref)
    ref_norm = _normalize_name(ref)
    best = None
    best_score = -1
    for tex in textures:
        name = tex.name
        name_tokens = _tokenize_name(name)
        if not name_tokens:
            continue
        overlap = ref_tokens.intersection(name_tokens)
        score = sum(len(tok) for tok in overlap)
        if _normalize_name(name) in ref_norm:
            score += 5
        score += _texture_type_bonus(texture_type, name)
        if score > best_score:
            best_score = score
            best = tex
    return best if best_score > 0 else None


def _match_texture_for_material(texture_type: str, material_name: str, textures: Sequence[Path]) -> Optional[Path]:
    if not textures:
        return None
    material_key = _normalize_name(material_name)
    if not material_key:
        return None
    keyword_map = {
        "map_kd": ["basecolor", "diffuse", "color", "albedo"],
        "map_bump": ["normal", "bump"],
        "map_ks": ["specular", "metallic"],
        "map_d": ["roughness", "smooth"],
        "map_ns": ["roughness", "smooth", "gloss"],
    }
    candidates = []
    for tex in textures:
        name = tex.name
        name_key = _normalize_name(name)
        if material_key in name_key:
            score = 2
            for keyword in keyword_map.get(texture_type, []):
                if keyword in name_key:
                    score = 0
                    break
            candidates.append((score, -len(name), tex))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][2]


def _extract_texture_ref(tokens: Sequence[str]) -> Optional[str]:
    for token in reversed(tokens):
        cleaned = token.strip().strip("\"'")
        normalized = cleaned.replace("\\", "/")
        for ext in TEXTURE_EXTENSIONS:
            if normalized.lower().endswith(ext):
                return cleaned
    return None


def _find_texture_in_line(line: str, textures: Sequence[Path]) -> Optional[Path]:
    if not textures:
        return None
    lower_line = line.lower()
    matches = [tex for tex in textures if tex.name.lower() in lower_line]
    if not matches:
        return None
    return max(matches, key=lambda t: len(t.name))


def _replace_texture_in_line(line: str, old_name: str, new_name: str) -> str:
    if not old_name:
        return line
    pattern = re.compile(re.escape(old_name), re.IGNORECASE)
    return pattern.sub(new_name, line, count=1)


def _rewrite_map_line_with_options(line: str, cmd: str, tokens: Sequence[str], new_name: str) -> str:
    if not cmd:
        return line
    option_arg_counts = {
        "-o": 3,
        "-s": 3,
        "-t": 3,
        "-mm": 2,
        "-bm": 1,
        "-clamp": 1,
        "-blendu": 1,
        "-blendv": 1,
        "-boost": 1,
        "-imfchan": 1,
        "-type": 1,
        "-texres": 1,
    }
    kept = [cmd]
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith("-"):
            kept.append(tok)
            count = option_arg_counts.get(tok, None)
            if count is None:
                # Heuristic: keep the next token if it doesn't look like another option.
                if i + 1 < len(tokens) and not tokens[i + 1].startswith("-"):
                    kept.append(tokens[i + 1])
                    i += 2
                    continue
                i += 1
                continue
            for j in range(count):
                if i + 1 + j < len(tokens):
                    kept.append(tokens[i + 1 + j])
            i += 1 + count
            continue
        break
    kept.append(new_name)
    return " ".join(kept)


def _basename_from_ref(ref: str) -> str:
    if not ref:
        return ref
    parts = re.split(r"[\\/]+", ref)
    return parts[-1] if parts else ref


def _normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower()) if value else ""


def _tokenize_name(value: str) -> set[str]:
    if not value:
        return set()
    return {tok for tok in re.split(r"[^a-z0-9]+", value.lower()) if tok}


def _texture_type_bonus(texture_type: str, name: str) -> int:
    name_key = _normalize_name(name)
    if texture_type == "map_kd":
        if any(key in name_key for key in ("ao", "roughness", "metallic", "normal", "opacity", "specular", "gloss")):
            return -3
        return 2
    if texture_type in ("map_bump", "bump"):
        return 2 if any(key in name_key for key in ("normal", "bump")) else 0
    if texture_type in ("map_d", "map_ns"):
        return 2 if any(key in name_key for key in ("roughness", "smooth", "gloss")) else 0
    if texture_type == "map_ks":
        return 2 if any(key in name_key for key in ("metallic", "specular")) else 0
    return 0


def _pick_preferred_texture(texture_type: str, textures: Sequence[Path]) -> Optional[Path]:
    if not textures:
        return None
    ranked = []
    for tex in textures:
        name = tex.name
        score = _texture_type_bonus(texture_type, name)
        ranked.append((score, name.lower(), tex))
    ranked.sort(reverse=True)
    return ranked[0][2]


def _escape_texture_name(name: str) -> str:
    if not name:
        return name
    return name.replace(" ", "_")


def _map_texture_name(name: str, name_map: Optional[Dict[str, str]]) -> str:
    if not name or not name_map:
        return name
    return name_map.get(name, name)


def _safe_texture_name(name: str, used: set[str]) -> str:
    base = name.replace(" ", "_")
    if base not in used:
        used.add(base)
        return base
    stem = Path(base).stem
    suffix = Path(base).suffix
    idx = 1
    while True:
        candidate = f"{stem}_{idx}{suffix}"
        if candidate not in used:
            used.add(candidate)
            return candidate
        idx += 1


def _collect_textures(directory: Path, recursive: bool) -> List[Path]:
    pattern = "**/*" if recursive else "*"
    textures: List[Path] = []
    for candidate in directory.glob(pattern):
        if candidate.is_file() and candidate.suffix.lower() in TEXTURE_EXTENSIONS:
            textures.append(candidate)
    return textures


def ensure_assimp_available() -> None:
    """Check if assimp is available. Print install instructions if not found."""
    import platform
    try:
        result = subprocess.run(["assimp", "version"], capture_output=True, text=True)
        if result.returncode == 0:
            return
    except FileNotFoundError:
        pass
    
    system = platform.system()
    print("[error] assimp not found. Please install it:")
    if system == "Darwin":
        print("  macOS: brew install assimp")
    elif system == "Linux":
        print("  Linux: sudo apt install assimp-utils")
    elif system == "Windows":
        print("  Windows: Download from https://github.com/assimp/assimp/releases")
        print("           Or use: choco install assimp")
    else:
        print(f"  {system}: Install assimp from https://github.com/assimp/assimp")
    raise RuntimeError("assimp is required for FBX to OBJ conversion but not installed")


def select_source_model_file(source_dir: Path, model_name: str) -> Optional[Path]:
    candidates = sorted(source_dir.glob("*.fbx"), key=lambda p: p.name.lower())
    if candidates:
        return _pick_best_match(candidates, model_name)
    obj_candidates = sorted(source_dir.glob("*.obj"), key=lambda p: p.name.lower())
    if obj_candidates:
        return _pick_best_match(obj_candidates, model_name)
    return None


def _pick_best_match(files: Sequence[Path], model_name: str) -> Path:
    target = model_name.lower()

    def score(path: Path) -> Tuple[int, str]:
        stem = path.stem.lower()
        if stem == target:
            return (0, path.name)
        if stem.startswith(target):
            return (1, path.name)
        if target in stem:
            return (2, path.name)
        return (3, path.name)

    return min(files, key=score)


def build_bundle(model_path: Path, mc: ModelCreator, output_root: Path, scale: float) -> Path:
    model_name = model_path.stem
    output_directory = output_root / model_name
    if output_directory.exists():
        shutil.rmtree(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    temp_dir = output_root / "_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    prepared_model = prepare_model_files(model_path, temp_dir)
    mc.source_file_to_asset_bundles(
        name=model_name,
        source_file=str(prepared_model),
        output_directory=str(output_directory),
        internal_materials=True,
        scale_factor=scale,
        cleanup=False,
        validate=False,
        write_physics_quality=False,
        vhacd_resolution=100000,
    )

    record_dir = output_directory / model_name
    record_path = record_dir / "record.json"
    log_path = record_dir / "log.txt"

    if not record_path.exists():
        fallback_record = output_directory / "record.json"
        fallback_log = output_directory / "log.txt"
        if fallback_record.exists():
            record_path = fallback_record
            log_path = fallback_log
        else:
            if log_path.exists():
                try:
                    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
                    tail = "\n".join(lines[-50:])
                    print(f"[log] Last lines for {model_name}:\n{tail}")
                except Exception:
                    pass
            raise FileNotFoundError(f"record.json missing for {model_name} (expected at {record_path})")

    return record_path


def build_door_bundle(
    mc: ModelCreator,
    door_root: Path,
    record_root: Path,
    scale: float,
    output_record_json: Path,
    tos_root: Path,
) -> Path:
    door_root = door_root.resolve()
    source_dir = door_root / "source"
    if not source_dir.exists():
        raise FileNotFoundError(f"Door source directory missing: {source_dir}")

    model_file = select_source_model_file(source_dir, door_root.name or "door")
    if not model_file:
        raise FileNotFoundError(f"No FBX/OBJ found under {source_dir}")

    record_path = build_bundle(model_file, mc, record_root, scale)
    output_record_json = output_record_json.resolve()
    output_record_json.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(record_path, output_record_json)
    rel_path = relative_record_path(output_record_json, tos_root)
    print(f"[ok] door: record saved to {output_record_json} (relative: {rel_path})")
    return record_path


def find_unity_editor() -> Optional[Path]:
    candidates = [
        "/Applications/Unity/Unity.app/Contents/MacOS/Unity",
        "/Applications/Unity/Hub/Editor/2022.3.61f1c1/Unity.app/Contents/MacOS/Unity",
        "/Applications/Unity/Hub/Editor/2020.3.24f1c2/Unity.app/Contents/MacOS/Unity",
    ]
    for path in candidates:
        candidate = Path(path)
        if candidate.exists():
            return candidate
    return None


def read_models_json(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_models_json(path: Path, data: List[Dict]) -> None:
    tmp_path = path.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
    tmp_path.replace(path)


def relative_record_path(record_path: Path, tos_root: Path) -> str:
    try:
        return record_path.relative_to(tos_root).as_posix()
    except ValueError:
        return str(record_path)


def _read_unity_path_from_config(config_path: Path) -> Optional[Path]:
    if not config_path.exists():
        return None
    in_block = False
    for line in config_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not in_block and stripped == "model_import:":
            in_block = True
            continue
        if in_block and not line.startswith("  "):
            break
        if in_block and stripped.startswith("unity_path:"):
            value = stripped.split(":", 1)[1].strip().strip('"').strip("'")
            if value:
                return Path(value).expanduser()
            return None
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch model bundler for TDW custom assets.")
    parser.add_argument("--unity-path", type=str, help="Path to Unity editor binary.")
    parser.add_argument("--models-json", type=str, help="Path to custom_models.json.")
    parser.add_argument("--model-lib", type=str, help="Path to model_lib directory.")
    parser.add_argument("--record-root", type=str, help="Path to model_record output directory.")
    parser.add_argument("--only-model", action="append", default=[], help="Process only the given model_name values.")
    parser.add_argument("--build-door", action="store_true", help="Also build the standalone door asset.")
    parser.add_argument("--door-only", action="store_true", help="Only build the standalone door asset.")
    parser.add_argument("--door-root", type=str, help="Door model directory containing source/ and textures/.")
    parser.add_argument("--door-record-json", type=str, help="Destination door record.json used by mask2scene.")
    parser.add_argument("--door-scale", type=float, default=0.12, help="Scale factor to apply to the door asset.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    models_dir = script_dir.parent
    tos_root = models_dir.parent

    models_json = Path(args.models_json).expanduser().resolve() if args.models_json else (models_dir / "custom_models.json")
    model_lib_root = Path(args.model_lib).expanduser().resolve() if args.model_lib else (script_dir / "model_lib")
    record_root = Path(args.record_root).expanduser().resolve() if args.record_root else (script_dir / "model_record")

    unity_path = Path(args.unity_path).expanduser().resolve() if args.unity_path else None
    if unity_path is None:
        config_path = tos_root / "config.yaml"
        config_unity = _read_unity_path_from_config(config_path)
        if config_unity:
            unity_path = config_unity.resolve()
    if not unity_path or not unity_path.exists():
        raise FileNotFoundError(
            "Unity editor not found. Provide --unity-path or set model_import.unity_path in config.yaml."
        )

    process_custom_models = not args.door_only
    build_door_asset = args.build_door or args.door_only

    if process_custom_models and not models_json.exists():
        raise FileNotFoundError(f"custom_models.json not found at {models_json}")

    if not model_lib_root.exists():
        raise FileNotFoundError(f"model_lib directory not found at {model_lib_root}")

    record_root.mkdir(parents=True, exist_ok=True)

    door_root = Path(args.door_root).expanduser().resolve() if args.door_root else (model_lib_root / "door")
    door_record_json = Path(args.door_record_json).expanduser().resolve() if args.door_record_json else (models_dir / "door_record.json")

    ensure_assimp_available()

    mc = ModelCreator(unity_editor_path=str(unity_path), quiet=False)

    models: List[Dict] = []
    success = 0
    failures: List[Tuple[str, str]] = []

    if process_custom_models:
        models = read_models_json(models_json)
        only_models = {name.lower() for name in args.only_model} if args.only_model else set()

        build_targets: List[Dict] = []
        for entry in models:
            model_name = entry.get("model_name")
            category = entry.get("category")
            if not model_name or not category:
                continue
            if only_models and model_name.lower() not in only_models:
                continue
            build_targets.append(entry)

        missing_targets = only_models - {e.get("model_name", "").lower() for e in build_targets}
        if missing_targets:
            missing_list = ", ".join(sorted(missing_targets))
            raise ValueError(f"Model(s) not found in custom_models.json: {missing_list}")

        for entry in build_targets:
            model_name = entry.get("model_name")
            category = entry.get("category")

            entry["record"] = None

            source_dir = model_lib_root / category / model_name / "source"
            if not source_dir.exists():
                raise FileNotFoundError(f"{model_name}: source directory missing ({source_dir})")

            model_file = select_source_model_file(source_dir, model_name)
            if not model_file:
                raise FileNotFoundError(f"{model_name}: no model file found in {source_dir}")

            scale =  1.0

            try:
                record_path = build_bundle(model_file, mc, record_root, scale)
            except Exception as exc:
                raise RuntimeError(f"{model_name}: {exc}") from exc

            entry["record"] = relative_record_path(record_path, tos_root)
            success += 1
            write_models_json(models_json, models)
            print(f"[ok] {model_name}: record saved to {entry['record']}")

    door_status = None
    if build_door_asset:
        try:
            build_door_bundle(
                mc=mc,
                door_root=door_root,
                record_root=record_root,
                scale=float(args.door_scale),
                output_record_json=door_record_json,
                tos_root=tos_root,
            )
            door_status = "success"
        except Exception as exc:
            door_status = f"failed: {exc}"
            print(f"[error] door: {exc}")

    if process_custom_models:
        print(f"\n==== Summary ====")
        processed_count = len(build_targets)
        print(f"Processed models: {processed_count}")
        print(f"Successful builds: {success}")
        print(f"Failed builds: {len(failures)}")
        if failures:
            for name, reason in failures[:20]:
                print(f" - {name}: {reason}")
    else:
        print("\n==== Summary ====")
        print("Custom model build skipped (door-only mode).")

    if door_status:
        print(f"Door build status: {door_status}")


if __name__ == "__main__":
    main()

