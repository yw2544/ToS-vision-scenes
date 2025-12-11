#!/usr/bin/env python3

import argparse
import json
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

    def _copy_textures(src_dir: Path, keep_subdirs: bool = False) -> int:
        """Copy textures from src_dir (flat or recursive) into work_dir.
        If the source dir is named 'textures', preserve that folder name so MTL refs like 'textures/foo.png' work."""
        if not src_dir.exists():
            return 0
        copied = 0
        if keep_subdirs:
            base = work_dir / (src_dir.name if src_dir.name.lower() == "textures" else "")
            for tex_file in src_dir.rglob("*"):
                if tex_file.is_file() and tex_file.suffix.lower() in TEXTURE_EXTENSIONS:
                    rel = tex_file.relative_to(src_dir)
                    dst = base / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(tex_file, dst)
                    copied += 1
        else:
    for ext in TEXTURE_EXTENSIONS:
                for tex_file in src_dir.glob(f"*{ext}"):
            shutil.copy2(tex_file, work_dir / tex_file.name)
                    copied += 1
        return copied

    # Textures alongside the model file
    texture_count += _copy_textures(model_path.parent)
    # Shared textures folder at model_root/textures (keep structure)
    texture_count += _copy_textures(model_root / "textures", keep_subdirs=True)
    # Nested textures folder under source/textures (keep structure)
    texture_count += _copy_textures(model_path.parent / "textures", keep_subdirs=True)

    if model_dst.suffix.lower() == ".obj":
        mtl_path = model_dst.with_suffix(".mtl")
        if mtl_path.exists():
            fix_mtl_texture_references(mtl_path, work_dir)
        else:
            print(f"[warn] {model_name}: converted OBJ has no MTL file")

    return model_dst


def fix_mtl_texture_references(mtl_path: Path, texture_dir: Path) -> None:
    available_textures = []
    for ext in TEXTURE_EXTENSIONS:
        available_textures.extend(texture_dir.glob(f"*{ext}"))

    if not available_textures:
        return

    try:
        content = mtl_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        print(f"[warn] Unable to read {mtl_path}: {exc}")
        return

    has_texture_maps = any(
        line.strip().startswith(
            ("map_Kd", "map_Ka", "map_Ks", "map_bump", "map_d", "map_Ns", "bump")
        )
        for line in content.splitlines()
    )

    if not has_texture_maps:
        auto_add_texture_references(mtl_path, available_textures, content)
        return

    lines = content.splitlines()
    modified = False

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith(
            ("map_Kd", "map_Ka", "map_Ks", "map_bump", "map_d", "map_Ns", "bump")
        ):
            continue

        if " " in stripped:
            cmd, _, ref = stripped.partition(" ")
            texture_ref = ref.strip()
        else:
            parts = stripped.split()
            if len(parts) < 2:
                continue
            cmd, texture_ref = parts[0], parts[-1]

        texture_name = texture_ref.replace("\\", "/").strip("\"'").split("/")[-1]

        replacement = None
        for tex in available_textures:
            if tex.name == texture_name:
                replacement = tex
                break

        if replacement is None:
            replacement = _match_texture_by_type(cmd.lower(), available_textures)

        if replacement:
            if replacement.name != texture_name:
                lines[idx] = f"{cmd} {replacement.name}"
                modified = True

    if modified:
        try:
            mtl_path.write_text("\n".join(lines), encoding="utf-8")
        except Exception as exc:
            print(f"[warn] Unable to write {mtl_path}: {exc}")


def auto_add_texture_references(mtl_path: Path, available_textures: List[Path], content: str) -> None:
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
        texture_lines.append(f"map_Kd {diffuse_tex}")
    if normal_tex:
        texture_lines.append(f"map_bump {normal_tex}")
    if metallic_tex:
        texture_lines.append(f"map_Ks {metallic_tex}")

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
    return textures[0] if textures else None


def ensure_assimp_available() -> None:
    try:
        result = subprocess.run(["which", "assimp"], capture_output=True, text=True)
        if result.returncode == 0:
            return
        print("[info] assimp not found. Trying to install via Homebrew...")
        subprocess.run(["brew", "install", "assimp"], check=True)
    except Exception as exc:
        print(f"[warn] Unable to verify or install assimp: {exc}")


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
    attempts = [
        ("standard", {
            "internal_materials": True,
            "write_physics_quality": False,
            "vhacd_resolution": 100000,
        }),
        ("external-materials", {
            "internal_materials": False,
            "write_physics_quality": False,
            "vhacd_resolution": 50000,
        }),
        ("minimal", {}),
    ]

    common_args = {
        "name": model_name,
        "source_file": str(prepared_model),
        "output_directory": str(output_directory),
        "scale_factor": scale,
        "cleanup": False,
        "validate": False,
    }

    last_error: Optional[Exception] = None
    for label, extra in attempts:
        try:
            args = dict(common_args)
            args.update(extra)
            mc.source_file_to_asset_bundles(**args)
            break
        except Exception as exc:
            last_error = exc
            print(f"[warn] {model_name}: {label} build failed ({exc})")
    else:
        raise RuntimeError(f"Failed to build asset bundle for {model_name}: {last_error}")

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

    unity_path = Path(args.unity_path).expanduser().resolve() if args.unity_path else find_unity_editor()
    if not unity_path or not unity_path.exists():
        raise FileNotFoundError("Unity editor not found. Please provide --unity-path.")

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

        for entry in models:
            model_name = entry.get("model_name")
            category = entry.get("category")
            if not model_name or not category:
                continue

            if only_models and model_name.lower() not in only_models:
                continue

            entry["record"] = None

            source_dir = model_lib_root / category / model_name / "source"
            if not source_dir.exists():
                failures.append((model_name, f"Missing source directory {source_dir}"))
                print(f"[warn] {model_name}: source directory missing ({source_dir})")
                continue

            model_file = select_source_model_file(source_dir, model_name)
            if not model_file:
                failures.append((model_name, f"No FBX/OBJ under {source_dir}"))
                print(f"[warn] {model_name}: no model file found in {source_dir}")
                continue

            scale = float(entry.get("scale", 1.0) or 1.0)

            try:
                record_path = build_bundle(model_file, mc, record_root, scale)
                entry["record"] = relative_record_path(record_path, tos_root)
                success += 1
                print(f"[ok] {model_name}: record saved to {entry['record']}")
            except Exception as exc:
                failures.append((model_name, str(exc)))
                print(f"[error] {model_name}: {exc}")

        write_models_json(models_json, models)

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
        print(f"Processed models: {len(models)}")
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

