"""Create the private, portable model bundle used by ``gaze-pipeline demo``."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

CHECKPOINT_SHA256 = "127ca3014d9ef27cdabc327b0145e2c66a1741642bb278fb41a432156a24225f"


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            project_root
            / "mlruns/1/8fcf7781bf534120befe6f385f4ee386/artifacts/training/"
            "checkpoints/best/best_weights.pt"
        ),
    )
    parser.add_argument(
        "--front-weights",
        type=Path,
        default=project_root / "models/blazegaze_mpiifacegaze.keras",
    )
    parser.add_argument(
        "--face-landmarker",
        type=Path,
        default=project_root / "models/face_landmarker_v2_with_blendshapes.task",
    )
    parser.add_argument(
        "--stage-dir",
        type=Path,
        default=project_root / "models/demo",
        help="현재 컴퓨터에서 demo가 바로 사용하는 폴더",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "release/live-demo-models.zip",
    )
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_bundle(args: argparse.Namespace) -> tuple[Path, Path]:
    sources = {
        "best_weights.pt": args.checkpoint.expanduser().resolve(strict=False),
        "blazegaze_mpiifacegaze.keras": args.front_weights.expanduser().resolve(strict=False),
        "face_landmarker.task": args.face_landmarker.expanduser().resolve(strict=False),
    }
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("bundle source file is missing:\n  " + "\n  ".join(missing))
    checkpoint_hash = _sha256(sources["best_weights.pt"])
    if checkpoint_hash != CHECKPOINT_SHA256:
        raise ValueError(
            "refusing to package a checkpoint other than the verified 83.1% model: "
            f"expected {CHECKPOINT_SHA256}, got {checkpoint_hash}"
        )

    stage_dir = args.stage_dir.expanduser().resolve(strict=False)
    stage_dir.mkdir(parents=True, exist_ok=True)
    manifest_files: list[dict[str, object]] = []
    for name, source in sources.items():
        destination = stage_dir / name
        shutil.copy2(source, destination)
        manifest_files.append(
            {"name": name, "sha256": _sha256(destination), "size_bytes": destination.stat().st_size}
        )
    manifest = {
        "schema_version": 1,
        "model": "process_data_pairwise 3x3 same-cell 83.137%",
        "training_run_id": "8fcf7781bf534120befe6f385f4ee386",
        "test_run_id": "579b79b77cc145fba234f53fa685306b",
        "files": manifest_files,
    }
    manifest_path = stage_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    readme_path = stage_dir / "README_MODELS.txt"
    readme_path.write_text(
        "Private live-demo model assets.\n"
        "Extract the models directory at the repository root.\n"
        "Do not commit biometric data, MLflow directories, or these weights to public Git.\n",
        encoding="utf-8",
    )

    output = args.output.expanduser().resolve(strict=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        packaged_names = [*sources, "manifest.json", "README_MODELS.txt"]
        for name in sorted(packaged_names):
            path = stage_dir / name
            archive.write(path, Path("models/demo") / path.name)
    return stage_dir, output


def main() -> int:
    stage_dir, output = build_bundle(build_parser().parse_args())
    print(f"demo assets: {stage_dir}")
    print(f"portable zip: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
