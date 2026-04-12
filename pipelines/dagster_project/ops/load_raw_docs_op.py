"""
pipelines/dagster_project/ops/load_raw_docs_op.py

Op : Scan de data/raw/ et détection des fichiers à traiter.

Sortie : liste de chemins de fichiers à ingérer.
Logique : selon le mode (full/incremental), décide quels fichiers traiter.
"""

import sys
import json
import hashlib
from pathlib import Path
from datetime import datetime

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from dagster import op, Out, Output, OpExecutionContext
from pipelines.dagster_project.resources.config_resource import PipelineConfigResource

SUPPORTED_FORMATS = {".pdf", ".docx", ".html", ".htm", ".txt", ".md"}


def _get_file_hash(file_path: Path) -> str:
    """Hash SHA256 court d'un fichier."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()[:16]


def _is_already_processed(file_path: Path, processed_dir: Path) -> bool:
    """Vérifie si un fichier a déjà été traité (via fichier .meta)."""
    clean_name = "".join(
        c if c.isalnum() or c in "_-" else "_"
        for c in file_path.stem
    )[:60]
    meta_path = processed_dir / f"{clean_name}.meta"
    md_path = processed_dir / f"{clean_name}.md"

    if not meta_path.exists() or not md_path.exists():
        return False

    try:
        stored_hash = meta_path.read_text(encoding="utf-8").strip()
        current_hash = _get_file_hash(file_path)
        return stored_hash == current_hash
    except Exception:
        return False


@op(
    out={"file_paths": Out(list)},
    description="Scan data/raw/ et retourne les fichiers à traiter selon le mode.",
)
def load_raw_docs_op(
    context: OpExecutionContext,
    config: PipelineConfigResource,
) -> Output:
    """
    Scanne data/raw/ et détermine quels fichiers traiter.

    Modes :
    - full/force    : tous les fichiers
    - incremental   : seulement les nouveaux/modifiés
    - processed     : aucun (skip cette op)
    """
    data_raw = ROOT_DIR / config.data_raw_path
    data_processed = ROOT_DIR / config.data_processed_path

    if not data_raw.exists():
        context.log.error(f"Répertoire data/raw/ introuvable : {data_raw}")
        return Output([], output_name="file_paths")

    all_files = [
        f for f in data_raw.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_FORMATS
    ]

    if not all_files:
        context.log.warning("Aucun fichier supporté dans data/raw/")
        return Output([], output_name="file_paths")

    # Mode from_processed : skip tout
    if config.mode == "processed":
        context.log.info("Mode 'processed' : skip scan data/raw/")
        return Output([], output_name="file_paths")

    # Mode full/force : tous les fichiers
    if config.mode == "full":
        file_paths = [str(f) for f in all_files]
        context.log.info(
            f"Mode 'full' : {len(file_paths)} fichiers à traiter",
            extra={"files": [f.name for f in all_files]},
        )
        return Output(file_paths, output_name="file_paths")

    # Mode incremental : seulement les nouveaux/modifiés
    to_process = []
    skipped = []

    for f in all_files:
        if _is_already_processed(f, data_processed):
            skipped.append(f.name)
        else:
            to_process.append(str(f))

    context.log.info(
        f"Mode 'incremental' : {len(to_process)} nouveaux, {len(skipped)} déjà traités",
        extra={
            "to_process": [Path(p).name for p in to_process],
            "skipped": skipped,
        },
    )

    return Output(to_process, output_name="file_paths")