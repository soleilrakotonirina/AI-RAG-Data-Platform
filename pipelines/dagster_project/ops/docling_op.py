"""
pipelines/dagster_project/ops/docling_op.py

Op : Extraction de texte via Docling.

Input  : liste de chemins fichiers bruts
Output : liste de RawDocument (dicts sérialisables)
"""

import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from dagster import op, In, Out, Output, OpExecutionContext
from pipelines.dagster_project.resources.config_resource import PipelineConfigResource
from ingestion.docling_pipeline import process_single_file, load_all_processed


def _raw_document_to_dict(doc) -> dict:
    """Convertit un RawDocument en dict sérialisable."""
    return {
        "id": doc.id,
        "text": doc.text,
        "source": doc.source,
        "file_path": doc.file_path,
        "file_type": doc.file_type,
        "char_count": doc.char_count,
        "word_count": doc.word_count,
        "metadata": doc.metadata,
    }


@op(
    ins={"file_paths": In(list)},
    out={"raw_documents": Out(list)},
    description="Extrait le texte des documents via Docling (mode fast, sans OCR).",
)
def docling_op(
    context: OpExecutionContext,
    file_paths: list,
    config: PipelineConfigResource,
) -> Output:
    """
    Traite chaque fichier via Docling.

    Mode 'processed' : charge depuis data/processed/ sans Docling.
    Autres modes     : extraction Docling + sauvegarde data/processed/.
    """
    start_time = time.time()

    # Mode processed : charger directement depuis data/processed/
    if config.mode == "processed":
        context.log.info("Mode 'processed' : chargement depuis data/processed/")
        docs = load_all_processed()
        raw_documents = [_raw_document_to_dict(d) for d in docs]
        context.log.info(f"Documents chargés depuis processed/ : {len(raw_documents)}")
        return Output(raw_documents, output_name="raw_documents")

    # Aucun fichier à traiter
    if not file_paths:
        context.log.info("Aucun fichier à traiter")
        return Output([], output_name="raw_documents")

    raw_documents = []
    failed = []
    force = config.mode == "full"

    context.log.info(f"Extraction Docling : {len(file_paths)} fichiers")

    for file_path_str in file_paths:
        file_path = Path(file_path_str)
        context.log.info(f"Traitement : {file_path.name}")

        doc = process_single_file(file_path, force=force)

        if doc is None:
            failed.append(file_path.name)
            context.log.warning(f"Échec traitement : {file_path.name}")
            continue

        raw_documents.append(_raw_document_to_dict(doc))
        context.log.info(
            f"OK : {file_path.name}",
            extra={"chars": doc.char_count, "words": doc.word_count},
        )

    duration_ms = (time.time() - start_time) * 1000

    context.log.info(
        f"Docling terminé",
        extra={
            "total_files": len(file_paths),
            "documents_ok": len(raw_documents),
            "failed": len(failed),
            "duration_ms": round(duration_ms),
        },
    )

    if failed:
        context.log.warning(f"Fichiers échoués : {failed}")

    return Output(raw_documents, output_name="raw_documents")