"""
indexing/index_builder.py

Indexation finale dans ChromaDB.

Réutilise VectorStore (Phase 2) — ne réimplémente pas.

Fonctionnalités :
- Insertion par batch (évite surcharge mémoire)
- Rapport d'indexation complet
- Compatible avec reset ou ajout incrémental

Préparation Dagster : fonction pure, transformable en op.
"""

import sys
import time
from pathlib import Path
from dataclasses import dataclass

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from backend.app.db.vector_store import VectorStore, Document
from backend.app.core.logger import get_logger
from indexing.embeddings import EmbeddedChunk

logger = get_logger(__name__)

DEFAULT_BATCH_SIZE = 50


@dataclass
class IndexingReport:
    """Rapport d'indexation ChromaDB."""
    total_chunks: int
    indexed: int
    failed: int
    duration_ms: float
    final_db_count: int


def build_index(
    embedded_chunks: list[EmbeddedChunk],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> IndexingReport:
    """
    Insère les chunks avec embeddings dans ChromaDB.

    Args:
        embedded_chunks: Chunks vectorisés
        batch_size: Taille des batches d'insertion

    Returns:
        IndexingReport avec métriques
    """
    if not embedded_chunks:
        logger.warning("No chunks to index")
        store = VectorStore()
        return IndexingReport(
            total_chunks=0, indexed=0, failed=0,
            duration_ms=0.0, final_db_count=store.count(),
        )

    start_time = time.time()
    store = VectorStore()
    indexed = 0
    failed = 0

    logger.info(
        "Starting ChromaDB indexation",
        total=len(embedded_chunks),
        batch_size=batch_size,
    )

    total_batches = (len(embedded_chunks) + batch_size - 1) // batch_size

    for batch_start in range(0, len(embedded_chunks), batch_size):
        batch = embedded_chunks[batch_start:batch_start + batch_size]
        batch_num = batch_start // batch_size + 1

        try:
            documents = [
                Document(
                    id=chunk.id,
                    text=chunk.text,
                    metadata=chunk.metadata,
                    embedding=chunk.embedding,
                )
                for chunk in batch
            ]

            store.add_documents(documents)
            indexed += len(documents)

            logger.info(
                f"Batch {batch_num}/{total_batches} indexed",
                count=len(documents),
                total_indexed=indexed,
            )

        except Exception as e:
            logger.error(
                f"Batch {batch_num} failed",
                error=str(e),
                batch_start=batch_start,
            )
            failed += len(batch)

    duration_ms = (time.time() - start_time) * 1000
    final_count = store.count()

    report = IndexingReport(
        total_chunks=len(embedded_chunks),
        indexed=indexed,
        failed=failed,
        duration_ms=round(duration_ms),
        final_db_count=final_count,
    )

    logger.info(
        "ChromaDB indexation completed",
        indexed=indexed,
        failed=failed,
        final_db_count=final_count,
        duration_ms=round(duration_ms),
    )

    return report