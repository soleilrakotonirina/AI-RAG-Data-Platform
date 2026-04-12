"""
pipelines/dagster_project/ops/chromadb_op.py

Op : Indexation dans ChromaDB.

Input  : liste de dicts EmbeddedChunk
Output : rapport d'indexation (dict)
"""

import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from dagster import op, In, Out, Output, OpExecutionContext
from pipelines.dagster_project.resources.config_resource import PipelineConfigResource
from pipelines.dagster_project.resources.chromadb_resource import ChromaDBResource
from backend.app.db.vector_store import Document


@op(
    ins={"embedded_chunks": In(list)},
    out={"indexing_report": Out(dict)},
    description="Insère les chunks vectorisés dans ChromaDB par batches.",
)
def chromadb_op(
    context: OpExecutionContext,
    embedded_chunks: list,
    config: PipelineConfigResource,
    chromadb: ChromaDBResource,
) -> Output:
    """
    Indexation ChromaDB.

    Réutilise VectorStore (Phase 2).
    Reset si config.reset_chromadb = True.
    """
    start_time = time.time()
    batch_size = config.indexing_batch_size

    # Reset si demandé
    if config.reset_chromadb:
        chromadb.reset()
        context.log.warning("ChromaDB collection reset")

    count_before = chromadb.count()

    if not embedded_chunks:
        context.log.info("Aucun chunk à indexer")
        return Output(
            {"indexed": 0, "failed": 0, "total_in_db": count_before},
            output_name="indexing_report",
        )

    store = chromadb.get_store()
    indexed = 0
    failed = 0
    total_batches = (len(embedded_chunks) + batch_size - 1) // batch_size

    context.log.info(
        f"Indexation ChromaDB : {len(embedded_chunks)} chunks",
        extra={"batch_size": batch_size, "total_batches": total_batches},
    )

    for batch_num in range(total_batches):
        batch_start = batch_num * batch_size
        batch = embedded_chunks[batch_start:batch_start + batch_size]

        try:
            documents = [
                Document(
                    id=chunk["id"],
                    text=chunk["text"],
                    metadata=chunk["metadata"],
                    embedding=chunk["embedding"],
                )
                for chunk in batch
            ]
            store.add_documents(documents)
            indexed += len(documents)

            context.log.info(
                f"Batch {batch_num + 1}/{total_batches} indexé : {len(documents)} chunks",
            )

        except Exception as e:
            failed += len(batch)
            context.log.error(
                f"Batch {batch_num + 1} échoué : {e}",
                extra={"batch_start": batch_start, "error": str(e)},
            )

    count_after = chromadb.count()
    duration_ms = (time.time() - start_time) * 1000

    report = {
        "indexed": indexed,
        "failed": failed,
        "total_in_db": count_after,
        "added": count_after - count_before,
        "duration_ms": round(duration_ms),
    }

    context.log.info(
        f"Indexation terminée",
        extra=report,
    )

    return Output(report, output_name="indexing_report")