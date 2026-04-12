"""
pipelines/dagster_project/ops/chunking_op.py

Op : Découpage des documents en chunks.

Input  : liste de dicts RawDocument
Output : liste de dicts TextChunk
"""

import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from dagster import op, In, Out, Output, OpExecutionContext
from pipelines.dagster_project.resources.config_resource import PipelineConfigResource
from indexing.chunking import chunk_text


def _dict_to_raw_document(d: dict):
    """Reconstruit un objet-like depuis un dict."""
    class _Doc:
        pass
    doc = _Doc()
    for k, v in d.items():
        setattr(doc, k, v)
    return doc


@op(
    ins={"raw_documents": In(list)},
    out={"chunks": Out(list)},
    description="Découpe les documents en chunks avec overlap.",
)
def chunking_op(
    context: OpExecutionContext,
    raw_documents: list,
    config: PipelineConfigResource,
) -> Output:
    """
    Chunking de chaque document.
    Réutilise indexing/chunking.py existant.
    """
    if not raw_documents:
        context.log.info("Aucun document à chunker")
        return Output([], output_name="chunks")

    start_time = time.time()
    all_chunks = []
    global_index = 0

    context.log.info(
        f"Chunking : {len(raw_documents)} documents",
        extra={"chunk_size": config.chunk_size, "overlap": config.overlap},
    )

    for doc_dict in raw_documents:
        doc_chunks = chunk_text(
            text=doc_dict["text"],
            source=doc_dict["source"],
            chunk_size=config.chunk_size,
            overlap=config.overlap,
            extra_metadata=doc_dict.get("metadata", {}),
        )

        for chunk in doc_chunks:
            chunk.chunk_index = global_index
            chunk.metadata["chunk_index"] = global_index
            chunk.metadata["total_chunks_approx"] = len(doc_chunks)

            all_chunks.append({
                "chunk_index": chunk.chunk_index,
                "text": chunk.text,
                "source": chunk.source,
                "page_number": chunk.page_number,
                "char_start": chunk.char_start,
                "char_end": chunk.char_end,
                "metadata": chunk.metadata,
            })
            global_index += 1

        context.log.info(
            f"  {doc_dict['source']} → {len(doc_chunks)} chunks",
        )

    duration_ms = (time.time() - start_time) * 1000

    total_chars = sum(len(c["text"]) for c in all_chunks)
    avg_len = total_chars // max(len(all_chunks), 1)

    context.log.info(
        f"Chunking terminé",
        extra={
            "total_chunks": len(all_chunks),
            "avg_chunk_len": avg_len,
            "duration_ms": round(duration_ms),
        },
    )

    return Output(all_chunks, output_name="chunks")