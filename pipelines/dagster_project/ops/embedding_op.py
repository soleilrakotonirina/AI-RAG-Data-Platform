"""
pipelines/dagster_project/ops/embedding_op.py

Op : Génération des embeddings avec cache disque.

Input  : liste de dicts EnrichedChunk
Output : liste de dicts EmbeddedChunk (avec vecteurs)
"""

import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from dagster import op, In, Out, Output, OpExecutionContext
from pipelines.dagster_project.resources.config_resource import PipelineConfigResource
from pipelines.dagster_project.resources.openrouter_resource import OpenRouterResource
from indexing.embeddings import (
    _load_disk_cache,
    _save_disk_cache,
    _get_text_hash,
)


@op(
    ins={"enriched_chunks": In(list)},
    out={"embedded_chunks": Out(list)},
    description="Génère les embeddings via OpenRouter avec cache disque.",
)
def embedding_op(
    context: OpExecutionContext,
    enriched_chunks: list,
    config: PipelineConfigResource,
    openrouter: OpenRouterResource,
) -> Output:
    """
    Génère les embeddings par batch.

    Optimisations :
    - Cache disque (data/embeddings/embeddings_cache.json)
    - Cache mémoire (embedding_service.py)
    - Traitement par batches
    """
    if not enriched_chunks:
        context.log.info("Aucun chunk à vectoriser")
        return Output([], output_name="embedded_chunks")

    start_time = time.time()
    batch_size = config.embedding_batch_size
    use_cache = config.use_disk_cache

    # Chargement du cache disque
    disk_cache = _load_disk_cache() if use_cache else {}
    cache_hits = 0

    # Séparation chunks cachés vs à calculer
    to_compute_indices = []
    to_compute_texts = []
    results = [None] * len(enriched_chunks)

    for i, chunk in enumerate(enriched_chunks):
        cache_key = _get_text_hash(chunk["text"])
        if use_cache and cache_key in disk_cache:
            results[i] = {
                **chunk,
                "embedding": disk_cache[cache_key],
            }
            cache_hits += 1
        else:
            to_compute_indices.append(i)
            to_compute_texts.append(chunk["text"])

    context.log.info(
        f"Embeddings : {len(enriched_chunks)} chunks",
        extra={
            "from_cache": cache_hits,
            "to_compute": len(to_compute_indices),
            "batch_size": batch_size,
        },
    )

    # Génération par batch pour les chunks non cachés
    api_calls = 0
    total_batches = (len(to_compute_texts) + batch_size - 1) // batch_size

    for batch_num in range(total_batches):
        batch_start = batch_num * batch_size
        batch_texts = to_compute_texts[batch_start:batch_start + batch_size]
        batch_indices = to_compute_indices[batch_start:batch_start + batch_size]

        context.log.info(f"Batch {batch_num + 1}/{total_batches} : {len(batch_texts)} chunks")

        vectors = openrouter.embed_texts(
            texts=batch_texts,
            use_cache=True,
        )
        api_calls += 1

        for idx, (original_idx, vector) in enumerate(zip(batch_indices, vectors)):
            chunk = enriched_chunks[original_idx]
            results[original_idx] = {
                **chunk,
                "embedding": vector,
            }

            # Mise à jour cache disque
            if use_cache:
                cache_key = _get_text_hash(chunk["text"])
                disk_cache[cache_key] = vector

    # Sauvegarde cache disque
    if use_cache and to_compute_texts:
        _save_disk_cache(disk_cache)
        context.log.info(
            f"Cache disque mis à jour : {len(disk_cache)} embeddings",
        )

    embedded = [r for r in results if r is not None]
    duration_ms = (time.time() - start_time) * 1000

    context.log.info(
        f"Embeddings terminés",
        extra={
            "total": len(embedded),
            "cache_hits": cache_hits,
            "api_calls": api_calls,
            "duration_ms": round(duration_ms),
        },
    )

    return Output(embedded, output_name="embedded_chunks")