"""
pipelines/dagster_project/ops/enrichment_op.py

Op : Enrichissement des métadonnées des chunks.

Input  : liste de dicts TextChunk
Output : liste de dicts EnrichedChunk
"""

import sys
import time
from pathlib import Path
from datetime import datetime

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from dagster import op, In, Out, Output, OpExecutionContext
from pipelines.dagster_project.resources.config_resource import PipelineConfigResource
from indexing.enrichment import (
    detect_language,
    detect_domain,
    detect_countries,
    generate_chunk_id,
)


@op(
    ins={"chunks": In(list)},
    out={"enriched_chunks": Out(list)},
    description="Enrichit les chunks avec langue, domaine, pays, timestamp.",
)
def enrichment_op(
    context: OpExecutionContext,
    chunks: list,
    config: PipelineConfigResource,
) -> Output:
    """
    Enrichissement des métadonnées.
    Réutilise indexing/enrichment.py existant.

    ChromaDB n'accepte que str/int/float/bool :
    - countries → str (jointure virgule)
    - tous les autres champs → scalaires
    """
    if not chunks:
        context.log.info("Aucun chunk à enrichir")
        return Output([], output_name="enriched_chunks")

    start_time = time.time()
    domain_override = config.domain_override or None

    context.log.info(
        f"Enrichissement : {len(chunks)} chunks",
        extra={"domain_override": domain_override or "auto"},
    )

    enriched = []
    langs = {}
    domains = {}

    for chunk_dict in chunks:
        text = chunk_dict["text"]
        source = chunk_dict["source"]
        chunk_index = chunk_dict["chunk_index"]

        language = detect_language(text)
        domain = domain_override or detect_domain(text)
        countries = detect_countries(text)
        chunk_id = generate_chunk_id(source, chunk_index)

        # ChromaDB : uniquement str/int/float/bool
        metadata = {
            **chunk_dict.get("metadata", {}),
            "language": language,
            "domain": domain,
            "countries": ",".join(countries),   # list → str
            "word_count": len(text.split()),
            "char_count": len(text),
            "indexed_at": datetime.now().isoformat(),
        }

        enriched.append({
            "id": chunk_id,
            "text": text,
            "metadata": metadata,
        })

        langs[language] = langs.get(language, 0) + 1
        domains[domain] = domains.get(domain, 0) + 1

    duration_ms = (time.time() - start_time) * 1000

    context.log.info(
        f"Enrichissement terminé",
        extra={
            "total": len(enriched),
            "languages": langs,
            "domains": domains,
            "duration_ms": round(duration_ms),
        },
    )

    return Output(enriched, output_name="enriched_chunks")