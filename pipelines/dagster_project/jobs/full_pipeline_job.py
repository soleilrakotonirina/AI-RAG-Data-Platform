"""
pipelines/dagster_project/jobs/full_pipeline_job.py

Job 3 : Pipeline complet.
data/raw/ → Docling → chunking → enrichment → embeddings → ChromaDB

C'est le job principal — équivalent à :
python scripts/ingest_data.py --reset
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from dagster import job
from pipelines.dagster_project.resources.config_resource import PipelineConfigResource
from pipelines.dagster_project.resources.chromadb_resource import ChromaDBResource
from pipelines.dagster_project.resources.openrouter_resource import OpenRouterResource
from pipelines.dagster_project.ops.load_raw_docs_op import load_raw_docs_op
from pipelines.dagster_project.ops.docling_op import docling_op
from pipelines.dagster_project.ops.chunking_op import chunking_op
from pipelines.dagster_project.ops.enrichment_op import enrichment_op
from pipelines.dagster_project.ops.embedding_op import embedding_op
from pipelines.dagster_project.ops.chromadb_op import chromadb_op


@job(
    description="Pipeline complet : data/raw/ → ChromaDB (tous les modes)",
    resource_defs={
        "config": PipelineConfigResource.configure_at_launch(),
        "chromadb": ChromaDBResource.configure_at_launch(),
        "openrouter": OpenRouterResource.configure_at_launch(),
    },
)
def full_pipeline_job():
    """
    Pipeline complet RAG.

    Modes configurables via config :
    - full        : tout retraiter (--reset --force)
    - processed   : depuis data/processed/ (--from-processed)
    - incremental : seulement nouveaux fichiers (défaut)

    Config reset_chromadb=True pour vider ChromaDB avant.
    """
    file_paths = load_raw_docs_op()
    raw_documents = docling_op(file_paths)
    chunks = chunking_op(raw_documents)
    enriched = enrichment_op(chunks)
    embedded = embedding_op(enriched)
    chromadb_op(embedded)