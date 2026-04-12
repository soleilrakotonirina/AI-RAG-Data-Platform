"""
pipelines/dagster_project/jobs/indexing_job.py

Job 2 : Indexation uniquement.
data/processed/ → chunking → enrichment → embeddings → ChromaDB

Utile pour relancer l'indexation sans re-extraire les PDFs.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from dagster import job
from pipelines.dagster_project.resources.config_resource import PipelineConfigResource
from pipelines.dagster_project.resources.chromadb_resource import ChromaDBResource
from pipelines.dagster_project.resources.openrouter_resource import OpenRouterResource
from pipelines.dagster_project.ops.docling_op import docling_op
from pipelines.dagster_project.ops.chunking_op import chunking_op
from pipelines.dagster_project.ops.enrichment_op import enrichment_op
from pipelines.dagster_project.ops.embedding_op import embedding_op
from pipelines.dagster_project.ops.chromadb_op import chromadb_op


@job(
    description="Indexation seule : data/processed/ → ChromaDB",
    resource_defs={
        "config": PipelineConfigResource.configure_at_launch(),
        "chromadb": ChromaDBResource.configure_at_launch(),
        "openrouter": OpenRouterResource.configure_at_launch(),
    },
)
def indexing_job():
    """
    Pipeline d'indexation depuis data/processed/.

    Étapes :
    1. Chargement data/processed/ (mode processed)
    2. Chunking
    3. Enrichissement
    4. Embeddings (cache si disponible)
    5. Indexation ChromaDB

    Note : passer mode='processed' dans la config.
    """
    raw_documents = docling_op(file_paths=[])
    chunks = chunking_op(raw_documents)
    enriched = enrichment_op(chunks)
    embedded = embedding_op(enriched)
    chromadb_op(embedded)