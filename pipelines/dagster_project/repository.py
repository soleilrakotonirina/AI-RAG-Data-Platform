"""
pipelines/dagster_project/repository.py

Point d'entrée Dagster — enregistre tous les jobs et schedules.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from dagster import Definitions

from pipelines.dagster_project.jobs.ingestion_job import ingestion_job
from pipelines.dagster_project.jobs.indexing_job import indexing_job
from pipelines.dagster_project.jobs.full_pipeline_job import full_pipeline_job
from pipelines.dagster_project.schedules.daily_schedule import (
    daily_schedule,
    incremental_schedule,
)
from pipelines.dagster_project.resources.config_resource import PipelineConfigResource
from pipelines.dagster_project.resources.chromadb_resource import ChromaDBResource
from pipelines.dagster_project.resources.openrouter_resource import OpenRouterResource


defs = Definitions(
    jobs=[
        ingestion_job,
        indexing_job,
        full_pipeline_job,
    ],
    schedules=[
        daily_schedule,
        incremental_schedule,
    ],
    resources={
        "config": PipelineConfigResource(
            mode="incremental",
            reset_chromadb=False,
            chunk_size=600,
            overlap=100,
            embedding_batch_size=50,
            use_disk_cache=True,
        ),
        "chromadb": ChromaDBResource(
            collection_name="rag_documents",
        ),
        "openrouter": OpenRouterResource(
            model="nvidia/llama-nemotron-embed-vl-1b-v2:free",
        ),
    },
)