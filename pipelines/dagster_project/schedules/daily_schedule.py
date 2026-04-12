"""
pipelines/dagster_project/schedules/daily_schedule.py

Schedules automatiques :
- daily_schedule     : pipeline complet chaque jour à minuit
- incremental_schedule : seulement nouveaux fichiers chaque heure
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from dagster import ScheduleDefinition
from pipelines.dagster_project.jobs.full_pipeline_job import full_pipeline_job


# Schedule quotidien — pipeline complet avec reset
daily_schedule = ScheduleDefinition(
    name="daily_full_pipeline",
    cron_schedule="0 0 * * *",           # Tous les jours à minuit
    job=full_pipeline_job,
    run_config={
        "resources": {
            "config": {
                "config": {
                    "mode": "full",
                    "reset_chromadb": True,
                    "chunk_size": 600,
                    "overlap": 100,
                    "embedding_batch_size": 50,
                    "use_disk_cache": True,
                }
            },
            "chromadb": {
                "config": {
                    "collection_name": "rag_documents",
                }
            },
            "openrouter": {
                "config": {
                    "model": "nvidia/llama-nemotron-embed-vl-1b-v2:free",
                }
            },
        }
    },
    description="Pipeline complet chaque jour à minuit — reset + reindexation",
)


# Schedule horaire — incrémental (seulement nouveaux fichiers)
incremental_schedule = ScheduleDefinition(
    name="hourly_incremental_pipeline",
    cron_schedule="0 * * * *",           # Toutes les heures
    job=full_pipeline_job,
    run_config={
        "resources": {
            "config": {
                "config": {
                    "mode": "incremental",
                    "reset_chromadb": False,
                    "chunk_size": 600,
                    "overlap": 100,
                    "embedding_batch_size": 50,
                    "use_disk_cache": True,
                }
            },
            "chromadb": {
                "config": {
                    "collection_name": "rag_documents",
                }
            },
            "openrouter": {
                "config": {
                    "model": "nvidia/llama-nemotron-embed-vl-1b-v2:free",
                }
            },
        }
    },
    description="Pipeline incrémental chaque heure — seulement nouveaux fichiers",
)