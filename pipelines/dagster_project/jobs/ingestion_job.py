"""
pipelines/dagster_project/jobs/ingestion_job.py

Job 1 : Ingestion uniquement.
raw → Docling → data/processed/

Ne touche pas ChromaDB.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from dagster import job, Config
from pipelines.dagster_project.resources.config_resource import PipelineConfigResource
from pipelines.dagster_project.ops.load_raw_docs_op import load_raw_docs_op
from pipelines.dagster_project.ops.docling_op import docling_op


@job(
    description="Ingestion seule : data/raw/ → Docling → data/processed/",
    resource_defs={
        "config": PipelineConfigResource.configure_at_launch(),
    },
)
def ingestion_job():
    """
    Pipeline d'ingestion uniquement.

    Étapes :
    1. Scan data/raw/
    2. Extraction Docling
    3. Sauvegarde data/processed/

    Config par défaut : mode incremental (seulement nouveaux fichiers).
    """
    file_paths = load_raw_docs_op()
    docling_op(file_paths)