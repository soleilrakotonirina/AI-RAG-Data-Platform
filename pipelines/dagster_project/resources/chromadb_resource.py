"""
pipelines/dagster_project/resources/chromadb_resource.py

Ressource ChromaDB pour Dagster.
Wraps le VectorStore existant (Phase 2).
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from dagster import ConfigurableResource, InitResourceContext
from backend.app.db.vector_store import VectorStore


class ChromaDBResource(ConfigurableResource):
    """
    Ressource Dagster wrappant VectorStore.
    Réutilise la connexion existante (singleton).
    """

    collection_name: str = "rag_documents"

    def get_store(self) -> VectorStore:
        """Retourne une instance VectorStore."""
        return VectorStore()

    def reset(self) -> None:
        """Vide la collection ChromaDB."""
        store = self.get_store()
        store.reset()

    def count(self) -> int:
        """Nombre de documents dans la collection."""
        return self.get_store().count()