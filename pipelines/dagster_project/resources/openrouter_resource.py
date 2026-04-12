"""
pipelines/dagster_project/resources/openrouter_resource.py

Ressource OpenRouter pour Dagster.
Wraps l'embedding_service existant (Phase 3).
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from dagster import ConfigurableResource
from backend.app.services.embedding_service import embed_batch, get_cache_stats


class OpenRouterResource(ConfigurableResource):
    """
    Ressource Dagster pour OpenRouter.
    Délègue à embedding_service.py existant.
    """

    model: str = "nvidia/llama-nemotron-embed-vl-1b-v2:free"

    def embed_texts(
        self,
        texts: list[str],
        use_cache: bool = True,
        show_progress: bool = False,
    ) -> list[list[float]]:
        """Génère les embeddings pour une liste de textes."""
        return embed_batch(
            texts=texts,
            use_cache=use_cache,
            show_progress=show_progress,
        )

    def get_cache_stats(self) -> dict:
        """Retourne les stats du cache embeddings."""
        return get_cache_stats()