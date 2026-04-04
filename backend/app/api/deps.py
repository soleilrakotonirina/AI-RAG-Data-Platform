"""
backend/app/api/deps.py

Dépendances FastAPI injectées dans les routes via Depends().
Ce fichier centralisera l'accès aux services (DB, LLM, Retriever, etc.)
au fur et à mesure des phases.

Phase actuelle : fondation uniquement (settings + logger).
"""

from backend.app.core.settings import Settings, get_settings
from backend.app.core.logger import get_logger

logger = get_logger(__name__)


def get_app_settings() -> Settings:
    """
    Dépendance FastAPI : retourne la configuration de l'application.

    Usage dans une route :
        @router.get("/example")
        def example(settings: Settings = Depends(get_app_settings)):
            ...
    """
    return get_settings()