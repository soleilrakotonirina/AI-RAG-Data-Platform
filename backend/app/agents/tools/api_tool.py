"""
backend/app/agents/tools/api_tool.py

Tool d'appel API interne.
Accède aux endpoints du backend FastAPI pour récupérer :
- statut du système
- métriques ChromaDB
- informations pipeline

Phase 12+ : ce tool sera enrichi avec des données
issues des pipelines Dagster (jobs status, dernière ingestion, etc.)
"""

import time
from datetime import datetime

import httpx

from backend.app.core.logger import get_logger

logger = get_logger(__name__)

# URL du backend FastAPI (configurable)
BACKEND_BASE_URL = "http://localhost:8000/api/v1"
API_TIMEOUT = 10.0


def api_tool(action: str, params: dict = None) -> dict:
    """
    Appelle l'API interne pour récupérer des données système.

    Actions disponibles :
    - "pipeline_status"  : état ChromaDB + pipeline RAG
    - "system_info"      : informations système général
    - "collection_stats" : statistiques de la collection vectorielle

    Args:
        action: Action à exécuter
        params: Paramètres additionnels (optionnel)

    Returns:
        Dict structuré avec résultats et métadonnées
    """
    params = params or {}
    start_time = time.time()

    logger.info(
        "API tool called",
        action=action,
        params=params,
    )

    try:
        result = _dispatch_action(action, params)
        duration_ms = (time.time() - start_time) * 1000

        output = {
            "action": action,
            "result": result,
            "success": True,
            "error": None,
            "duration_ms": round(duration_ms),
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(
            "API tool completed",
            action=action,
            duration_ms=round(duration_ms),
        )

        return output

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        logger.error("API tool failed", action=action, error=str(e))
        return {
            "action": action,
            "result": None,
            "success": False,
            "error": str(e),
            "duration_ms": round(duration_ms),
            "timestamp": datetime.now().isoformat(),
        }


def _dispatch_action(action: str, params: dict) -> dict:
    """
    Dispatche l'action vers la bonne fonction.

    Args:
        action: Nom de l'action
        params: Paramètres

    Returns:
        Résultat de l'action
    """
    if action == "pipeline_status":
        return _get_pipeline_status()
    elif action == "system_info":
        return _get_system_info()
    elif action == "collection_stats":
        return _get_collection_stats()
    else:
        raise ValueError(f"Action inconnue : '{action}'. "
                         f"Actions valides : pipeline_status, system_info, collection_stats")


def _get_pipeline_status() -> dict:
    """Récupère le statut du pipeline RAG via l'API."""
    try:
        response = httpx.get(
            f"{BACKEND_BASE_URL}/chat/status",
            timeout=API_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()
    except httpx.ConnectError:
        # Fallback local si API non lancée
        return _get_collection_stats_local()


def _get_system_info() -> dict:
    """Informations générales du système."""
    import sys
    import platform
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.system(),
        "backend_url": BACKEND_BASE_URL,
        "timestamp": datetime.now().isoformat(),
    }


def _get_collection_stats() -> dict:
    """Statistiques ChromaDB directes (sans passer par l'API HTTP)."""
    return _get_collection_stats_local()


def _get_collection_stats_local() -> dict:
    """Accès direct à ChromaDB (fallback)."""
    from backend.app.db.vector_store import VectorStore
    store = VectorStore()
    count = store.count()
    return {
        "collection_name": "rag_documents",
        "document_count": count,
        "status": "connected",
        "source": "direct_chromadb",
    }