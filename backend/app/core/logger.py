"""
backend/app/core/logger.py

Configuration du système de logging.
Utilise structlog pour des logs structurés (JSON en production, lisibles en dev).
Un logger unique est exporté et réutilisé dans tout le projet.
"""

import logging
import sys
import structlog
from backend.app.core.settings import get_settings


def configure_logging() -> None:
    """
    Configure le système de logging global.
    - En développement : logs colorés et lisibles
    - En production : logs JSON structurés

    À appeler UNE SEULE FOIS au démarrage de l'application.
    """
    settings = get_settings()
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    # Configuration du renderer selon l'environnement
    if settings.app_env == "development":
        renderer = structlog.dev.ConsoleRenderer(colors=True)
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )

    # Synchronise le logger stdlib (utilisé par uvicorn, FastAPI)
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )


def get_logger(name: str = __name__) -> structlog.BoundLogger:
    """
    Retourne un logger structlog lié au nom du module appelant.

    Args:
        name: Nom du module (utiliser __name__ par convention)

    Returns:
        Instance de BoundLogger prête à l'emploi
    """
    return structlog.get_logger(name)