"""
backend/app/api/main.py

Point d'entrée de l'application FastAPI.
Configure l'application, les middlewares, et enregistre les routers.
Ce fichier est importé par run.py et uvicorn.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.core.config import (
    API_PREFIX,
    API_TITLE,
    API_DESCRIPTION,
    API_VERSION,
)
from backend.app.core.logger import configure_logging, get_logger
from backend.app.core.settings import get_settings
from backend.app.api.routes import health, chat

# Initialisation du logging AVANT tout le reste
configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestionnaire de cycle de vie de l'application.
    - startup : initialisation des ressources
    - shutdown : nettoyage propre

    Utilise le pattern asynccontextmanager recommandé par FastAPI (>= 0.93).
    """
    settings = get_settings()
    logger.info(
        "Application starting",
        app_name=settings.app_name,
        version=settings.app_version,
        environment=settings.app_env,
    )
    yield
    logger.info("Application shutting down")


def create_app() -> FastAPI:
    """
    Factory function : crée et configure l'instance FastAPI.
    Pattern recommandé pour la testabilité et la modularité.

    Returns:
        Instance FastAPI configurée
    """
    settings = get_settings()

    app = FastAPI(
        title=API_TITLE,
        description=API_DESCRIPTION,
        version=API_VERSION,
        debug=settings.debug,
        lifespan=lifespan,
    )

    # CORS — à restreindre en production
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Enregistrement des routers
    app.include_router(health.router, prefix=API_PREFIX)
    app.include_router(chat.router, prefix=API_PREFIX)

    logger.info(
        "FastAPI application created",
        prefix=API_PREFIX,
        routes=["/health", "/chat", "/chat/status"],
    )
    return app


# Instance exportée — importée par uvicorn et run.py
app = create_app()