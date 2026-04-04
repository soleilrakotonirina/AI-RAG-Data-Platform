"""
backend/app/api/routes/health.py

Endpoint de santé de l'API.
Utilisé par les load balancers, orchestrateurs (K8s, Docker),
et comme premier test de validité du déploiement.
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from backend.app.api.deps import get_app_settings
from backend.app.core.settings import Settings
from backend.app.core.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


class HealthResponse(BaseModel):
    """Schéma de réponse du endpoint /health."""
    status: str
    app_name: str
    version: str
    environment: str


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Vérification de l'état de l'API",
    description="Retourne le statut de l'API et les métadonnées de l'application.",
    tags=["System"],
)
def health_check(
    settings: Settings = Depends(get_app_settings),
) -> HealthResponse:
    """
    Endpoint de santé.

    Returns:
        HealthResponse: Statut OK avec métadonnées de l'application
    """
    logger.info("health_check called", env=settings.app_env)

    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        version=settings.app_version,
        environment=settings.app_env,
    )