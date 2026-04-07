"""
backend/app/core/settings.py

Source unique de vérité pour toute la configuration de l'application.
Utilise pydantic-settings pour valider et typer les variables d'environnement.
Le chemin vers .env est résolu en absolu depuis la racine du projet.
"""

from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# Chemin absolu vers la racine du projet (rag-agent-system/)
# settings.py est dans backend/app/core/ → 3 niveaux au-dessus
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
ENV_FILE = ROOT_DIR / ".env"


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = "rag-agent-system"
    app_version: str = "0.1.0"
    app_env: str = "development"
    debug: bool = False

    # API Server
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # OpenRouter
    openrouter_api_key: str = "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"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # ChromaDB
    chroma_host: str = "localhost"
    chroma_port: int = 8001
    chroma_collection_name: str = "rag_documents"

    # Retrieval
    retrieval_top_k: int = 5
    retrieval_score_threshold: float = 0.0

    # Logging
    log_level: str = "INFO"


@lru_cache()
def get_settings() -> Settings:
    """
    Retourne une instance unique de Settings (singleton via cache).

    Returns:
        Instance Settings validée et prête à l'emploi
    """
    return Settings()