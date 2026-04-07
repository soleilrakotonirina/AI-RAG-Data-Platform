"""
backend/app/core/settings.py

Source unique de vérité pour toute la configuration de l'application.
Utilise pydantic-settings pour valider et typer les variables d'environnement.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=".env",
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