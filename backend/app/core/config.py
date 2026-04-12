"""
backend/app/core/config.py

Constantes globales de l'application.
Ce fichier centralise les valeurs fixes qui ne dépendent pas de l'environnement.
Les valeurs dynamiques (clés API, hosts) sont dans settings.py.
"""

# Préfixe global de toutes les routes API
API_PREFIX = "/api/v1"

# Métadonnées OpenAPI
API_TITLE = "RAG Agent System API"
API_DESCRIPTION = (
    "Backend API pour le système RAG + Agents. "
    "Expose les endpoints de chat, retrieval et gestion de documents."
)
API_VERSION = "0.1.0"

# Endpoints internes
HEALTH_ENDPOINT = "/health"
CHAT_ENDPOINT = "/chat"

# Retrieval
DEFAULT_TOP_K = 5
DEFAULT_SCORE_THRESHOLD = 0.7

# LLM
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MODEL = "mistralai/mistral-7b-instruct"

EMBEDDING_MODEL: str = "nvidia/llama-nemotron-embed-vl-1b-v2:free"