"""
backend/app/services/embedding_service.py

Couche d'accès aux embeddings via OpenRouter.
Responsabilités :
- Générer un embedding pour un texte donné
- Générer des embeddings en lot (batch)
- Fallback local si OpenRouter indisponible
- Logger chaque appel

Modèle utilisé : text-embedding-3-small (OpenAI via OpenRouter)
Dimension de sortie : 1536 floats
"""

import hashlib
import math
import httpx

from backend.app.core.settings import get_settings
from backend.app.core.logger import get_logger

logger = get_logger(__name__)

# Dimension du modèle text-embedding-3-small
EMBEDDING_DIM = 1536

# Modèle d'embedding utilisé via OpenRouter
EMBEDDING_MODEL = "openai/text-embedding-3-small"


# ---------------------------------------------------------------------------
# Fallback local — embedding déterministe basé sur le hash du texte
# ---------------------------------------------------------------------------

def _fallback_embedding(text: str, dim: int = EMBEDDING_DIM) -> list[float]:
    """
    Génère un embedding déterministe à partir du hash SHA-256 du texte.

    Contrairement aux embeddings aléatoires de Phase 2, cet embedding
    est stable : le même texte produit toujours le même vecteur.
    Il ne capture pas la sémantique mais permet de tester la pipeline
    sans clé API.

    Args:
        text: Texte source
        dim: Dimension du vecteur de sortie

    Returns:
        Vecteur normalisé de dimension `dim`
    """
    seed = hashlib.sha256(text.encode("utf-8")).digest()

    # Génère dim floats depuis les bytes du hash (cycle si nécessaire)
    values = []
    for i in range(dim):
        byte_index = i % len(seed)
        # Normalise le byte en float entre -1 et 1
        values.append((seed[byte_index] / 127.5) - 1.0)

    # Normalisation L2
    norm = math.sqrt(sum(x ** 2 for x in values))
    if norm == 0:
        return values
    return [x / norm for x in values]


# ---------------------------------------------------------------------------
# Client OpenRouter — embedding
# ---------------------------------------------------------------------------

def embed_text(text: str) -> list[float]:
    """
    Génère un embedding pour un texte unique via OpenRouter.

    Si la clé API est absente ou si l'appel échoue,
    bascule automatiquement sur le fallback local.

    Args:
        text: Texte à transformer en vecteur

    Returns:
        Vecteur float de dimension EMBEDDING_DIM (1536)
    """
    settings = get_settings()

    if not settings.openrouter_api_key or settings.openrouter_api_key == "your_openrouter_api_key_here":
        logger.warning(
            "OpenRouter API key not configured — using fallback embedding",
            model=EMBEDDING_MODEL,
        )
        return _fallback_embedding(text)

    try:
        response = httpx.post(
            url=f"{settings.openrouter_base_url}/embeddings",
            headers={
                "Authorization": f"Bearer {settings.openrouter_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": EMBEDDING_MODEL,
                "input": text,
            },
            timeout=30.0,
        )

        response.raise_for_status()
        data = response.json()

        embedding = data["data"][0]["embedding"]

        logger.info(
            "Embedding generated via OpenRouter",
            model=EMBEDDING_MODEL,
            dim=len(embedding),
            text_preview=text[:50],
        )

        return embedding

    except httpx.HTTPStatusError as e:
        logger.error(
            "OpenRouter HTTP error — falling back to local embedding",
            status_code=e.response.status_code,
            error=str(e),
        )
        return _fallback_embedding(text)

    except httpx.RequestError as e:
        logger.error(
            "OpenRouter connection error — falling back to local embedding",
            error=str(e),
        )
        return _fallback_embedding(text)

    except (KeyError, IndexError) as e:
        logger.error(
            "Unexpected OpenRouter response format — falling back",
            error=str(e),
        )
        return _fallback_embedding(text)


def embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Génère des embeddings pour une liste de textes.

    Traite les textes individuellement pour simplifier la gestion
    d'erreur par document. En Phase 12 (Dagster), un vrai batching
    sera implémenté pour optimiser les appels API.

    Args:
        texts: Liste de textes à transformer

    Returns:
        Liste de vecteurs, dans le même ordre que les textes

    Raises:
        ValueError: Si la liste est vide
    """
    if not texts:
        raise ValueError("La liste de textes est vide.")

    logger.info("Starting batch embedding", total=len(texts))

    embeddings = []
    for i, text in enumerate(texts):
        embedding = embed_text(text)
        embeddings.append(embedding)
        logger.info(
            "Embedding progress",
            current=i + 1,
            total=len(texts),
        )

    logger.info("Batch embedding completed", total=len(embeddings))
    return embeddings