"""
backend/app/services/embedding_service.py

Couche d'accès aux embeddings via OpenRouter.
Améliorations Phase 6 :
- Cache LRU des embeddings (évite appels API redondants)
- Batch optimisé avec retry automatique
- Normalisation L2 systématique
- Métriques de cache hit/miss
"""

import hashlib
import math
import time
import httpx
from functools import lru_cache

from backend.app.core.settings import get_settings
from backend.app.core.logger import get_logger

logger = get_logger(__name__)

EMBEDDING_DIM = 1536
EMBEDDING_MODEL = "openai/text-embedding-3-small"
MAX_RETRIES = 3
RETRY_DELAY = 1.0


# ---------------------------------------------------------------------------
# Cache embeddings — évite les appels API redondants
# ---------------------------------------------------------------------------

_embedding_cache: dict[str, list[float]] = {}
_cache_hits = 0
_cache_misses = 0


def _get_cache_key(text: str) -> str:
    """Génère une clé de cache SHA256 pour un texte."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_cache_stats() -> dict:
    """Retourne les statistiques du cache embeddings."""
    total = _cache_hits + _cache_misses
    hit_rate = (_cache_hits / total * 100) if total > 0 else 0
    return {
        "hits": _cache_hits,
        "misses": _cache_misses,
        "total": total,
        "hit_rate_pct": round(hit_rate, 1),
        "cached_embeddings": len(_embedding_cache),
    }


def clear_embedding_cache() -> None:
    """Vide le cache des embeddings."""
    global _cache_hits, _cache_misses
    _embedding_cache.clear()
    _cache_hits = 0
    _cache_misses = 0
    logger.info("Embedding cache cleared")


# ---------------------------------------------------------------------------
# Normalisation L2
# ---------------------------------------------------------------------------

def _normalize_l2(vector: list[float]) -> list[float]:
    """
    Normalise un vecteur en L2.
    Garantit que tous les embeddings ont une norme unitaire,
    ce qui améliore la cohérence de la similarité cosine.
    """
    norm = math.sqrt(sum(x ** 2 for x in vector))
    if norm == 0:
        return vector
    return [x / norm for x in vector]


# ---------------------------------------------------------------------------
# Fallback déterministe
# ---------------------------------------------------------------------------

def _fallback_embedding(text: str, dim: int = EMBEDDING_DIM) -> list[float]:
    """
    Embedding déterministe basé sur SHA256.
    Stable : même texte → même vecteur.
    Utilisé si OpenRouter est indisponible.
    """
    seed = hashlib.sha256(text.encode("utf-8")).digest()
    values = []
    for i in range(dim):
        byte_val = seed[i % len(seed)]
        values.append((byte_val / 127.5) - 1.0)
    return _normalize_l2(values)


# ---------------------------------------------------------------------------
# Client OpenRouter — embedding avec retry et cache
# ---------------------------------------------------------------------------

def embed_text(text: str, use_cache: bool = True) -> list[float]:
    """
    Génère un embedding pour un texte avec cache et retry.

    Args:
        text: Texte à transformer en vecteur
        use_cache: Utiliser le cache LRU (défaut: True)

    Returns:
        Vecteur float normalisé de dimension EMBEDDING_DIM
    """
    global _cache_hits, _cache_misses

    if not text or not text.strip():
        logger.warning("Empty text passed to embed_text — returning zero vector")
        return [0.0] * EMBEDDING_DIM

    # Vérification cache
    cache_key = _get_cache_key(text)
    if use_cache and cache_key in _embedding_cache:
        _cache_hits += 1
        logger.info(
            "Embedding cache hit",
            text_preview=text[:40],
            cache_stats=get_cache_stats(),
        )
        return _embedding_cache[cache_key]

    _cache_misses += 1

    settings = get_settings()

    if not settings.openrouter_api_key or \
       settings.openrouter_api_key == "your_openrouter_api_key_here":
        logger.warning("OpenRouter API key not configured — using fallback")
        return _fallback_embedding(text)

    # Retry avec backoff exponentiel
    last_error = None
    for attempt in range(MAX_RETRIES):
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
            normalized = _normalize_l2(embedding)

            # Mise en cache
            if use_cache:
                _embedding_cache[cache_key] = normalized

            logger.info(
                "Embedding generated via OpenRouter",
                model=EMBEDDING_MODEL,
                dim=len(normalized),
                text_preview=text[:50],
                attempt=attempt + 1,
            )
            return normalized

        except httpx.HTTPStatusError as e:
            last_error = e
            if e.response.status_code == 429:
                wait = RETRY_DELAY * (2 ** attempt)
                logger.warning(
                    "Embedding rate limit — retrying",
                    attempt=attempt + 1,
                    wait_seconds=wait,
                )
                time.sleep(wait)
                continue
            logger.error("Embedding HTTP error", status=e.response.status_code)
            break

        except (httpx.RequestError, KeyError, IndexError) as e:
            last_error = e
            logger.error("Embedding error", error=str(e), attempt=attempt + 1)
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)

    logger.error(
        "All embedding attempts failed — using fallback",
        error=str(last_error),
    )
    return _fallback_embedding(text)


def embed_batch(
    texts: list[str],
    use_cache: bool = True,
    show_progress: bool = True,
) -> list[list[float]]:
    """
    Génère des embeddings pour une liste de textes.
    Optimisé : saute les textes déjà en cache.

    Args:
        texts: Liste de textes
        use_cache: Utiliser le cache
        show_progress: Logger la progression

    Returns:
        Liste de vecteurs dans le même ordre
    """
    if not texts:
        raise ValueError("La liste de textes est vide.")

    # Identifier les textes non cachés
    uncached_indices = []
    results = [None] * len(texts)

    for i, text in enumerate(texts):
        cache_key = _get_cache_key(text)
        if use_cache and cache_key in _embedding_cache:
            results[i] = _embedding_cache[cache_key]
        else:
            uncached_indices.append(i)

    cached_count = len(texts) - len(uncached_indices)
    logger.info(
        "Batch embedding started",
        total=len(texts),
        cached=cached_count,
        to_compute=len(uncached_indices),
    )

    # Générer les embeddings manquants
    for j, i in enumerate(uncached_indices):
        results[i] = embed_text(texts[i], use_cache=use_cache)
        if show_progress:
            logger.info(
                "Embedding progress",
                current=j + 1,
                total=len(uncached_indices),
                text_preview=texts[i][:40],
            )

    logger.info(
        "Batch embedding completed",
        total=len(results),
        cache_stats=get_cache_stats(),
    )
    return results