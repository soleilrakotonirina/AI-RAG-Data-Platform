"""
indexing/embeddings.py

Génération et cache des embeddings pour les chunks.

Réutilise embedding_service.py (Phase 3) — ne réimplémente pas.

Cache local :
- Sauvegarde dans data/embeddings/ (JSON par batch)
- Évite les appels API en cas de relance
- Clé de cache : hash SHA256 du texte

Préparation Dagster : fonction pure, transformable en op.
"""

import sys
import json
import hashlib
import time
from pathlib import Path
from dataclasses import dataclass, field

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from backend.app.services.embedding_service import embed_batch, get_cache_stats
from backend.app.core.logger import get_logger
from indexing.enrichment import EnrichedChunk

logger = get_logger(__name__)

DATA_EMBEDDINGS = ROOT_DIR / "data" / "embeddings"
EMBEDDINGS_CACHE_FILE = DATA_EMBEDDINGS / "embeddings_cache.json"


@dataclass
class EmbeddedChunk:
    """Chunk avec son embedding — prêt pour ChromaDB."""
    id: str
    text: str
    embedding: list[float]
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Cache persistant sur disque
# ---------------------------------------------------------------------------

def _load_disk_cache() -> dict:
    """Charge le cache d'embeddings depuis le disque."""
    DATA_EMBEDDINGS.mkdir(parents=True, exist_ok=True)
    if not EMBEDDINGS_CACHE_FILE.exists():
        return {}
    try:
        with open(EMBEDDINGS_CACHE_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Failed to load embeddings cache", error=str(e))
        return {}


def _save_disk_cache(cache: dict) -> None:
    """Sauvegarde le cache d'embeddings sur le disque."""
    DATA_EMBEDDINGS.mkdir(parents=True, exist_ok=True)
    try:
        with open(EMBEDDINGS_CACHE_FILE, "w") as f:
            json.dump(cache, f)
    except Exception as e:
        logger.warning("Failed to save embeddings cache", error=str(e))


def _get_text_hash(text: str) -> str:
    """Génère une clé de cache pour un texte."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]


# ---------------------------------------------------------------------------
# Génération embeddings
# ---------------------------------------------------------------------------

def generate_embeddings(
    enriched_chunks: list[EnrichedChunk],
    use_disk_cache: bool = True,
    batch_size: int = 20,
) -> list[EmbeddedChunk]:
    """
    Génère les embeddings pour une liste de chunks enrichis.

    Optimisations :
    - Cache en mémoire (embedding_service.py)
    - Cache sur disque (data/embeddings/)
    - Traitement par batch

    Args:
        enriched_chunks: Chunks à vectoriser
        use_disk_cache: Utiliser le cache disque
        batch_size: Taille des batches

    Returns:
        Liste de EmbeddedChunk avec vecteurs
    """
    if not enriched_chunks:
        return []

    start_time = time.time()
    disk_cache = _load_disk_cache() if use_disk_cache else {}
    embedded = []
    to_compute = []
    to_compute_indices = []

    # Identification des chunks non cachés
    for i, chunk in enumerate(enriched_chunks):
        cache_key = _get_text_hash(chunk.text)
        if use_disk_cache and cache_key in disk_cache:
            embedded.append(EmbeddedChunk(
                id=chunk.id,
                text=chunk.text,
                embedding=disk_cache[cache_key],
                metadata=chunk.metadata,
            ))
        else:
            to_compute.append(chunk)
            to_compute_indices.append(i)
            embedded.append(None)  # Placeholder

    cache_hits = len(enriched_chunks) - len(to_compute)
    logger.info(
        "Embedding generation started",
        total=len(enriched_chunks),
        from_cache=cache_hits,
        to_compute=len(to_compute),
    )

    if to_compute:
        # Génération par batch
        for batch_start in range(0, len(to_compute), batch_size):
            batch = to_compute[batch_start:batch_start + batch_size]
            batch_indices = to_compute_indices[batch_start:batch_start + batch_size]
            batch_num = batch_start // batch_size + 1
            total_batches = (len(to_compute) + batch_size - 1) // batch_size

            logger.info(f"Embedding batch {batch_num}/{total_batches}", size=len(batch))

            texts = [c.text for c in batch]
            vectors = embed_batch(texts, use_cache=True, show_progress=False)

            for chunk, vector, original_idx in zip(batch, vectors, batch_indices):
                ec = EmbeddedChunk(
                    id=chunk.id,
                    text=chunk.text,
                    embedding=vector,
                    metadata=chunk.metadata,
                )
                embedded[original_idx] = ec

                # Mise à jour cache disque
                if use_disk_cache:
                    disk_cache[_get_text_hash(chunk.text)] = vector

        # Sauvegarde cache disque
        if use_disk_cache:
            _save_disk_cache(disk_cache)
            logger.info(
                "Embeddings disk cache updated",
                total_cached=len(disk_cache),
                cache_file=str(EMBEDDINGS_CACHE_FILE),
            )

    duration_ms = (time.time() - start_time) * 1000
    api_cache_stats = get_cache_stats()

    logger.info(
        "Embedding generation completed",
        total=len(embedded),
        cache_hits_disk=cache_hits,
        duration_ms=round(duration_ms),
        api_cache_stats=api_cache_stats,
    )

    return [e for e in embedded if e is not None]