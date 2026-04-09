"""
backend/app/services/retriever_service.py

Couche de retrieval RAG.
Améliorations Phase 6 :
- MMR (Maximal Marginal Relevance) : diversité des résultats
- Top-k adaptatif selon la distribution des scores
- Filtrage par score threshold dynamique
- Métriques de qualité du retrieval
"""

import math
from dataclasses import dataclass, field

from backend.app.services.embedding_service import embed_text
from backend.app.db.vector_store import VectorStore, SearchResult
from backend.app.core.settings import get_settings
from backend.app.core.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Types de données
# ---------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    """Résultat structuré d'une opération de retrieval."""
    query: str
    documents: list[SearchResult]
    top_k: int
    embedding_dim: int
    total_in_db: int
    retrieval_method: str = "cosine"
    score_stats: dict = field(default_factory=dict)

    @property
    def found(self) -> int:
        return len(self.documents)

    @property
    def is_empty(self) -> bool:
        return len(self.documents) == 0

    @property
    def best_score(self) -> float:
        if not self.documents:
            return 0.0
        return self.documents[0].score

    @property
    def avg_score(self) -> float:
        if not self.documents:
            return 0.0
        return sum(d.score for d in self.documents) / len(self.documents)


# ---------------------------------------------------------------------------
# MMR — Maximal Marginal Relevance
# ---------------------------------------------------------------------------

def _cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Calcule la similarité cosine entre deux vecteurs."""
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a ** 2 for a in v1))
    norm2 = math.sqrt(sum(b ** 2 for b in v2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def _apply_mmr(
    query_embedding: list[float],
    candidates: list[SearchResult],
    candidate_embeddings: list[list[float]],
    top_k: int,
    lambda_param: float = 0.7,
) -> list[SearchResult]:
    """
    Maximal Marginal Relevance — sélectionne des documents
    à la fois pertinents ET diversifiés.

    MMR score = lambda * relevance - (1 - lambda) * max_similarity_to_selected

    lambda=1.0 → pur ranking par score (pas de diversification)
    lambda=0.0 → pur diversité (ignore la pertinence)
    lambda=0.7 → équilibre recommandé

    Args:
        query_embedding: Vecteur de la requête
        candidates: Documents candidats triés par score
        candidate_embeddings: Embeddings des candidats
        top_k: Nombre de documents à sélectionner
        lambda_param: Équilibre pertinence/diversité

    Returns:
        Liste de documents sélectionnés par MMR
    """
    if not candidates:
        return []

    selected_indices = []
    selected_embeddings = []
    remaining = list(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        best_idx = None
        best_score = float("-inf")

        for i in remaining:
            relevance = candidates[i].score

            if not selected_embeddings:
                mmr_score = relevance
            else:
                max_sim = max(
                    _cosine_similarity(candidate_embeddings[i], sel_emb)
                    for sel_emb in selected_embeddings
                )
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i

        if best_idx is not None:
            selected_indices.append(best_idx)
            selected_embeddings.append(candidate_embeddings[best_idx])
            remaining.remove(best_idx)

    return [candidates[i] for i in selected_indices]


# ---------------------------------------------------------------------------
# Retriever Service
# ---------------------------------------------------------------------------

class RetrieverService:
    """
    Service de retrieval sémantique amélioré.

    Fonctionnalités :
    - Retrieval standard (cosine similarity)
    - MMR pour diversifier les résultats
    - Top-k adaptatif selon distribution des scores
    - Métriques de qualité

    Usage :
        retriever = RetrieverService()
        result = retriever.retrieve("Qu'est-ce que ChromaDB ?")
    """

    def __init__(self, top_k: int = None):
        settings = get_settings()
        self._top_k = top_k or settings.retrieval_top_k
        self._store = VectorStore()
        logger.info("RetrieverService initialized", top_k=self._top_k)

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        score_threshold: float = None,
        use_mmr: bool = False,
        mmr_lambda: float = 0.7,
        adaptive_k: bool = False,
    ) -> RetrievalResult:
        """
        Retrieval sémantique avec options avancées.

        Args:
            query: Question en langage naturel
            top_k: Nombre de documents
            score_threshold: Score minimal
            use_mmr: Activer MMR pour diversifier les résultats
            mmr_lambda: Paramètre MMR (0.7 recommandé)
            adaptive_k: Ajuster top_k selon la distribution des scores

        Returns:
            RetrievalResult avec documents et métriques
        """
        if not query or not query.strip():
            raise ValueError("La requête ne peut pas être vide.")

        settings = get_settings()
        effective_top_k = top_k or self._top_k
        effective_threshold = score_threshold \
            if score_threshold is not None \
            else settings.retrieval_score_threshold

        logger.info(
            "Starting retrieval",
            query=query,
            top_k=effective_top_k,
            score_threshold=effective_threshold,
            use_mmr=use_mmr,
            adaptive_k=adaptive_k,
        )

        # Embedding de la requête (avec cache)
        query_embedding = embed_text(query, use_cache=True)

        # Récupérer plus de candidats si MMR activé
        fetch_k = effective_top_k * 3 if use_mmr else effective_top_k

        raw_results = self._store.search(
            query_text=query,
            top_k=min(fetch_k, self._store.count() or 1),
            query_embedding=query_embedding,
        )

        # Filtrage par score threshold
        filtered = [
            doc for doc in raw_results
            if doc.score >= effective_threshold
        ]

        if len(filtered) < len(raw_results):
            logger.info(
                "Documents filtered by threshold",
                before=len(raw_results),
                after=len(filtered),
                threshold=effective_threshold,
            )

        # Top-k adaptatif
        if adaptive_k and filtered:
            filtered = self._adaptive_top_k(filtered, effective_top_k)

        # MMR pour diversité
        if use_mmr and len(filtered) > 1:
            # Récupérer les embeddings des candidats pour MMR
            candidate_embeddings = [
                embed_text(doc.text, use_cache=True) for doc in filtered
            ]
            filtered = _apply_mmr(
                query_embedding=query_embedding,
                candidates=filtered,
                candidate_embeddings=candidate_embeddings,
                top_k=effective_top_k,
                lambda_param=mmr_lambda,
            )
            logger.info(
                "MMR applied",
                selected=len(filtered),
                lambda_param=mmr_lambda,
            )
        else:
            filtered = filtered[:effective_top_k]

        # Calcul métriques
        score_stats = {}
        if filtered:
            scores = [d.score for d in filtered]
            score_stats = {
                "best": round(max(scores), 4),
                "worst": round(min(scores), 4),
                "avg": round(sum(scores) / len(scores), 4),
                "spread": round(max(scores) - min(scores), 4),
            }

        result = RetrievalResult(
            query=query,
            documents=filtered,
            top_k=effective_top_k,
            embedding_dim=len(query_embedding),
            total_in_db=self._store.count(),
            retrieval_method="mmr" if use_mmr else "cosine",
            score_stats=score_stats,
        )

        logger.info(
            "Retrieval completed",
            found=result.found,
            method=result.retrieval_method,
            score_stats=score_stats,
        )

        return result

    def _adaptive_top_k(
        self,
        documents: list[SearchResult],
        max_k: int,
    ) -> list[SearchResult]:
        """
        Ajuste le nombre de documents selon la distribution des scores.

        Stratégie : inclure les documents dont le score est
        supérieur à (best_score * 0.6) pour éviter les documents
        trop éloignés sémantiquement.

        Args:
            documents: Documents triés par score décroissant
            max_k: Limite maximale

        Returns:
            Sous-ensemble adaptatif de documents
        """
        if not documents:
            return documents

        best_score = documents[0].score
        threshold = best_score * 0.6

        adaptive = [d for d in documents if d.score >= threshold]
        result = adaptive[:max_k]

        logger.info(
            "Adaptive top-k applied",
            original=len(documents),
            selected=len(result),
            adaptive_threshold=round(threshold, 4),
        )

        return result

    def retrieve_raw(
        self,
        query: str,
        top_k: int = None,
    ) -> list[SearchResult]:
        """Version simplifiée sans filtrage."""
        effective_top_k = top_k or self._top_k
        query_embedding = embed_text(query, use_cache=True)
        return self._store.search(
            query_text=query,
            top_k=effective_top_k,
            query_embedding=query_embedding,
        )