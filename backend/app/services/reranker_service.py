"""
backend/app/services/reranker_service.py

Service de reranking sémantique post-retrieval.
Responsabilités :
- Recevoir query + documents du retriever
- Calculer un score de pertinence sémantique via LLM (OpenRouter)
- Trier et filtrer les documents
- Retourner les top_n documents les plus pertinents

Position dans le pipeline :
    Retriever → [RerankerService] → ContextBuilder

Stratégie : LLM reranking via OpenRouter
- Le LLM évalue la pertinence de chaque document par rapport à la query
- Score entre 0 et 10 retourné par le LLM
- Fallback sur scores de retrieval si erreur API

Optimisations :
- Cache des scores (évite recalcul)
- Fallback robuste
- Seuil de pertinence configurable
"""

import hashlib
import time
from dataclasses import dataclass, field

import httpx

from backend.app.db.vector_store import SearchResult
from backend.app.core.settings import get_settings
from backend.app.core.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

RERANKER_MODEL = "mistralai/mistral-small-3.1-24b-instruct"
RERANKER_TIMEOUT = 30.0
MAX_RETRIES = 2
DEFAULT_TOP_N = 4
MIN_RERANK_SCORE = 0.0


# ---------------------------------------------------------------------------
# Cache des scores de reranking
# ---------------------------------------------------------------------------

_rerank_cache: dict[str, float] = {}


def _get_rerank_cache_key(query: str, doc_text: str) -> str:
    """Génère une clé de cache pour un couple (query, document)."""
    content = f"{query.strip().lower()}|||{doc_text[:200].strip().lower()}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def clear_rerank_cache() -> None:
    """Vide le cache de reranking."""
    _rerank_cache.clear()
    logger.info("Rerank cache cleared")


def get_rerank_cache_stats() -> dict:
    """Statistiques du cache de reranking."""
    return {"cached_scores": len(_rerank_cache)}


# ---------------------------------------------------------------------------
# Types de données
# ---------------------------------------------------------------------------

@dataclass
class RankedDocument:
    """
    Document avec score de reranking.
    Encapsule le SearchResult original avec le nouveau score.
    """
    original: SearchResult
    rerank_score: float
    retrieval_score: float
    combined_score: float
    from_cache: bool = False

    @property
    def id(self) -> str:
        return self.original.id

    @property
    def text(self) -> str:
        return self.original.text

    @property
    def metadata(self) -> dict:
        return self.original.metadata


@dataclass
class RerankedResult:
    """
    Résultat complet du reranking.
    """
    query: str
    documents: list[RankedDocument]
    original_count: int
    reranked_count: int
    top_n: int
    method: str
    duration_ms: float
    cache_hits: int = 0
    fallback_used: bool = False
    score_stats: dict = field(default_factory=dict)

    def to_search_results(self) -> list[SearchResult]:
        """
        Convertit les RankedDocument en SearchResult
        avec les scores de reranking pour compatibilité ContextBuilder.
        """
        results = []
        for doc in self.documents:
            results.append(SearchResult(
                id=doc.original.id,
                text=doc.original.text,
                metadata=doc.original.metadata,
                distance=doc.original.distance,
                score=doc.combined_score,
            ))
        return results


# ---------------------------------------------------------------------------
# RerankerService
# ---------------------------------------------------------------------------

class RerankerService:
    """
    Service de reranking sémantique via LLM (OpenRouter).

    Fonctionnement :
    1. Pour chaque document, envoie un prompt au LLM
    2. Le LLM évalue la pertinence (0-10)
    3. Calcule un score combiné (retrieval + reranking)
    4. Trie et retourne les top_n documents

    Fallback :
    - Si API indisponible → utilise scores de retrieval uniquement
    - Si timeout → utilise scores de retrieval

    Usage :
        reranker = RerankerService()
        result = reranker.rerank(
            query="Quels sont les défis économiques ?",
            documents=retrieval_results,
            top_n=4,
        )
        search_results = result.to_search_results()
    """

    # Prompt pour évaluer la pertinence d'un document
    RERANK_PROMPT_TEMPLATE = """Tu es un expert en recherche documentaire.

Évalue la pertinence du document suivant par rapport à la question posée.

Question : {query}

Document :
{document_text}

Instructions :
- Donne un score de pertinence entre 0 et 10
- 0 = totalement non pertinent
- 5 = partiellement pertinent
- 10 = parfaitement pertinent et directement utile
- Réponds UNIQUEMENT avec un nombre entier entre 0 et 10
- Aucun texte supplémentaire

Score :"""

    def __init__(
        self,
        model: str = None,
        top_n: int = DEFAULT_TOP_N,
        min_score: float = MIN_RERANK_SCORE,
        rerank_weight: float = 0.7,
        use_cache: bool = True,
    ):
        """
        Args:
            model: Modèle LLM pour le reranking (défaut : mistral-small)
            top_n: Nombre de documents à conserver après reranking
            min_score: Score minimal pour conserver un document (0-10)
            rerank_weight: Poids du score reranking vs retrieval (0.7 = 70% rerank)
            use_cache: Activer le cache des scores
        """
        settings = get_settings()
        self._api_key = settings.openrouter_api_key
        self._base_url = settings.openrouter_base_url
        self._model = model or RERANKER_MODEL
        self._top_n = top_n
        self._min_score = min_score
        self._rerank_weight = rerank_weight
        self._retrieval_weight = 1.0 - rerank_weight
        self._use_cache = use_cache

        logger.info(
            "RerankerService initialized",
            model=self._model,
            top_n=self._top_n,
            min_score=self._min_score,
            rerank_weight=self._rerank_weight,
            use_cache=use_cache,
        )

    def rerank(
        self,
        query: str,
        documents: list[SearchResult],
        top_n: int = None,
    ) -> RerankedResult:
        """
        Reranke les documents selon leur pertinence sémantique.

        Étapes :
        1. Pour chaque document : calcul score reranking via LLM
        2. Calcul score combiné (reranking × weight + retrieval × weight)
        3. Tri par score combiné décroissant
        4. Filtrage par score minimum
        5. Sélection top_n

        Args:
            query: Question de l'utilisateur
            documents: Documents du retriever (SearchResult)
            top_n: Nombre de documents à garder (override défaut)

        Returns:
            RerankedResult avec documents rerankés et métriques
        """
        effective_top_n = top_n or self._top_n
        start_time = time.time()
        cache_hits = 0
        fallback_used = False

        if not documents:
            return RerankedResult(
                query=query,
                documents=[],
                original_count=0,
                reranked_count=0,
                top_n=effective_top_n,
                method="none",
                duration_ms=0.0,
            )

        logger.info(
            "Reranking started",
            query=query[:60],
            document_count=len(documents),
            top_n=effective_top_n,
            model=self._model,
        )

        ranked_docs = []

        for doc in documents:
            cache_key = _get_rerank_cache_key(query, doc.text)

            # Vérification cache
            if self._use_cache and cache_key in _rerank_cache:
                rerank_score = _rerank_cache[cache_key]
                cache_hits += 1
                from_cache = True
                logger.info(
                    "Rerank cache hit",
                    doc_id=doc.id,
                    score=rerank_score,
                )
            else:
                # Calcul score via LLM
                rerank_score = self._compute_score(query, doc.text)
                from_cache = False

                if self._use_cache:
                    _rerank_cache[cache_key] = rerank_score

            # Score normalisé (0-10 → 0-1)
            normalized_rerank = rerank_score / 10.0

            # Score combiné pondéré
            combined = (
                self._rerank_weight * normalized_rerank
                + self._retrieval_weight * doc.score
            )

            ranked_docs.append(RankedDocument(
                original=doc,
                rerank_score=rerank_score,
                retrieval_score=doc.score,
                combined_score=round(combined, 4),
                from_cache=from_cache,
            ))

        # Tri par score combiné décroissant
        ranked_docs.sort(key=lambda d: d.combined_score, reverse=True)

        # Filtrage par score minimum
        filtered = [
            d for d in ranked_docs
            if d.rerank_score >= self._min_score
        ]

        if len(filtered) < len(ranked_docs):
            logger.info(
                "Documents filtered by min_score",
                before=len(ranked_docs),
                after=len(filtered),
                min_score=self._min_score,
            )

        # Sélection top_n
        final_docs = filtered[:effective_top_n]

        # Métriques
        duration_ms = (time.time() - start_time) * 1000
        scores = [d.combined_score for d in final_docs]
        score_stats = {}
        if scores:
            score_stats = {
                "best": round(max(scores), 4),
                "worst": round(min(scores), 4),
                "avg": round(sum(scores) / len(scores), 4),
                "rerank_scores": [round(d.rerank_score, 1) for d in final_docs],
            }

        result = RerankedResult(
            query=query,
            documents=final_docs,
            original_count=len(documents),
            reranked_count=len(final_docs),
            top_n=effective_top_n,
            method="llm",
            duration_ms=round(duration_ms),
            cache_hits=cache_hits,
            fallback_used=fallback_used,
            score_stats=score_stats,
        )

        logger.info(
            "Reranking completed",
            original_count=len(documents),
            reranked_count=len(final_docs),
            cache_hits=cache_hits,
            duration_ms=round(duration_ms),
            score_stats=score_stats,
        )

        return result

    def _compute_score(self, query: str, document_text: str) -> float:
        """
        Calcule le score de pertinence d'un document via LLM.

        Envoie un prompt au LLM pour obtenir un score entre 0 et 10.
        Retry automatique en cas d'erreur.
        Fallback sur score 5.0 (neutre) si toutes les tentatives échouent.

        Args:
            query: Question de l'utilisateur
            document_text: Texte du document à évaluer

        Returns:
            Score de pertinence entre 0.0 et 10.0
        """
        # Tronquer le document pour éviter les prompts trop longs
        doc_preview = document_text[:800]

        prompt = self.RERANK_PROMPT_TEMPLATE.format(
            query=query,
            document_text=doc_preview,
        )

        for attempt in range(MAX_RETRIES):
            try:
                response = httpx.post(
                    url=f"{self._base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self._model,
                        "messages": [
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.0,
                        "max_tokens": 5,
                    },
                    timeout=RERANKER_TIMEOUT,
                )
                response.raise_for_status()
                data = response.json()
                raw_answer = data["choices"][0]["message"]["content"].strip()

                # Extraction du score numérique
                score = self._parse_score(raw_answer)
                logger.info(
                    "Rerank score computed",
                    doc_preview=document_text[:40],
                    raw_answer=raw_answer,
                    score=score,
                    attempt=attempt + 1,
                )
                return score

            except httpx.TimeoutException:
                logger.warning(
                    "Reranker timeout",
                    attempt=attempt + 1,
                    doc_preview=document_text[:40],
                )
                if attempt < MAX_RETRIES - 1:
                    time.sleep(1.0)

            except httpx.HTTPStatusError as e:
                logger.error(
                    "Reranker HTTP error",
                    status_code=e.response.status_code,
                    attempt=attempt + 1,
                )
                break

            except Exception as e:
                logger.error(
                    "Reranker unexpected error",
                    error=str(e),
                    attempt=attempt + 1,
                )
                break

        # Fallback : score neutre
        logger.warning(
            "Reranker fallback — using neutral score 5.0",
            doc_preview=document_text[:40],
        )
        return 5.0

    def _parse_score(self, raw: str) -> float:
        """
        Extrait le score numérique de la réponse LLM.

        Gère les cas :
        - "7" → 7.0
        - "7.5" → 7.5
        - "Score: 8" → 8.0
        - Texte parasite → 5.0 (neutre)

        Args:
            raw: Réponse brute du LLM

        Returns:
            Score entre 0.0 et 10.0
        """
        import re
        # Cherche le premier nombre dans la réponse
        matches = re.findall(r'\b(\d+(?:\.\d+)?)\b', raw)
        if matches:
            score = float(matches[0])
            # Clamp entre 0 et 10
            return max(0.0, min(10.0, score))

        logger.warning(
            "Could not parse reranker score",
            raw_answer=raw,
        )
        return 5.0

    def rerank_with_fallback(
        self,
        query: str,
        documents: list[SearchResult],
        top_n: int = None,
    ) -> tuple[list[SearchResult], bool]:
        """
        Version sécurisée : retourne toujours un résultat.
        Si le reranking échoue, retourne les documents de retrieval.

        Args:
            query: Question
            documents: Documents du retriever
            top_n: Nombre de documents à conserver

        Returns:
            Tuple (documents, fallback_used)
        """
        effective_top_n = top_n or self._top_n
        try:
            result = self.rerank(query, documents, top_n=effective_top_n)
            return result.to_search_results(), result.fallback_used
        except Exception as e:
            logger.error(
                "Reranker completely failed — using retrieval scores",
                error=str(e),
            )
            # Fallback : retourner les top_n premiers documents de retrieval
            return documents[:effective_top_n], True