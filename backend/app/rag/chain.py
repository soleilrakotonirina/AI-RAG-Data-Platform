"""
backend/app/rag/chain.py

Pipeline RAG complet — version améliorée Phase 6.
Améliorations :
- Cache des requêtes récentes (évite recomputation)
- Métriques détaillées par étape
- Gestion adaptative selon confiance du contexte
- Support MMR pour diversité des résultats
- Rapport de qualité complet

Pipeline RAG complet — Phase 8 : ajout reranking.

Nouvelle séquence :
1. Retrieval   — embedding + ChromaDB + MMR
2. Reranking   — score sémantique LLM (NOUVEAU)
3. Context     — déduplication + qualité
4. Prompt      — template adaptatif EN→FR
5. LLM         — génération réponse
"""

import time
import hashlib
from dataclasses import dataclass, field

from backend.app.services.retriever_service import RetrieverService
from backend.app.services.reranker_service import RerankerService
from backend.app.services.llm_service import LLMService
from backend.app.rag.context import ContextBuilder
from backend.app.rag.prompts import PromptBuilder
from backend.app.core.settings import get_settings
from backend.app.core.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Cache des réponses
# ---------------------------------------------------------------------------

_query_cache: dict[str, "RAGPipelineResult"] = {}
_MAX_CACHE_SIZE = 50


def _get_query_cache_key(question: str, top_k: int, model: str) -> str:
    """Génère une clé de cache pour une requête."""
    content = f"{question.strip().lower()}|{top_k}|{model}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def clear_query_cache() -> None:
    """Vide le cache des requêtes."""
    _query_cache.clear()
    logger.info("Query cache cleared")


def get_query_cache_stats() -> dict:
    """Statistiques du cache requêtes."""
    return {"cached_queries": len(_query_cache), "max_size": _MAX_CACHE_SIZE}


# ---------------------------------------------------------------------------
# Types de données
# ---------------------------------------------------------------------------

@dataclass
class PipelineStep:
    """Résultat d'une étape avec timing et détails."""
    name: str
    duration_ms: float
    success: bool
    details: dict = field(default_factory=dict)


@dataclass
class RAGPipelineResult:
    """Résultat complet du pipeline RAG."""
    question: str
    answer: str
    context_text: str
    sources: list[dict]
    document_count: int
    context_used: bool
    model: str
    steps: list[PipelineStep]
    total_duration_ms: float
    quality_score: float = 0.0
    confidence_level: str = "none"
    from_cache: bool = False
    reranking_used: bool = False

    def format_full(self) -> str:
        lines = [
            "=" * 60,
            f"QUESTION : {self.question}",
            f"CONFIANCE : {self.confidence_level.upper()} "
            f"(score={self.quality_score:.3f})",
            f"RERANKING : {'OUI' if self.reranking_used else 'NON'}",
            "=" * 60,
            "",
            "CONTEXTE UTILISE :",
            "-" * 40,
            self.context_text if self.context_text else "(aucun contexte)",
            "-" * 40,
            "",
            "REPONSE :",
            self.answer,
            "",
        ]

        if self.sources:
            lines.append("SOURCES :")
            for i, src in enumerate(self.sources, 1):
                topic = src.get("metadata", {}).get("topic", "?")
                score = src.get("score", 0)
                confidence = src.get("confidence", "?")
                lines.append(
                    f"  [{i}] {src['id'][:50]} | topic={topic} | "
                    f"score={score:.4f} | confiance={confidence}"
                )
            lines.append("")

        lines.extend([
            "METRIQUES :",
            f"  Modele          : {self.model}",
            f"  Documents       : {self.document_count}",
            f"  Contexte used   : {self.context_used}",
            f"  Qualité         : {self.quality_score:.3f}",
            f"  Confiance       : {self.confidence_level}",
            f"  Reranking       : {self.reranking_used}",
            f"  Depuis cache    : {self.from_cache}",
            f"  Durée totale    : {self.total_duration_ms:.0f}ms",
            "",
            "ETAPES :",
        ])

        for step in self.steps:
            status = "OK" if step.success else "FAIL"
            lines.append(
                f"  [{status}] {step.name:<22} {step.duration_ms:.0f}ms"
            )
            if step.details:
                for k, v in step.details.items():
                    lines.append(f"           {k}: {v}")

        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# RAG Pipeline
# ---------------------------------------------------------------------------

class RAGPipeline:
    """
    Pipeline RAG complet — Phase 8.

    Séquence :
    1. Retrieval   (ChromaDB + MMR)
    2. Reranking   (LLM sémantique) ← NOUVEAU
    3. Context     (déduplication + qualité)
    4. Prompt      (adaptatif EN→FR)
    5. LLM         (génération)
    """

    def __init__(
        self,
        top_k: int = None,
        score_threshold: float = None,
        max_chars_per_doc: int = 800,
        max_total_chars: int = 4000,
        model: str = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        use_mmr: bool = True,
        mmr_lambda: float = 0.7,
        adaptive_k: bool = True,
        use_cache: bool = True,
        use_reranking: bool = True,
        rerank_top_n: int = 4,
        rerank_min_score: float = 0.0,
    ):
        """
        Args:
            top_k: Documents à récupérer (retrieval)
            score_threshold: Score minimal similarité
            max_chars_per_doc: Limite caractères par document
            max_total_chars: Limite totale du contexte
            model: Modèle LLM OpenRouter
            temperature: Créativité LLM
            max_tokens: Tokens maximum sortie
            use_mmr: Activer MMR
            mmr_lambda: Paramètre MMR
            adaptive_k: Top-k adaptatif
            use_cache: Cache requêtes
            use_reranking: Activer le reranking sémantique
            rerank_top_n: Documents conservés après reranking
            rerank_min_score: Score LLM minimal (0-10)
        """
        settings = get_settings()

        self._top_k = top_k or settings.retrieval_top_k
        self._score_threshold = score_threshold \
            if score_threshold is not None \
            else settings.retrieval_score_threshold
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._use_mmr = use_mmr
        self._mmr_lambda = mmr_lambda
        self._adaptive_k = adaptive_k
        self._use_cache = use_cache
        self._use_reranking = use_reranking

        self._retriever = RetrieverService(top_k=self._top_k)
        self._reranker = RerankerService(
            top_n=rerank_top_n,
            min_score=rerank_min_score,
            use_cache=use_cache,
        ) if use_reranking else None
        self._context_builder = ContextBuilder(
            max_chars_per_doc=max_chars_per_doc,
            max_total_chars=max_total_chars,
        )
        self._prompt_builder = PromptBuilder()
        self._llm = LLMService(model=model)

        logger.info(
            "RAGPipeline initialized",
            top_k=self._top_k,
            score_threshold=self._score_threshold,
            use_mmr=use_mmr,
            adaptive_k=adaptive_k,
            use_reranking=use_reranking,
            rerank_top_n=rerank_top_n,
            use_cache=use_cache,
            model=self._llm._model,
        )

    def run(self, question: str) -> RAGPipelineResult:
        """
        Exécute le pipeline RAG complet avec reranking.

        Étapes :
        1. Vérification cache
        2. Retrieval (cosine + MMR)
        3. Reranking sémantique LLM (si activé)
        4. Construction contexte
        5. Construction prompt adaptatif
        6. Génération LLM
        7. Mise en cache

        Args:
            question: Question en langage naturel

        Returns:
            RAGPipelineResult avec réponse et métriques
        """
        if not question or not question.strip():
            raise ValueError("La question ne peut pas être vide.")

        # Vérification cache
        cache_key = _get_query_cache_key(
            question, self._top_k, self._llm._model
        )
        if self._use_cache and cache_key in _query_cache:
            cached = _query_cache[cache_key]
            logger.info("Query cache hit", question=question[:60])
            cached.from_cache = True
            return cached

        pipeline_start = time.time()
        steps = []
        reranking_used = False

        logger.info(
            "RAG Pipeline started",
            question=question,
            top_k=self._top_k,
            use_mmr=self._use_mmr,
            use_reranking=self._use_reranking,
        )

        # ------------------------------------------------------------------
        # Étape 1 : Retrieval
        # ------------------------------------------------------------------
        step_start = time.time()
        try:
            retrieval_result = self._retriever.retrieve(
                query=question,
                score_threshold=self._score_threshold,
                use_mmr=self._use_mmr,
                mmr_lambda=self._mmr_lambda,
                adaptive_k=self._adaptive_k,
            )
            step_duration = (time.time() - step_start) * 1000
            steps.append(PipelineStep(
                name="retrieval",
                duration_ms=step_duration,
                success=True,
                details={
                    "found": retrieval_result.found,
                    "method": retrieval_result.retrieval_method,
                    "score_stats": retrieval_result.score_stats,
                    "total_in_db": retrieval_result.total_in_db,
                },
            ))
            logger.info(
                "Step 1/5 — Retrieval",
                found=retrieval_result.found,
                method=retrieval_result.retrieval_method,
                score_stats=retrieval_result.score_stats,
                duration_ms=round(step_duration),
            )
        except Exception as e:
            step_duration = (time.time() - step_start) * 1000
            steps.append(PipelineStep(
                name="retrieval",
                duration_ms=step_duration,
                success=False,
                details={"error": str(e)},
            ))
            logger.error("Retrieval failed", error=str(e))
            raise

        # ------------------------------------------------------------------
        # Étape 2 : Reranking (NOUVEAU)
        # ------------------------------------------------------------------
        documents_for_context = retrieval_result.documents

        if self._use_reranking and self._reranker and retrieval_result.found > 0:
            step_start = time.time()
            try:
                reranked_docs, fallback_used = self._reranker.rerank_with_fallback(
                    query=question,
                    documents=retrieval_result.documents,
                )

                step_duration = (time.time() - step_start) * 1000
                reranking_used = not fallback_used

                rerank_stats = get_rerank_cache_stats() if not fallback_used else {}

                steps.append(PipelineStep(
                    name="reranking",
                    duration_ms=step_duration,
                    success=True,
                    details={
                        "original_count": retrieval_result.found,
                        "reranked_count": len(reranked_docs),
                        "fallback_used": fallback_used,
                        "cache_stats": rerank_stats,
                    },
                ))
                logger.info(
                    "Step 2/5 — Reranking",
                    original_count=retrieval_result.found,
                    reranked_count=len(reranked_docs),
                    fallback_used=fallback_used,
                    duration_ms=round(step_duration),
                )

                # Mettre à jour les documents pour le contexte
                documents_for_context = reranked_docs

            except Exception as e:
                step_duration = (time.time() - step_start) * 1000
                steps.append(PipelineStep(
                    name="reranking",
                    duration_ms=step_duration,
                    success=False,
                    details={"error": str(e), "fallback": "retrieval_scores"},
                ))
                logger.error(
                    "Reranking failed — using retrieval results",
                    error=str(e),
                )
                # Fallback : continuer avec les résultats du retriever
        else:
            if self._use_reranking:
                logger.info("Reranking skipped — no documents to rerank")

        # ------------------------------------------------------------------
        # Étape 3 : Construction contexte
        # ------------------------------------------------------------------
        step_start = time.time()
        try:
            from backend.app.services.retriever_service import RetrievalResult
            context_input = RetrievalResult(
                query=question,
                documents=documents_for_context,
                top_k=len(documents_for_context),
                embedding_dim=retrieval_result.embedding_dim,
                total_in_db=retrieval_result.total_in_db,
                retrieval_method=retrieval_result.retrieval_method,
            )
            context = self._context_builder.build(context_input)
            step_duration = (time.time() - step_start) * 1000
            steps.append(PipelineStep(
                name="context_build",
                duration_ms=step_duration,
                success=True,
                details={
                    "document_count": context.document_count,
                    "context_length": len(context.text),
                    "quality_score": round(context.quality_score, 3),
                    "confidence_level": context.confidence_level,
                    "truncated": context.truncated,
                },
            ))
            logger.info(
                "Step 3/5 — Context built",
                document_count=context.document_count,
                quality_score=round(context.quality_score, 3),
                confidence_level=context.confidence_level,
                duration_ms=round(step_duration),
            )
        except Exception as e:
            step_duration = (time.time() - step_start) * 1000
            steps.append(PipelineStep(
                name="context_build",
                duration_ms=step_duration,
                success=False,
            ))
            logger.error("Context build failed", error=str(e))
            raise

        # ------------------------------------------------------------------
        # Étape 4 : Construction prompt
        # ------------------------------------------------------------------
        step_start = time.time()
        try:
            if context.is_empty:
                prompt = self._prompt_builder.build_no_context_prompt(
                    question=question,
                )
            else:
                prompt = self._prompt_builder.build_rag_prompt(
                    question=question,
                    context_block=context.to_prompt_block(),
                    confidence_level=context.confidence_level,
                )

            messages = prompt.to_messages()
            step_duration = (time.time() - step_start) * 1000
            steps.append(PipelineStep(
                name="prompt_build",
                duration_ms=step_duration,
                success=True,
                details={
                    "prompt_type": prompt.prompt_type,
                    "confidence_level": prompt.confidence_level,
                    "prompt_length": prompt.total_length(),
                },
            ))
            logger.info(
                "Step 4/5 — Prompt built",
                prompt_type=prompt.prompt_type,
                confidence_level=prompt.confidence_level,
                duration_ms=round(step_duration),
            )
        except Exception as e:
            step_duration = (time.time() - step_start) * 1000
            steps.append(PipelineStep(
                name="prompt_build",
                duration_ms=step_duration,
                success=False,
            ))
            logger.error("Prompt build failed", error=str(e))
            raise

        # ------------------------------------------------------------------
        # Étape 5 : Génération LLM
        # ------------------------------------------------------------------
        step_start = time.time()
        try:
            answer = self._llm._client.generate_completion(
                messages=messages,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
            step_duration = (time.time() - step_start) * 1000
            steps.append(PipelineStep(
                name="llm_generation",
                duration_ms=step_duration,
                success=True,
                details={"answer_length": len(answer)},
            ))
            logger.info(
                "Step 5/5 — LLM generation",
                answer_length=len(answer),
                duration_ms=round(step_duration),
            )
        except Exception as e:
            step_duration = (time.time() - step_start) * 1000
            steps.append(PipelineStep(
                name="llm_generation",
                duration_ms=step_duration,
                success=False,
            ))
            logger.error("LLM generation failed", error=str(e))
            raise

        # ------------------------------------------------------------------
        # Résultat final
        # ------------------------------------------------------------------
        total_duration_ms = (time.time() - pipeline_start) * 1000

        sources = [
            {
                "id": src.id,
                "score": src.score,
                "confidence": src.confidence,
                "metadata": src.metadata,
            }
            for src in context.sources
        ]

        result = RAGPipelineResult(
            question=question,
            answer=answer,
            context_text=context.text,
            sources=sources,
            document_count=context.document_count,
            context_used=not context.is_empty,
            model=self._llm._model,
            steps=steps,
            total_duration_ms=total_duration_ms,
            quality_score=context.quality_score,
            confidence_level=context.confidence_level,
            from_cache=False,
            reranking_used=reranking_used,
        )

        # Mise en cache
        if self._use_cache:
            if len(_query_cache) >= _MAX_CACHE_SIZE:
                oldest_key = next(iter(_query_cache))
                del _query_cache[oldest_key]
            _query_cache[cache_key] = result

        logger.info(
            "RAG Pipeline completed",
            question=question[:60],
            answer_length=len(answer),
            quality_score=round(context.quality_score, 3),
            confidence_level=context.confidence_level,
            reranking_used=reranking_used,
            total_duration_ms=round(total_duration_ms),
        )

        return result

    def compare(self, question: str) -> dict:
        """Compare réponse avec RAG vs sans RAG."""
        logger.info("Starting RAG comparison", question=question)

        rag_result = self.run(question)

        prompt_no_rag = self._prompt_builder.build_comparison_prompt(question)
        answer_no_rag = self._llm._client.generate_completion(
            messages=prompt_no_rag.to_messages(),
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )

        return {
            "question": question,
            "with_rag": {
                "answer": rag_result.answer,
                "sources": rag_result.sources,
                "document_count": rag_result.document_count,
                "quality_score": rag_result.quality_score,
                "confidence_level": rag_result.confidence_level,
                "reranking_used": rag_result.reranking_used,
                "duration_ms": rag_result.total_duration_ms,
            },
            "without_rag": {
                "answer": answer_no_rag,
                "sources": [],
                "document_count": 0,
                "quality_score": 0.0,
                "confidence_level": "none",
                "reranking_used": False,
                "duration_ms": 0,
            },
        }


# ---------------------------------------------------------------------------
# Import helper pour chain.py
# ---------------------------------------------------------------------------

def get_rerank_cache_stats() -> dict:
    from backend.app.services.reranker_service import get_rerank_cache_stats as _stats
    return _stats()


# ---------------------------------------------------------------------------
# Fonction utilitaire — point d'entrée
# ---------------------------------------------------------------------------

def run_rag_pipeline(
    question: str,
    top_k: int = None,
    score_threshold: float = None,
    model: str = None,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    use_mmr: bool = True,
    adaptive_k: bool = True,
    use_cache: bool = True,
    use_reranking: bool = True,
    rerank_top_n: int = 4,
) -> RAGPipelineResult:
    """
    Point d'entrée unique du pipeline RAG avec reranking.

    Args:
        question: Question en langage naturel
        top_k: Documents à récupérer (retrieval)
        score_threshold: Score minimal
        model: Modèle LLM
        temperature: Créativité LLM
        max_tokens: Tokens maximum
        use_mmr: Activer MMR
        adaptive_k: Top-k adaptatif
        use_cache: Cache requêtes
        use_reranking: Activer reranking sémantique
        rerank_top_n: Documents conservés après reranking

    Returns:
        RAGPipelineResult complet
    """
    pipeline = RAGPipeline(
        top_k=top_k,
        score_threshold=score_threshold,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        use_mmr=use_mmr,
        adaptive_k=adaptive_k,
        use_cache=use_cache,
        use_reranking=use_reranking,
        rerank_top_n=rerank_top_n,
    )
    return pipeline.run(question)