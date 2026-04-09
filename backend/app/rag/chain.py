"""
backend/app/rag/chain.py

Pipeline RAG complet — version améliorée Phase 6.
Améliorations :
- Cache des requêtes récentes (évite recomputation)
- Métriques détaillées par étape
- Gestion adaptative selon confiance du contexte
- Support MMR pour diversité des résultats
- Rapport de qualité complet
"""

import time
import hashlib
from dataclasses import dataclass, field

from backend.app.services.retriever_service import RetrieverService
from backend.app.services.llm_service import LLMService
from backend.app.rag.context import ContextBuilder
from backend.app.rag.prompts import PromptBuilder
from backend.app.core.settings import get_settings
from backend.app.core.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Cache des réponses récentes
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

    def format_full(self) -> str:
        """Formate le résultat complet pour affichage."""
        lines = [
            "=" * 60,
            f"QUESTION : {self.question}",
            f"CONFIANCE : {self.confidence_level.upper()} "
            f"(score={self.quality_score:.3f})",
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
                    f"  [{i}] {src['id']} | topic={topic} | "
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
            f"  Depuis cache    : {self.from_cache}",
            f"  Durée totale    : {self.total_duration_ms:.0f}ms",
            "",
            "ETAPES :",
        ])

        for step in self.steps:
            status = "OK" if step.success else "FAIL"
            lines.append(
                f"  [{status}] {step.name:<20} {step.duration_ms:.0f}ms"
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
    Pipeline RAG complet et optimisé.

    Améliorations Phase 6 :
    - Cache requêtes pour éviter recomputation
    - MMR optionnel pour diversifier les résultats
    - Prompts adaptatifs selon confiance du contexte
    - Métriques détaillées par étape
    - Rapport de qualité complet

    Usage :
        pipeline = RAGPipeline()
        result = pipeline.run("Qu'est-ce que ChromaDB ?")
        print(result.format_full())
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
    ):
        """
        Args:
            top_k: Documents à récupérer
            score_threshold: Score minimal similarité
            max_chars_per_doc: Limite caractères par document
            max_total_chars: Limite totale du contexte
            model: Modèle LLM OpenRouter
            temperature: Créativité LLM
            max_tokens: Tokens maximum sortie
            use_mmr: Activer MMR pour diversité
            mmr_lambda: Paramètre MMR (0.7 = équilibre)
            adaptive_k: Top-k adaptatif selon scores
            use_cache: Cache des requêtes récentes
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

        self._retriever = RetrieverService(top_k=self._top_k)
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
            use_cache=use_cache,
            model=self._llm._model,
        )

    def run(self, question: str) -> RAGPipelineResult:
        """
        Exécute le pipeline RAG complet.

        Étapes :
        1. Vérification cache
        2. Retrieval (cosine + MMR optionnel)
        3. Construction contexte (avec déduplication)
        4. Construction prompt adaptatif
        5. Génération LLM
        6. Mise en cache du résultat

        Args:
            question: Question en langage naturel

        Returns:
            RAGPipelineResult avec réponse et métriques complètes
        """
        if not question or not question.strip():
            raise ValueError("La question ne peut pas être vide.")

        # Vérification cache
        cache_key = _get_query_cache_key(
            question, self._top_k, self._llm._model
        )
        if self._use_cache and cache_key in _query_cache:
            cached = _query_cache[cache_key]
            logger.info(
                "Query cache hit",
                question=question,
                cache_key=cache_key,
            )
            cached.from_cache = True
            return cached

        pipeline_start = time.time()
        steps = []

        logger.info(
            "RAG Pipeline started",
            question=question,
            top_k=self._top_k,
            use_mmr=self._use_mmr,
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
                "Step 1/4 — Retrieval",
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
        # Étape 2 : Construction contexte
        # ------------------------------------------------------------------
        step_start = time.time()
        try:
            context = self._context_builder.build(retrieval_result)
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
                "Step 2/4 — Context built",
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
        # Étape 3 : Construction prompt adaptatif
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
                "Step 3/4 — Prompt built",
                prompt_type=prompt.prompt_type,
                confidence_level=prompt.confidence_level,
                prompt_length=prompt.total_length(),
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
        # Étape 4 : Génération LLM
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
                "Step 4/4 — LLM generation",
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
        )

        # Mise en cache
        if self._use_cache:
            if len(_query_cache) >= _MAX_CACHE_SIZE:
                oldest_key = next(iter(_query_cache))
                del _query_cache[oldest_key]
            _query_cache[cache_key] = result

        logger.info(
            "RAG Pipeline completed",
            question=question,
            answer_length=len(answer),
            quality_score=round(context.quality_score, 3),
            confidence_level=context.confidence_level,
            total_duration_ms=round(total_duration_ms),
            from_cache=False,
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
                "duration_ms": rag_result.total_duration_ms,
            },
            "without_rag": {
                "answer": answer_no_rag,
                "sources": [],
                "document_count": 0,
                "quality_score": 0.0,
                "confidence_level": "none",
                "duration_ms": 0,
            },
        }


# ---------------------------------------------------------------------------
# Fonction utilitaire
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
) -> RAGPipelineResult:
    """
    Point d'entrée unique du pipeline RAG amélioré.

    Args:
        question: Question en langage naturel
        top_k: Documents à récupérer
        score_threshold: Score minimal
        model: Modèle LLM
        temperature: Créativité LLM
        max_tokens: Tokens maximum
        use_mmr: Activer MMR
        adaptive_k: Top-k adaptatif
        use_cache: Cache requêtes

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
    )
    return pipeline.run(question)