"""
backend/app/rag/context.py

Couche de construction de contexte RAG.
Améliorations Phase 6 :
- Déduplication des documents similaires
- Score de qualité du contexte global
- Formatage enrichi avec indicateur de confiance
- Compression intelligente du contexte
"""

from dataclasses import dataclass, field
import math

from backend.app.db.vector_store import SearchResult
from backend.app.services.retriever_service import RetrievalResult
from backend.app.core.logger import get_logger

logger = get_logger(__name__)

MAX_CHARS_PER_DOC = 1000
DOCUMENT_SEPARATOR = "\n\n---\n\n"
DEDUP_SIMILARITY_THRESHOLD = 0.95


# ---------------------------------------------------------------------------
# Types de données
# ---------------------------------------------------------------------------

@dataclass
class SourceReference:
    """Référence à un document source."""
    id: str
    score: float
    metadata: dict
    confidence: str = "medium"

    def __post_init__(self):
        if self.score >= 0.7:
            self.confidence = "high"
        elif self.score >= 0.4:
            self.confidence = "medium"
        else:
            self.confidence = "low"


@dataclass
class Context:
    """Contexte RAG complet prêt pour injection dans un prompt LLM."""
    query: str
    text: str
    sources: list[SourceReference]
    document_count: int
    truncated: bool = False
    quality_score: float = 0.0
    confidence_level: str = "low"

    @property
    def is_empty(self) -> bool:
        return self.document_count == 0

    def to_prompt_block(self) -> str:
        """Formate le contexte pour injection dans le prompt."""
        if self.is_empty:
            return "<context>\nAucun document pertinent trouvé.\n</context>"
        return f"<context>\n{self.text}\n</context>"

    def format_sources(self) -> str:
        """Formate les sources pour affichage."""
        if not self.sources:
            return "Aucune source."
        lines = []
        for i, src in enumerate(self.sources, 1):
            topic = src.metadata.get("topic", "?")
            source = src.metadata.get("source", "?")
            lines.append(
                f"  [{i}] {src.id} | score={src.score:.4f} | "
                f"confidence={src.confidence} | "
                f"topic={topic} | source={source}"
            )
        return "\n".join(lines)

    def get_quality_report(self) -> dict:
        """Retourne un rapport de qualité du contexte."""
        return {
            "quality_score": round(self.quality_score, 3),
            "confidence_level": self.confidence_level,
            "document_count": self.document_count,
            "truncated": self.truncated,
            "context_length": len(self.text),
            "sources": [
                {
                    "id": s.id,
                    "score": s.score,
                    "confidence": s.confidence,
                }
                for s in self.sources
            ],
        }


# ---------------------------------------------------------------------------
# Context Builder
# ---------------------------------------------------------------------------

class ContextBuilder:
    """
    Construit un objet Context à partir des résultats de retrieval.

    Améliorations :
    - Déduplication des documents trop similaires
    - Score de qualité global du contexte
    - Indicateur de confiance par document
    - Compression intelligente si contexte trop long
    """

    def __init__(
        self,
        max_chars_per_doc: int = MAX_CHARS_PER_DOC,
        separator: str = DOCUMENT_SEPARATOR,
        max_total_chars: int = 4000,
        dedup_threshold: float = DEDUP_SIMILARITY_THRESHOLD,
    ):
        """
        Args:
            max_chars_per_doc: Limite de caractères par document
            separator: Séparateur entre blocs
            max_total_chars: Limite totale du contexte
            dedup_threshold: Seuil de déduplication (0.0-1.0)
        """
        self._max_chars = max_chars_per_doc
        self._separator = separator
        self._max_total = max_total_chars
        self._dedup_threshold = dedup_threshold

        logger.info(
            "ContextBuilder initialized",
            max_chars_per_doc=max_chars_per_doc,
            max_total_chars=max_total_chars,
            dedup_threshold=dedup_threshold,
        )

    def build(self, retrieval_result: RetrievalResult) -> Context:
        """
        Construit un Context optimisé depuis un RetrievalResult.

        Args:
            retrieval_result: Résultat du RetrieverService

        Returns:
            Context avec qualité évaluée
        """
        if retrieval_result.is_empty:
            logger.warning(
                "No documents for context",
                query=retrieval_result.query,
            )
            return Context(
                query=retrieval_result.query,
                text="",
                sources=[],
                document_count=0,
                quality_score=0.0,
                confidence_level="none",
            )

        # Déduplication
        documents = self._deduplicate(retrieval_result.documents)

        blocks = []
        sources = []
        truncated = False
        total_chars = 0

        for i, doc in enumerate(documents, 1):
            text = doc.text

            # Troncature par document
            if len(text) > self._max_chars:
                text = text[:self._max_chars] + "..."
                truncated = True

            # Limite totale du contexte
            block = self._format_block(i, doc, text)
            if total_chars + len(block) > self._max_total:
                logger.warning(
                    "Context total limit reached",
                    documents_included=i - 1,
                    total_chars=total_chars,
                    limit=self._max_total,
                )
                break

            blocks.append(block)
            total_chars += len(block)

            sources.append(SourceReference(
                id=doc.id,
                score=doc.score,
                metadata=doc.metadata or {},
            ))

        full_context = self._separator.join(blocks)
        quality_score = self._compute_quality_score(sources)
        confidence_level = self._compute_confidence(quality_score)

        context = Context(
            query=retrieval_result.query,
            text=full_context,
            sources=sources,
            document_count=len(blocks),
            truncated=truncated,
            quality_score=quality_score,
            confidence_level=confidence_level,
        )

        logger.info(
            "Context built",
            query=retrieval_result.query,
            document_count=context.document_count,
            context_length=len(full_context),
            quality_score=round(quality_score, 3),
            confidence_level=confidence_level,
            truncated=truncated,
        )

        return context

    def build_from_results(
        self,
        query: str,
        results: list[SearchResult],
    ) -> Context:
        """Variante directe depuis une liste de SearchResult."""
        mock_result = RetrievalResult(
            query=query,
            documents=results,
            top_k=len(results),
            embedding_dim=0,
            total_in_db=len(results),
        )
        return self.build(mock_result)

    def _deduplicate(
        self,
        documents: list[SearchResult],
    ) -> list[SearchResult]:
        """
        Supprime les documents quasi-identiques.
        Compare les textes par similarité de Jaccard sur les tokens.

        Args:
            documents: Documents à dédupliquer

        Returns:
            Documents uniques
        """
        if len(documents) <= 1:
            return documents

        unique = [documents[0]]

        for doc in documents[1:]:
            is_duplicate = False
            for selected in unique:
                similarity = self._jaccard_similarity(doc.text, selected.text)
                if similarity >= self._dedup_threshold:
                    is_duplicate = True
                    logger.info(
                        "Duplicate document removed",
                        id=doc.id,
                        similar_to=selected.id,
                        similarity=round(similarity, 3),
                    )
                    break
            if not is_duplicate:
                unique.append(doc)

        if len(unique) < len(documents):
            logger.info(
                "Deduplication completed",
                original=len(documents),
                unique=len(unique),
                removed=len(documents) - len(unique),
            )

        return unique

    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """Calcule la similarité de Jaccard entre deux textes."""
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())
        if not tokens1 or not tokens2:
            return 0.0
        intersection = tokens1 & tokens2
        union = tokens1 | tokens2
        return len(intersection) / len(union)

    def _format_block(
        self,
        index: int,
        doc: SearchResult,
        text: str,
    ) -> str:
        """Formate un bloc document avec indicateur de confiance."""
        confidence = "HIGH" if doc.score >= 0.7 else \
                     "MED" if doc.score >= 0.4 else "LOW"

        topic = doc.metadata.get("topic", "")
        source = doc.metadata.get("source", "")

        header = f"[Document {index}] score={doc.score:.4f} | confiance={confidence}"
        meta_parts = []
        if source:
            meta_parts.append(f"Source: {source}")
        if topic:
            meta_parts.append(f"Topic: {topic}")

        meta_line = " | ".join(meta_parts)

        if meta_line:
            return f"{header}\n{meta_line}\n\n{text}"
        return f"{header}\n\n{text}"

    def _compute_quality_score(
        self,
        sources: list[SourceReference],
    ) -> float:
        """
        Calcule un score de qualité global du contexte.

        Formule :
        - Basé sur la moyenne pondérée des scores
        - Bonus pour nombre de sources (diversité)
        - Pénalité si tous les scores sont faibles

        Returns:
            Score entre 0.0 et 1.0
        """
        if not sources:
            return 0.0

        scores = [s.score for s in sources]
        avg_score = sum(scores) / len(scores)
        best_score = max(scores)

        # Pondération : 70% meilleur score + 30% moyenne
        weighted = 0.7 * best_score + 0.3 * avg_score

        # Bonus diversité (jusqu'à +10%)
        diversity_bonus = min(len(sources) / 10, 0.1)

        return min(weighted + diversity_bonus, 1.0)

    def _compute_confidence(self, quality_score: float) -> str:
        """Convertit le score qualité en niveau de confiance."""
        if quality_score >= 0.7:
            return "high"
        elif quality_score >= 0.4:
            return "medium"
        elif quality_score > 0.0:
            return "low"
        return "none"