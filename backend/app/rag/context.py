"""
backend/app/rag/context.py

Couche de construction de contexte RAG.
Responsabilités :
- Transformer les SearchResult en texte structuré
- Préparer le contexte pour injection dans le prompt LLM (Phase 5)
- Gérer la troncature si le contexte est trop long
- Exposer les métadonnées des sources

Ce fichier ne fait PAS de retrieval (rôle de retriever_service.py).
Ce fichier ne génère PAS de réponse LLM (rôle de llm_service.py).
"""

from dataclasses import dataclass, field

from backend.app.db.vector_store import SearchResult
from backend.app.services.retriever_service import RetrievalResult
from backend.app.core.logger import get_logger

logger = get_logger(__name__)

# Nombre maximum de caractères par document dans le contexte
MAX_CHARS_PER_DOC = 1000

# Séparateur entre les documents dans le contexte
DOCUMENT_SEPARATOR = "\n\n---\n\n"


# ---------------------------------------------------------------------------
# Types de données
# ---------------------------------------------------------------------------

@dataclass
class SourceReference:
    """Référence à un document source utilisé dans le contexte."""
    id: str
    score: float
    metadata: dict


@dataclass
class Context:
    """
    Contexte RAG complet prêt pour injection dans un prompt LLM.

    Attributs :
        query           : Question originale de l'utilisateur
        text            : Contexte formaté (concaténation des documents)
        sources         : Liste des sources utilisées avec scores
        document_count  : Nombre de documents inclus
        truncated       : True si des documents ont été tronqués
    """
    query: str
    text: str
    sources: list[SourceReference]
    document_count: int
    truncated: bool = False

    @property
    def is_empty(self) -> bool:
        """True si aucun document n'a été inclus dans le contexte."""
        return self.document_count == 0

    def to_prompt_block(self) -> str:
        """
        Formate le contexte pour injection directe dans un prompt LLM.

        Format :
            <context>
            [Document 1]
            ...texte...

            ---

            [Document 2]
            ...texte...
            </context>

        Returns:
            Bloc de texte prêt pour le prompt
        """
        if self.is_empty:
            return "<context>\nAucun document pertinent trouvé.\n</context>"

        return f"<context>\n{self.text}\n</context>"

    def format_sources(self) -> str:
        """
        Formate la liste des sources pour affichage ou logging.

        Returns:
            Chaîne listant les sources avec leurs scores
        """
        if not self.sources:
            return "Aucune source."

        lines = []
        for i, src in enumerate(self.sources, 1):
            topic = src.metadata.get("topic", "inconnu")
            source = src.metadata.get("source", "inconnu")
            lines.append(
                f"  [{i}] id={src.id} | score={src.score:.4f} | "
                f"topic={topic} | source={source}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Context Builder
# ---------------------------------------------------------------------------

class ContextBuilder:
    """
    Construit un objet Context à partir des résultats de retrieval.

    Usage :
        builder = ContextBuilder()
        context = builder.build(retrieval_result)
        prompt_block = context.to_prompt_block()
    """

    def __init__(
        self,
        max_chars_per_doc: int = MAX_CHARS_PER_DOC,
        separator: str = DOCUMENT_SEPARATOR,
    ):
        """
        Args:
            max_chars_per_doc: Limite de caractères par document
            separator: Séparateur entre les blocs de documents
        """
        self._max_chars = max_chars_per_doc
        self._separator = separator
        logger.info(
            "ContextBuilder initialized",
            max_chars_per_doc=max_chars_per_doc,
        )

    def build(self, retrieval_result: RetrievalResult) -> Context:
        """
        Construit un Context à partir d'un RetrievalResult.

        Args:
            retrieval_result: Résultat du RetrieverService

        Returns:
            Context prêt pour injection dans le prompt LLM
        """
        if retrieval_result.is_empty:
            logger.warning(
                "No documents found for query",
                query=retrieval_result.query,
            )
            return Context(
                query=retrieval_result.query,
                text="",
                sources=[],
                document_count=0,
                truncated=False,
            )

        blocks = []
        sources = []
        truncated = False

        for i, doc in enumerate(retrieval_result.documents, 1):
            # Troncature si le texte est trop long
            text = doc.text
            if len(text) > self._max_chars:
                text = text[:self._max_chars] + "..."
                truncated = True
                logger.warning(
                    "Document truncated",
                    id=doc.id,
                    original_len=len(doc.text),
                    max_chars=self._max_chars,
                )

            # Bloc formaté pour ce document
            block = self._format_document_block(
                index=i,
                doc_id=doc.id,
                text=text,
                score=doc.score,
                metadata=doc.metadata,
            )
            blocks.append(block)

            sources.append(SourceReference(
                id=doc.id,
                score=doc.score,
                metadata=doc.metadata,
            ))

        full_context = self._separator.join(blocks)

        context = Context(
            query=retrieval_result.query,
            text=full_context,
            sources=sources,
            document_count=len(blocks),
            truncated=truncated,
        )

        logger.info(
            "Context built",
            query=retrieval_result.query,
            document_count=context.document_count,
            context_length=len(full_context),
            truncated=truncated,
        )

        return context

    def build_from_results(self, query: str, results: list[SearchResult]) -> Context:
        """
        Construit un Context directement depuis une liste de SearchResult.
        Variante de build() pour usage sans RetrievalResult complet.

        Args:
            query: Question originale
            results: Liste de SearchResult

        Returns:
            Context prêt pour injection
        """
        from backend.app.services.retriever_service import RetrievalResult

        mock_result = RetrievalResult(
            query=query,
            documents=results,
            top_k=len(results),
            embedding_dim=0,
            total_in_db=len(results),
        )
        return self.build(mock_result)

    def _format_document_block(
        self,
        index: int,
        doc_id: str,
        text: str,
        score: float,
        metadata: dict,
    ) -> str:
        """
        Formate un document individuel pour le contexte.

        Format :
            [Document 1] (score: 0.8234)
            Source: tech_docs | Topic: chromadb

            ChromaDB est une base de données vectorielle...

        Args:
            index: Numéro du document (1-based)
            doc_id: Identifiant du document
            text: Contenu textuel
            score: Score de similarité
            metadata: Métadonnées du document

        Returns:
            Bloc formaté prêt pour concaténation
        """
        topic = metadata.get("topic", "")
        source = metadata.get("source", "")

        header = f"[Document {index}] (score: {score:.4f})"

        meta_parts = []
        if source:
            meta_parts.append(f"Source: {source}")
        if topic:
            meta_parts.append(f"Topic: {topic}")
        meta_line = " | ".join(meta_parts) if meta_parts else ""

        if meta_line:
            return f"{header}\n{meta_line}\n\n{text}"
        return f"{header}\n\n{text}"