"""
backend/app/services/retriever_service.py

Couche de retrieval RAG.
Responsabilités :
- Recevoir une requête texte
- Générer l'embedding de la requête via embedding_service
- Interroger ChromaDB via VectorStore
- Retourner les documents les plus pertinents

Ce fichier ne formate PAS le contexte (rôle de context.py).
Ce fichier ne génère PAS de réponse LLM (rôle de llm_service.py Phase 5).
"""

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
    """
    Résultat structuré d'une opération de retrieval.
    Encapsule les documents récupérés et les métadonnées de la recherche.
    """
    query: str
    documents: list[SearchResult]
    top_k: int
    embedding_dim: int
    total_in_db: int

    @property
    def found(self) -> int:
        """Nombre de documents effectivement retournés."""
        return len(self.documents)

    @property
    def is_empty(self) -> bool:
        """True si aucun document n'a été trouvé."""
        return len(self.documents) == 0


# ---------------------------------------------------------------------------
# Retriever Service
# ---------------------------------------------------------------------------

class RetrieverService:
    """
    Service de retrieval sémantique.

    Orchestre :
    1. Génération de l'embedding de la requête
    2. Recherche dans ChromaDB
    3. Retour des résultats structurés

    Usage :
        retriever = RetrieverService()
        result = retriever.retrieve("Comment fonctionne ChromaDB ?")
        for doc in result.documents:
            print(doc.score, doc.text)
    """

    def __init__(self, top_k: int = None):
        """
        Args:
            top_k: Nombre de documents à récupérer.
                   Défaut : valeur dans settings (DEFAULT_TOP_K = 5)
        """
        settings = get_settings()
        self._top_k = top_k or settings.retrieval_top_k
        self._store = VectorStore()
        logger.info("RetrieverService initialized", top_k=self._top_k)

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        score_threshold: float = None,
    ) -> RetrievalResult:
        """
        Effectue un retrieval complet pour une requête donnée.

        Étapes :
        1. Génère l'embedding de la requête
        2. Interroge ChromaDB
        3. Filtre par score_threshold si fourni
        4. Retourne RetrievalResult

        Args:
            query: Question ou requête en langage naturel
            top_k: Nombre de documents à récupérer (override du défaut)
            score_threshold: Score minimal pour inclure un document (0.0 à 1.0)

        Returns:
            RetrievalResult avec les documents pertinents classés

        Raises:
            ValueError: Si la requête est vide
        """
        if not query or not query.strip():
            raise ValueError("La requête ne peut pas être vide.")

        effective_top_k = top_k or self._top_k
        settings = get_settings()
        effective_threshold = score_threshold or settings.retrieval_score_threshold

        logger.info(
            "Starting retrieval",
            query=query,
            top_k=effective_top_k,
            score_threshold=effective_threshold,
        )

        # Étape 1 : Embedding de la requête
        query_embedding = embed_text(query)

        # Étape 2 : Recherche ChromaDB
        raw_results = self._store.search(
            query_text=query,
            top_k=effective_top_k,
            query_embedding=query_embedding,
        )

        # Étape 3 : Filtrage par score
        filtered_results = [
            doc for doc in raw_results
            if doc.score >= effective_threshold
        ]

        if len(filtered_results) < len(raw_results):
            logger.info(
                "Documents filtered by score threshold",
                before=len(raw_results),
                after=len(filtered_results),
                threshold=effective_threshold,
            )

        result = RetrievalResult(
            query=query,
            documents=filtered_results,
            top_k=effective_top_k,
            embedding_dim=len(query_embedding),
            total_in_db=self._store.count(),
        )

        logger.info(
            "Retrieval completed",
            query=query,
            found=result.found,
            top_k=effective_top_k,
            total_in_db=result.total_in_db,
        )

        return result

    def retrieve_raw(self, query: str, top_k: int = None) -> list[SearchResult]:
        """
        Version simplifiée : retourne directement la liste de SearchResult
        sans filtrage par score.

        Utile pour les tests et le debug.

        Args:
            query: Requête texte
            top_k: Nombre de résultats

        Returns:
            Liste de SearchResult
        """
        effective_top_k = top_k or self._top_k
        query_embedding = embed_text(query)

        return self._store.search(
            query_text=query,
            top_k=effective_top_k,
            query_embedding=query_embedding,
        )