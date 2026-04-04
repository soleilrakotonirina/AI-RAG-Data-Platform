"""
backend/app/db/vector_store.py

Couche d'accès aux données vectorielles.
Responsabilités :
- Ajouter des documents avec embeddings dans ChromaDB
- Effectuer des recherches par similarité (top-k)
- Supprimer des documents
- Retourner des résultats typés et structurés
"""

from dataclasses import dataclass, field
from typing import Optional
import random

from backend.app.db.chroma_client import ChromaDBClient
from backend.app.core.logger import get_logger

logger = get_logger(__name__)

FAKE_EMBEDDING_DIM = 1536  # aligné sur text-embedding-3-small


# ---------------------------------------------------------------------------
# Types de données
# ---------------------------------------------------------------------------

@dataclass
class Document:
    """Représente un document à indexer dans ChromaDB."""
    id: str
    text: str
    metadata: dict = field(default_factory=dict)
    embedding: Optional[list[float]] = None


@dataclass
class SearchResult:
    """Représente un résultat de recherche retourné par ChromaDB."""
    id: str
    text: str
    metadata: dict
    distance: float
    score: float


# ---------------------------------------------------------------------------
# Embeddings fictifs — fallback uniquement
# ---------------------------------------------------------------------------

def _generate_fake_embedding(text: str, dim: int = FAKE_EMBEDDING_DIM) -> list[float]:
    """
    Fallback interne : vecteur aléatoire normalisé.
    Utilisé UNIQUEMENT si aucun embedding n'est fourni.
    En production, embedding_service.py fournit toujours un embedding.
    """
    logger.warning(
        "Using fake random embedding — DO NOT use in production",
        text_preview=text[:40],
    )
    vector = [random.gauss(0, 1) for _ in range(dim)]
    norm = sum(x ** 2 for x in vector) ** 0.5
    return [x / norm for x in vector]


# ---------------------------------------------------------------------------
# VectorStore
# ---------------------------------------------------------------------------

class VectorStore:
    """
    Interface d'accès à ChromaDB pour les opérations de stockage
    et de recherche vectorielle.
    """

    def __init__(self):
        self._db = ChromaDBClient()
        self._collection = self._db.collection
        logger.info("VectorStore initialized")

    def add_documents(self, documents: list[Document]) -> None:
        """
        Ajoute une liste de documents dans ChromaDB.
        Si embedding absent → fallback fake (avec warning).

        Args:
            documents: Liste de Document à indexer

        Raises:
            ValueError: Si la liste est vide
        """
        if not documents:
            raise ValueError("La liste de documents est vide.")

        ids, texts, embeddings, metadatas = [], [], [], []

        for doc in documents:
            embedding = doc.embedding or _generate_fake_embedding(doc.text)
            ids.append(doc.id)
            texts.append(doc.text)
            embeddings.append(embedding)
            metadatas.append(doc.metadata or {})

        self._collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        logger.info(
            "Documents added to ChromaDB",
            count=len(documents),
            ids=ids,
        )

    def search(
        self,
        query_text: str,
        top_k: int = 5,
        query_embedding: Optional[list[float]] = None,
    ) -> list[SearchResult]:
        """
        Recherche par similarité dans ChromaDB.

        Args:
            query_text: Texte de la requête (pour logs)
            top_k: Nombre de résultats
            query_embedding: Vecteur réel fourni par embedding_service

        Returns:
            Liste de SearchResult triés par pertinence
        """
        embedding = query_embedding or _generate_fake_embedding(query_text)

        raw = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, self._collection.count() or 1),
            include=["documents", "metadatas", "distances"],
        )

        results = []
        ids = raw.get("ids", [[]])[0]
        documents = raw.get("documents", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]

        for doc_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
            results.append(SearchResult(
                id=doc_id,
                text=text,
                metadata=metadata or {},
                distance=round(distance, 6),
                score=round(1 - distance, 6),
            ))

        logger.info(
            "Search completed",
            query=query_text,
            top_k=top_k,
            results_found=len(results),
        )

        return results

    def delete_document(self, doc_id: str) -> None:
        """Supprime un document par son identifiant."""
        self._collection.delete(ids=[doc_id])
        logger.info("Document deleted", id=doc_id)

    def count(self) -> int:
        """Retourne le nombre de documents dans la collection."""
        return self._collection.count()

    def reset(self) -> None:
        """Vide complètement la collection."""
        self._db.reset_collection()
        self._collection = self._db.collection
        logger.warning("VectorStore reset — collection cleared")