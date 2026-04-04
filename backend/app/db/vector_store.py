"""
backend/app/db/vector_store.py

Couche d'accès aux données vectorielles.
Responsabilités :
- Ajouter des documents avec embeddings dans ChromaDB
- Effectuer des recherches par similarité (top-k)
- Supprimer des documents
- Retourner des résultats typés et structurés

Ce fichier utilise ChromaDBClient mais ne gère pas la connexion.
"""

from dataclasses import dataclass, field
from typing import Optional
import random

from backend.app.db.chroma_client import ChromaDBClient
from backend.app.core.logger import get_logger

logger = get_logger(__name__)

# Dimension des embeddings fictifs (pour les tests Phase 2)
# En Phase 3, cette valeur sera remplacée par la dimension réelle du modèle
FAKE_EMBEDDING_DIM = 384


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
    score: float  # 1 - distance (similarité cosine normalisée)


# ---------------------------------------------------------------------------
# Embeddings fictifs (Phase 2 uniquement)
# ---------------------------------------------------------------------------

def generate_fake_embedding(text: str, dim: int = FAKE_EMBEDDING_DIM) -> list[float]:
    """
    Génère un vecteur aléatoire normalisé simulant un embedding.

    IMPORTANT : Ce générateur est uniquement destiné à valider
    la pipeline ChromaDB en Phase 2. Il sera remplacé en Phase 3
    par les vrais embeddings du modèle (via OpenRouter).

    Args:
        text: Texte source (non utilisé, présent pour cohérence d'interface)
        dim: Dimension du vecteur

    Returns:
        Vecteur de floats de dimension `dim`, normalisé
    """
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

    Usage :
        store = VectorStore()
        store.add_documents([Document(id="1", text="hello world")])
        results = store.search("hello", top_k=3)
    """

    def __init__(self):
        self._db = ChromaDBClient()
        self._collection = self._db.collection
        logger.info("VectorStore initialized")

    # -----------------------------------------------------------------------
    # Ajout de documents
    # -----------------------------------------------------------------------

    def add_documents(self, documents: list[Document]) -> None:
        """
        Ajoute une liste de documents dans ChromaDB.

        Si un document ne contient pas d'embedding, un embedding fictif
        est généré automatiquement (Phase 2 uniquement).

        Args:
            documents: Liste de Document à indexer

        Raises:
            ValueError: Si la liste est vide
            Exception: Si ChromaDB rejette l'insertion
        """
        if not documents:
            raise ValueError("La liste de documents est vide.")

        ids = []
        texts = []
        embeddings = []
        metadatas = []

        for doc in documents:
            embedding = doc.embedding or generate_fake_embedding(doc.text)

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

    # -----------------------------------------------------------------------
    # Recherche vectorielle
    # -----------------------------------------------------------------------

    def search(
        self,
        query_text: str,
        top_k: int = 5,
        query_embedding: Optional[list[float]] = None,
    ) -> list[SearchResult]:
        """
        Effectue une recherche par similarité dans ChromaDB.

        En Phase 2 : utilise un embedding fictif pour la requête.
        En Phase 3 : query_embedding sera fourni par embedding_service.py.

        Args:
            query_text: Texte de la requête (pour logs)
            top_k: Nombre de résultats à retourner
            query_embedding: Vecteur de la requête (optionnel en Phase 2)

        Returns:
            Liste de SearchResult triés par pertinence décroissante
        """
        embedding = query_embedding or generate_fake_embedding(query_text)

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

    # -----------------------------------------------------------------------
    # Suppression
    # -----------------------------------------------------------------------

    def delete_document(self, doc_id: str) -> None:
        """
        Supprime un document par son identifiant.

        Args:
            doc_id: Identifiant du document à supprimer
        """
        self._collection.delete(ids=[doc_id])
        logger.info("Document deleted", id=doc_id)

    # -----------------------------------------------------------------------
    # Utilitaires
    # -----------------------------------------------------------------------

    def count(self) -> int:
        """Retourne le nombre de documents dans la collection."""
        return self._collection.count()

    def reset(self) -> None:
        """
        Vide complètement la collection. Utile pour les tests.
        Ne supprime pas la collection — la recrée vide.
        """
        self._db.reset_collection()
        self._collection = self._db.collection
        logger.warning("VectorStore reset — collection cleared")