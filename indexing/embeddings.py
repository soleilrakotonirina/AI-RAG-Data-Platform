"""
indexing/embeddings.py

Pipeline d'embedding offline.
Responsabilités :
- Prendre une liste de documents bruts (id + texte + metadata)
- Générer les embeddings via embedding_service
- Retourner des objets Document prêts pour ChromaDB
- Insérer dans ChromaDB via VectorStore

Utilisé :
- Directement via script (Phase 3)
- Via Dagster (Phase 12)
"""

import sys
from pathlib import Path

# Résolution des imports depuis la racine du projet
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from backend.app.services.embedding_service import embed_batch
from backend.app.db.vector_store import VectorStore, Document
from backend.app.core.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Types d'entrée
# ---------------------------------------------------------------------------

def build_documents_with_embeddings(
    raw_documents: list[dict],
) -> list[Document]:
    """
    Transforme une liste de documents bruts en objets Document
    avec embeddings générés.

    Format attendu pour chaque document brut :
        {
            "id": "doc_001",
            "text": "Contenu du document...",
            "metadata": {"source": "...", "topic": "..."}  # optionnel
        }

    Args:
        raw_documents: Liste de dicts avec id, text, metadata

    Returns:
        Liste de Document avec embeddings réels

    Raises:
        ValueError: Si un document est mal formé
    """
    # Validation
    for i, doc in enumerate(raw_documents):
        if "id" not in doc:
            raise ValueError(f"Document index {i} manque le champ 'id'")
        if "text" not in doc:
            raise ValueError(f"Document index {i} manque le champ 'text'")
        if not doc["text"].strip():
            raise ValueError(f"Document '{doc['id']}' a un texte vide")

    texts = [doc["text"] for doc in raw_documents]

    logger.info(
        "Generating embeddings for documents",
        document_count=len(texts),
    )

    embeddings = embed_batch(texts)

    documents = []
    for raw_doc, embedding in zip(raw_documents, embeddings):
        documents.append(Document(
            id=raw_doc["id"],
            text=raw_doc["text"],
            metadata=raw_doc.get("metadata", {}),
            embedding=embedding,
        ))

    logger.info(
        "Documents with embeddings ready",
        count=len(documents),
        embedding_dim=len(embeddings[0]) if embeddings else 0,
    )

    return documents


def index_documents(raw_documents: list[dict]) -> None:
    """
    Pipeline complet : documents bruts → embeddings → ChromaDB.

    Étapes :
    1. Génération des embeddings
    2. Construction des objets Document
    3. Insertion dans ChromaDB via VectorStore

    Args:
        raw_documents: Liste de dicts avec id, text, metadata
    """
    logger.info("Starting indexing pipeline", document_count=len(raw_documents))

    documents = build_documents_with_embeddings(raw_documents)

    store = VectorStore()
    store.add_documents(documents)

    logger.info(
        "Indexing pipeline completed",
        indexed=len(documents),
        total_in_db=store.count(),
    )


def search_documents(
    query_text: str,
    top_k: int = 5,
) -> list:
    """
    Recherche sémantique complète :
    1. Génère l'embedding de la requête
    2. Interroge ChromaDB
    3. Retourne les résultats

    Args:
        query_text: Question ou requête en langage naturel
        top_k: Nombre de résultats à retourner

    Returns:
        Liste de SearchResult
    """
    from backend.app.services.embedding_service import embed_text

    logger.info("Semantic search", query=query_text, top_k=top_k)

    query_embedding = embed_text(query_text)

    store = VectorStore()
    results = store.search(
        query_text=query_text,
        top_k=top_k,
        query_embedding=query_embedding,
    )

    return results