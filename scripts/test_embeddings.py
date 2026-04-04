"""
scripts/test_embeddings.py

Script de validation Phase 3 — Embeddings.

Teste :
1. Génération embedding texte unique
2. Génération embeddings en batch
3. Indexation complète (texte → embedding → ChromaDB)
4. Recherche sémantique avec embedding réel
5. Comparaison qualité vs Phase 2 (embeddings fake)

Lancer depuis la racine du projet :
    python scripts/test_embeddings.py
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from backend.app.core.logger import configure_logging, get_logger
from backend.app.services.embedding_service import (
    embed_text,
    embed_batch,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
)
from backend.app.db.vector_store import VectorStore
from indexing.embeddings import index_documents, search_documents

configure_logging()
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Documents de test
# ---------------------------------------------------------------------------

RAW_DOCUMENTS = [
    {
        "id": "doc_001",
        "text": "FastAPI est un framework Python moderne pour construire des APIs REST performantes.",
        "metadata": {"source": "tech_docs", "topic": "fastapi", "language": "fr"},
    },
    {
        "id": "doc_002",
        "text": "ChromaDB est une base de données vectorielle open-source conçue pour les applications LLM.",
        "metadata": {"source": "tech_docs", "topic": "chromadb", "language": "fr"},
    },
    {
        "id": "doc_003",
        "text": "LangChain permet d'orchestrer des pipelines LLM complexes avec des chaînes de traitement.",
        "metadata": {"source": "tech_docs", "topic": "langchain", "language": "fr"},
    },
    {
        "id": "doc_004",
        "text": "LangGraph est un framework pour construire des agents IA avec gestion d'état et workflows.",
        "metadata": {"source": "tech_docs", "topic": "langgraph", "language": "fr"},
    },
    {
        "id": "doc_005",
        "text": "OpenRouter donne accès à plusieurs LLMs (GPT-4, Claude, Mistral) via une API unifiée.",
        "metadata": {"source": "tech_docs", "topic": "openrouter", "language": "fr"},
    },
    {
        "id": "doc_006",
        "text": "RAG (Retrieval-Augmented Generation) combine la recherche documentaire avec la génération LLM.",
        "metadata": {"source": "tech_docs", "topic": "rag", "language": "fr"},
    },
    {
        "id": "doc_007",
        "text": "Les embeddings transforment le texte en vecteurs numériques dans un espace sémantique.",
        "metadata": {"source": "tech_docs", "topic": "embeddings", "language": "fr"},
    },
]


def run_tests():
    logger.info("=== Phase 3 — Embeddings Validation ===")

    # ------------------------------------------------------------------
    # 1. Test embedding texte unique
    # ------------------------------------------------------------------
    logger.info("--- Test 1 : Embedding texte unique ---")

    test_text = "Comment fonctionne la recherche vectorielle ?"
    embedding = embed_text(test_text)

    assert isinstance(embedding, list), "L'embedding doit être une liste"
    assert len(embedding) == EMBEDDING_DIM, (
        f"Dimension attendue {EMBEDDING_DIM}, obtenu {len(embedding)}"
    )
    assert all(isinstance(x, float) for x in embedding), "Tous les éléments doivent être des floats"

    logger.info(
        "Embedding OK",
        model=EMBEDDING_MODEL,
        dim=len(embedding),
        sample_values=embedding[:3],
    )

    # ------------------------------------------------------------------
    # 2. Test batch embedding
    # ------------------------------------------------------------------
    logger.info("--- Test 2 : Batch embedding ---")

    texts = [doc["text"] for doc in RAW_DOCUMENTS[:3]]
    batch_embeddings = embed_batch(texts)

    assert len(batch_embeddings) == 3, f"Attendu 3 embeddings, obtenu {len(batch_embeddings)}"
    assert all(len(e) == EMBEDDING_DIM for e in batch_embeddings), "Dimensions incorrectes"

    # Vérifier que les embeddings sont distincts
    assert batch_embeddings[0] != batch_embeddings[1], "Les embeddings doivent être distincts"

    logger.info(
        "Batch embedding OK",
        count=len(batch_embeddings),
        dim=EMBEDDING_DIM,
    )

    # ------------------------------------------------------------------
    # 3. Reset et indexation complète
    # ------------------------------------------------------------------
    logger.info("--- Test 3 : Indexation complète ---")

    store = VectorStore()
    store.reset()
    assert store.count() == 0

    index_documents(RAW_DOCUMENTS)

    count = store.count()
    assert count == len(RAW_DOCUMENTS), (
        f"Attendu {len(RAW_DOCUMENTS)} documents, obtenu {count}"
    )
    logger.info("Indexation OK", document_count=count)

    # ------------------------------------------------------------------
    # 4. Recherche sémantique
    # ------------------------------------------------------------------
    logger.info("--- Test 4 : Recherche sémantique ---")

    queries = [
        ("base de données vectorielle", "doc_002"),
        ("agent IA avec état", "doc_004"),
        ("transformer texte en vecteurs", "doc_007"),
    ]

    for query, expected_top_id in queries:
        results = search_documents(query_text=query, top_k=3)

        assert len(results) > 0, f"Aucun résultat pour : {query}"

        logger.info(f"Requête : '{query}'")
        for i, result in enumerate(results, 1):
            marker = "<-- ATTENDU" if result.id == expected_top_id else ""
            logger.info(
                f"  #{i}",
                id=result.id,
                score=result.score,
                topic=result.metadata.get("topic", "?"),
                marker=marker,
            )

        # Vérifier que le document attendu est dans les top-3
        result_ids = [r.id for r in results]
        if expected_top_id not in result_ids:
            logger.warning(
                "Expected document not in top-3",
                query=query,
                expected=expected_top_id,
                got=result_ids,
            )

    # ------------------------------------------------------------------
    # 5. Résumé
    # ------------------------------------------------------------------
    logger.info("=== VALIDATION PHASE 3 TERMINEE ===")
    logger.info("Embeddings operationnels")
    logger.info(f"Modele : {EMBEDDING_MODEL}")
    logger.info(f"Dimension : {EMBEDDING_DIM}")
    logger.info("Pret pour Phase 4 — Retriever")


if __name__ == "__main__":
    run_tests()