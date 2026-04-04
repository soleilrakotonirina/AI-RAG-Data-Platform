"""
scripts/test_chromadb.py

Script de validation Phase 2 — ChromaDB.

Teste :
1. Connexion ChromaDB
2. Insertion de documents
3. Recherche par similarité
4. Suppression d'un document
5. Reset collection

Lancer depuis la racine du projet :
    python scripts/test_chromadb.py
"""

import sys
from pathlib import Path

# Ajout racine projet au PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from backend.app.core.logger import configure_logging, get_logger
from backend.app.db.vector_store import VectorStore, Document

configure_logging()
logger = get_logger(__name__)


def run_tests():
    logger.info("=== Phase 2 — ChromaDB Validation ===")

    # ------------------------------------------------------------------
    # 1. Initialisation
    # ------------------------------------------------------------------
    logger.info("--- Test 1 : Initialisation VectorStore ---")
    store = VectorStore()
    logger.info("VectorStore OK", document_count=store.count())

    # ------------------------------------------------------------------
    # 2. Reset (état propre pour le test)
    # ------------------------------------------------------------------
    logger.info("--- Reset collection ---")
    store.reset()
    assert store.count() == 0, "La collection devrait être vide après reset"
    logger.info("Collection vide confirmée")

    # ------------------------------------------------------------------
    # 3. Insertion documents
    # ------------------------------------------------------------------
    logger.info("--- Test 2 : Insertion documents ---")

    documents = [
        Document(
            id="doc_001",
            text="FastAPI est un framework Python moderne pour construire des APIs REST.",
            metadata={"source": "tech_docs", "topic": "fastapi"},
        ),
        Document(
            id="doc_002",
            text="ChromaDB est une base de données vectorielle open-source.",
            metadata={"source": "tech_docs", "topic": "chromadb"},
        ),
        Document(
            id="doc_003",
            text="LangChain permet d'orchestrer des pipelines LLM complexes.",
            metadata={"source": "tech_docs", "topic": "langchain"},
        ),
        Document(
            id="doc_004",
            text="LangGraph est un framework pour construire des agents IA avec état.",
            metadata={"source": "tech_docs", "topic": "langgraph"},
        ),
        Document(
            id="doc_005",
            text="OpenRouter donne accès à plusieurs LLMs via une API unifiée.",
            metadata={"source": "tech_docs", "topic": "openrouter"},
        ),
    ]

    store.add_documents(documents)
    count = store.count()
    assert count == 5, f"Attendu 5 documents, obtenu {count}"
    logger.info("Insertion OK", document_count=count)

    # ------------------------------------------------------------------
    # 4. Recherche
    # ------------------------------------------------------------------
    logger.info("--- Test 3 : Recherche vectorielle ---")

    results = store.search(query_text="base de données vectorielle", top_k=3)

    assert len(results) > 0, "La recherche devrait retourner des résultats"

    logger.info("Résultats de recherche :")
    for i, result in enumerate(results, 1):
        logger.info(
            f"  Résultat {i}",
            id=result.id,
            score=result.score,
            distance=result.distance,
            text_preview=result.text[:60],
            metadata=result.metadata,
        )

    # ------------------------------------------------------------------
    # 5. Suppression
    # ------------------------------------------------------------------
    logger.info("--- Test 4 : Suppression document ---")
    store.delete_document("doc_001")
    count_after = store.count()
    assert count_after == 4, f"Attendu 4 documents après suppression, obtenu {count_after}"
    logger.info("Suppression OK", document_count=count_after)

    # ------------------------------------------------------------------
    # 6. Résumé
    # ------------------------------------------------------------------
    logger.info("=== VALIDATION PHASE 2 TERMINEE ===")
    logger.info("Tous les tests sont passes avec succes")
    logger.info("ChromaDB est operationnelle et prete pour la Phase 3")


if __name__ == "__main__":
    run_tests()