"""
scripts/test_retriever.py

Script de validation Phase 4 — Retriever.

Prérequis :
- ChromaDB initialisée avec documents (Phase 2)
- Embeddings fonctionnels (Phase 3)
- La collection rag_documents doit contenir des données

Si la collection est vide, le script indexe d'abord les documents de test.

Lancer depuis la racine du projet :
    python scripts/test_retriever.py
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from backend.app.core.logger import configure_logging, get_logger
from backend.app.services.retriever_service import RetrieverService
from backend.app.rag.context import ContextBuilder
from backend.app.db.vector_store import VectorStore
from indexing.embeddings import index_documents

configure_logging()
logger = get_logger(__name__)

# Documents de référence — mêmes que Phase 3
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


def ensure_documents_indexed():
    """Vérifie que des documents sont indexés, sinon les indexe."""
    store = VectorStore()
    count = store.count()
    if count == 0:
        logger.info("Collection vide — indexation des documents de test")
        index_documents(RAW_DOCUMENTS)
    else:
        logger.info("Documents déjà indexés", count=count)


def run_tests():
    logger.info("=== Phase 4 — Retriever Validation ===")

    # ------------------------------------------------------------------
    # Prérequis : s'assurer que des documents sont indexés
    # ------------------------------------------------------------------
    ensure_documents_indexed()

    retriever = RetrieverService(top_k=3)
    builder = ContextBuilder(max_chars_per_doc=500)

    # ------------------------------------------------------------------
    # Test 1 : Retrieval simple
    # ------------------------------------------------------------------
    logger.info("--- Test 1 : Retrieval simple ---")

    result = retriever.retrieve("Comment fonctionne une base de données vectorielle ?")

    assert not result.is_empty, "Le retrieval ne doit pas être vide"
    assert result.found <= result.top_k, "Ne doit pas dépasser top_k"

    logger.info(
        "Retrieval OK",
        query=result.query,
        found=result.found,
        top_k=result.top_k,
        total_in_db=result.total_in_db,
    )

    for i, doc in enumerate(result.documents, 1):
        logger.info(
            f"  Document {i}",
            id=doc.id,
            score=doc.score,
            topic=doc.metadata.get("topic"),
            text_preview=doc.text[:60],
        )

    # ------------------------------------------------------------------
    # Test 2 : Construction du contexte
    # ------------------------------------------------------------------
    logger.info("--- Test 2 : Construction du contexte ---")

    context = builder.build(result)

    assert not context.is_empty, "Le contexte ne doit pas être vide"
    assert context.document_count == result.found
    assert len(context.text) > 0

    logger.info(
        "Context OK",
        document_count=context.document_count,
        context_length=len(context.text),
        truncated=context.truncated,
    )

    # ------------------------------------------------------------------
    # Test 3 : Format prompt block
    # ------------------------------------------------------------------
    logger.info("--- Test 3 : Format prompt block ---")

    prompt_block = context.to_prompt_block()
    assert prompt_block.startswith("<context>")
    assert prompt_block.endswith("</context>")

    logger.info("Prompt block généré :")
    logger.info("\n" + prompt_block)

    # ------------------------------------------------------------------
    # Test 4 : Sources
    # ------------------------------------------------------------------
    logger.info("--- Test 4 : Sources ---")

    sources_str = context.format_sources()
    logger.info("Sources :\n" + sources_str)

    assert len(context.sources) == context.document_count

    # ------------------------------------------------------------------
    # Test 5 : Requêtes variées
    # ------------------------------------------------------------------
    logger.info("--- Test 5 : Requêtes variées ---")

    queries = [
        "Quel framework utiliser pour construire un agent IA ?",
        "Comment accéder à plusieurs LLMs avec une seule API ?",
        "Qu'est-ce que le RAG et comment ça fonctionne ?",
        "Comment générer des embeddings pour mes documents ?",
    ]

    for query in queries:
        result = retriever.retrieve(query, top_k=2)
        context = builder.build(result)

        logger.info(
            f"Query: '{query}'",
            top_doc_id=result.documents[0].id if result.documents else "none",
            top_score=result.documents[0].score if result.documents else 0,
            top_topic=result.documents[0].metadata.get("topic") if result.documents else "none",
            context_length=len(context.text),
        )

    # ------------------------------------------------------------------
    # Test 6 : Filtre par score
    # ------------------------------------------------------------------
    logger.info("--- Test 6 : Filtre par score threshold ---")

    result_filtered = retriever.retrieve(
        query="architecture système RAG complet",
        top_k=5,
        score_threshold=0.5,
    )

    logger.info(
        "Résultats après filtrage score >= 0.5",
        found=result_filtered.found,
    )

    # ------------------------------------------------------------------
    # Résumé
    # ------------------------------------------------------------------
    logger.info("=== VALIDATION PHASE 4 TERMINEE ===")
    logger.info("Retriever operationnel")
    logger.info("Context builder operationnel")
    logger.info("Pret pour Phase 5 — LLM")


if __name__ == "__main__":
    run_tests()