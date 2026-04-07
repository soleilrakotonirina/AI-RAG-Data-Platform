"""
scripts/test_llm.py

Script de validation Phase 5 — LLM.

Teste :
1. Initialisation OpenRouterClient
2. Génération simple (sans contexte)
3. Pipeline RAG complet (retrieval + contexte + LLM)
4. Comparaison réponse avec/sans RAG
5. Gestion contexte vide

Prérequis :
- OPENROUTER_API_KEY configurée dans .env
- Documents indexés dans ChromaDB (Phase 3)

Lancer depuis la racine du projet :
    python scripts/test_llm.py
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from backend.app.core.logger import configure_logging, get_logger
from backend.app.services.openrouter_client import OpenRouterClient
from backend.app.services.llm_service import LLMService
from backend.app.services.retriever_service import RetrieverService
from backend.app.rag.context import ContextBuilder
from backend.app.db.vector_store import VectorStore
from indexing.embeddings import index_documents

configure_logging()
logger = get_logger(__name__)

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
    store = VectorStore()
    if store.count() == 0:
        logger.info("Collection vide — indexation des documents de test")
        index_documents(RAW_DOCUMENTS)
    else:
        logger.info("Documents déjà indexés", count=store.count())


def separator(title: str):
    logger.info(f"--- {title} ---")


def run_tests():
    logger.info("=== Phase 5 — LLM Validation ===")

    ensure_documents_indexed()

    # ------------------------------------------------------------------
    # Test 1 : Initialisation client
    # ------------------------------------------------------------------
    separator("Test 1 : Initialisation OpenRouterClient")

    client = OpenRouterClient()
    logger.info("OpenRouterClient OK", model=client._model)

    # ------------------------------------------------------------------
    # Test 2 : Génération simple sans contexte
    # ------------------------------------------------------------------
    separator("Test 2 : Génération simple sans contexte RAG")

    answer_raw = client.generate_simple(
        prompt="En une phrase, qu'est-ce que ChromaDB ?",
        temperature=0.1,
        max_tokens=150,
    )

    assert isinstance(answer_raw, str) and len(answer_raw) > 0
    logger.info("Réponse LLM brute :\n" + answer_raw)

    # ------------------------------------------------------------------
    # Test 3 : Pipeline RAG complet
    # ------------------------------------------------------------------
    separator("Test 3 : Pipeline RAG complet")

    retriever = RetrieverService(top_k=3)
    builder = ContextBuilder(max_chars_per_doc=500)
    llm = LLMService()

    question = "Qu'est-ce que ChromaDB et pourquoi est-il utile pour les LLMs ?"
    # Sur ChromaDB
    # question = "Comment ChromaDB stocke-t-il les données ?"
    # question = "Pourquoi utiliser ChromaDB plutôt qu'une base classique ?"

    # Sur les embeddings
    # question = "Qu'est-ce qu'un embedding et à quoi ça sert ?"
    # question = "Comment les embeddings permettent-ils la recherche sémantique ?"

    # Sur RAG
    # question = "Qu'est-ce que le RAG et quels sont ses avantages ?"
    # question = "Comment fonctionne le Retrieval-Augmented Generation ?"

    # Sur LangChain
    # question = "Quel est le rôle de LangChain dans un pipeline LLM ?"

    # Sur LangGraph
    # question = "Quelle est la différence entre LangChain et LangGraph ?"
    # question = "Pourquoi utiliser LangGraph pour construire un agent IA ?"

    # Sur OpenRouter
    # question = "Comment OpenRouter permet-il d'accéder à plusieurs LLMs ?"

    # Sur FastAPI
    # question = "Pourquoi choisir FastAPI pour construire une API Python ?"

    # Questions croisées (plusieurs documents impliqués)
    # question = "Comment combiner ChromaDB et LangChain pour un système RAG ?"
    # question = "Quels outils utiliser pour construire un agent IA complet ?"
    # question = "Quelle est la relation entre embeddings et base vectorielle ?"
    
    # Retrieval
    retrieval_result = retriever.retrieve(question)
    logger.info(
        "Retrieval OK",
        found=retrieval_result.found,
        top_doc=retrieval_result.documents[0].id if retrieval_result.documents else "none",
    )

    # Contexte
    context = builder.build(retrieval_result)
    logger.info(
        "Context OK",
        document_count=context.document_count,
        context_length=len(context.text),
    )

    # Génération
    response = llm.generate(
        question=question,
        context=context,
    )

    assert isinstance(response.answer, str)
    assert len(response.answer) > 0
    assert response.context_used is True
    assert response.document_count > 0

    logger.info("RAG Response :")
    logger.info("\n" + response.format_full())

    # ------------------------------------------------------------------
    # Test 4 : Comparaison avec/sans RAG
    # ------------------------------------------------------------------
    separator("Test 4 : Comparaison avec/sans RAG")

    question_comparison = "Quel framework Python utiliser pour construire un agent IA ?"

    # Sans RAG
    answer_no_rag = llm.generate_without_context(question=question_comparison)
    logger.info("Réponse SANS RAG :\n" + answer_no_rag)

    # Avec RAG
    retrieval_result2 = retriever.retrieve(question_comparison)
    context2 = builder.build(retrieval_result2)
    response_rag = llm.generate(question=question_comparison, context=context2)
    logger.info("Réponse AVEC RAG :\n" + response_rag.answer)

    logger.info(
        "Comparaison",
        sans_rag_length=len(answer_no_rag),
        avec_rag_length=len(response_rag.answer),
        sources_used=len(response_rag.sources),
    )

    # ------------------------------------------------------------------
    # Test 5 : Contexte vide
    # ------------------------------------------------------------------
    separator("Test 5 : Gestion contexte vide")

    from backend.app.rag.context import Context
    empty_context = Context(
        query="question sans contexte",
        text="",
        sources=[],
        document_count=0,
    )

    response_empty = llm.generate(
        question="Qu'est-ce que Kubernetes ?",
        context=empty_context,
    )

    assert response_empty.context_used is False
    logger.info("Réponse contexte vide :\n" + response_empty.answer)

    # ------------------------------------------------------------------
    # Résumé
    # ------------------------------------------------------------------
    logger.info("=== VALIDATION PHASE 5 TERMINEE ===")
    logger.info("OpenRouterClient operationnel")
    logger.info("LLMService operationnel")
    logger.info("Pipeline RAG complet : question → retrieval → contexte → LLM → réponse")
    logger.info("Pret pour Phase 6 — RAG Pipeline complet")


if __name__ == "__main__":
    run_tests()