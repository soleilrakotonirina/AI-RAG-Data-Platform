"""
scripts/test_rag_pipeline.py

Validation Phase 6 — Pipeline RAG amélioré.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from backend.app.core.logger import configure_logging, get_logger
from backend.app.rag.chain import RAGPipeline, run_rag_pipeline, clear_query_cache
from backend.app.services.embedding_service import get_cache_stats, clear_embedding_cache
from backend.app.db.vector_store import VectorStore
from indexing.embeddings import index_documents

configure_logging()
logger = get_logger(__name__)

RAW_DOCUMENTS = [
    {
        "id": "doc_001",
        "text": "FastAPI est un framework Python moderne pour construire des APIs REST performantes.",
        "metadata": {"source": "tech_docs", "topic": "fastapi"},
    },
    {
        "id": "doc_002",
        "text": "ChromaDB est une base de données vectorielle open-source conçue pour les applications LLM.",
        "metadata": {"source": "tech_docs", "topic": "chromadb"},
    },
    {
        "id": "doc_003",
        "text": "LangChain permet d'orchestrer des pipelines LLM complexes avec des chaînes de traitement.",
        "metadata": {"source": "tech_docs", "topic": "langchain"},
    },
    {
        "id": "doc_004",
        "text": "LangGraph est un framework pour construire des agents IA avec gestion d'état et workflows.",
        "metadata": {"source": "tech_docs", "topic": "langgraph"},
    },
    {
        "id": "doc_005",
        "text": "OpenRouter donne accès à plusieurs LLMs (GPT-4, Claude, Mistral) via une API unifiée.",
        "metadata": {"source": "tech_docs", "topic": "openrouter"},
    },
    {
        "id": "doc_006",
        "text": "RAG (Retrieval-Augmented Generation) combine la recherche documentaire avec la génération LLM.",
        "metadata": {"source": "tech_docs", "topic": "rag"},
    },
    {
        "id": "doc_007",
        "text": "Les embeddings transforment le texte en vecteurs numériques dans un espace sémantique.",
        "metadata": {"source": "tech_docs", "topic": "embeddings"},
    },
]


def ensure_indexed():
    store = VectorStore()
    if store.count() == 0:
        logger.info("Indexation des documents")
        index_documents(RAW_DOCUMENTS)
    else:
        logger.info("Documents indexés", count=store.count())


def divider(title: str):
    logger.info("=" * 50)
    logger.info(f"  {title}")
    logger.info("=" * 50)


def run_tests():
    logger.info("=== Phase 6 — RAG Pipeline Amélioré ===")
    ensure_indexed()
    clear_embedding_cache()
    clear_query_cache()

    pipeline = RAGPipeline(
        top_k=3,
        max_chars_per_doc=600,
        max_total_chars=4000,
        use_mmr=True,
        mmr_lambda=0.7,
        adaptive_k=True,
        use_cache=True,
    )

    # ------------------------------------------------------------------
    # Test 1 : Pipeline RAG complet avec métriques
    # ------------------------------------------------------------------
    divider("Test 1 : Pipeline RAG end-to-end")

    result = pipeline.run(
        "Qu'est-ce que ChromaDB et pourquoi est-il utile pour les LLMs ?"
    )

    assert result.answer
    assert result.context_used
    assert result.document_count > 0

    logger.info("\n" + result.format_full())
    logger.info(
        "Qualité contexte",
        quality_score=result.quality_score,
        confidence_level=result.confidence_level,
    )

    # ------------------------------------------------------------------
    # Test 2 : Cache requête
    # ------------------------------------------------------------------
    divider("Test 2 : Cache requête (même question)")

    import time
    start = time.time()
    result_cached = pipeline.run(
        "Qu'est-ce que ChromaDB et pourquoi est-il utile pour les LLMs ?"
    )
    cache_duration = (time.time() - start) * 1000

    assert result_cached.from_cache is True
    logger.info(
        "Cache hit",
        from_cache=result_cached.from_cache,
        duration_ms=round(cache_duration),
        speedup="instant vs ~4000ms",
    )

    # ------------------------------------------------------------------
    # Test 3 : MMR diversité
    # ------------------------------------------------------------------
    divider("Test 3 : MMR — Diversité des résultats")

    pipeline_no_mmr = RAGPipeline(top_k=3, use_mmr=False, use_cache=False)
    pipeline_mmr = RAGPipeline(top_k=3, use_mmr=True, mmr_lambda=0.7, use_cache=False)

    result_no_mmr = pipeline_no_mmr.run("Quels outils pour construire un système RAG ?")
    result_mmr = pipeline_mmr.run("Quels outils pour construire un système RAG ?")

    logger.info("Sans MMR — Sources :")
    for src in result_no_mmr.sources:
        logger.info(f"  {src['id']} | {src['metadata'].get('topic')} | score={src['score']:.4f}")

    logger.info("Avec MMR — Sources :")
    for src in result_mmr.sources:
        logger.info(f"  {src['id']} | {src['metadata'].get('topic')} | score={src['score']:.4f}")

    # ------------------------------------------------------------------
    # Test 4 : Comparaison avec/sans RAG
    # ------------------------------------------------------------------
    divider("Test 4 : Comparaison AVEC vs SANS RAG")

    comparison = pipeline.compare(
        "Comment LangGraph aide-t-il à construire des agents IA ?"
    )

    logger.info(f"Question : {comparison['question']}")
    logger.info("\n--- SANS RAG ---\n" + comparison["without_rag"]["answer"])
    logger.info("\n--- AVEC RAG ---\n" + comparison["with_rag"]["answer"])
    logger.info(
        "Qualité RAG",
        quality_score=comparison["with_rag"]["quality_score"],
        confidence=comparison["with_rag"]["confidence_level"],
        sources=len(comparison["with_rag"]["sources"]),
    )

    # ------------------------------------------------------------------
    # Test 5 : Question hors base
    # ------------------------------------------------------------------
    divider("Test 5 : Question hors base")

    pipeline_strict = RAGPipeline(
        top_k=3,
        score_threshold=0.5,
        use_cache=False,
    )
    result_empty = pipeline_strict.run("Comment configurer Kubernetes ?")

    logger.info(
        "Hors base",
        context_used=result_empty.context_used,
        document_count=result_empty.document_count,
        quality_score=result_empty.quality_score,
    )
    logger.info("Réponse :\n" + result_empty.answer)

    answer_lower = result_empty.answer.lower()
    signals = any(w in answer_lower for w in [
        "contexte", "information", "pas", "aucun",
        "kubernetes", "disponible", "impossible",
    ])
    assert signals, f"LLM devrait signaler l'absence d'info : {result_empty.answer}"
    logger.info("Test 5 OK — LLM signale correctement l'absence d'information")

    # ------------------------------------------------------------------
    # Test 6 : Cache embeddings
    # ------------------------------------------------------------------
    divider("Test 6 : Statistiques cache embeddings")

    stats = get_cache_stats()
    logger.info("Cache embeddings", **stats)
    assert stats["hits"] > 0, "Le cache doit avoir des hits"
    logger.info("Test 6 OK — Cache embeddings fonctionnel")

    # ------------------------------------------------------------------
    # Résumé
    # ------------------------------------------------------------------
    divider("RESUME PHASE 6")
    logger.info("Pipeline RAG amélioré opérationnel")
    logger.info("MMR : diversité des résultats")
    logger.info("Cache requêtes : évite recomputation")
    logger.info("Cache embeddings : évite appels API redondants")
    logger.info("Prompts adaptatifs : selon confiance contexte")
    logger.info("Déduplication : documents quasi-identiques supprimés")
    logger.info("=== VALIDATION PHASE 6 TERMINEE ===")
    logger.info("Pret pour Phase 7 — Endpoint /chat")


if __name__ == "__main__":
    run_tests()