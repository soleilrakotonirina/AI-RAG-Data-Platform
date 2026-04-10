"""
scripts/test_agent.py

Validation Phase 9 — Agent IA LangGraph.

Tests :
1. Question générale → pas de retrieval
2. Question documentaire → retrieval activé
3. Vérification flux conditionnel
4. Vérification étapes exécutées
5. Cohérence des réponses

Prérequis :
- Documents indexés (python scripts/ingest_documents.py)
- OPENROUTER_API_KEY configurée

Lancer :
    python scripts/test_agent.py
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from backend.app.core.logger import configure_logging, get_logger
from backend.app.agents.agent import run_agent
from backend.app.db.vector_store import VectorStore

configure_logging()
logger = get_logger(__name__)


def check_prerequisites():
    store = VectorStore()
    count = store.count()
    if count == 0:
        logger.error("ChromaDB vide — lancer : python scripts/ingest_documents.py --reset")
        sys.exit(1)
    logger.info("Documents dans ChromaDB", count=count)


def divider(title: str):
    logger.info("=" * 55)
    logger.info(f"  {title}")
    logger.info("=" * 55)


def run_tests():
    logger.info("=== Phase 9 — Agent IA Validation ===")
    check_prerequisites()

    # ------------------------------------------------------------------
    # Cas 1 — Question générale → SANS retrieval
    # ------------------------------------------------------------------
    divider("Cas 1 : Question générale (sans retrieval attendu)")

    result1 = run_agent("Qu'est-ce que RAG ?")

    logger.info("\n" + result1.format_full())

    assert result1.answer, "La réponse ne doit pas être vide"
    assert result1.needs_retrieval is False, (
        f"Cette question ne devrait PAS déclencher de retrieval.\n"
        f"Raison décision : {result1.decision_reason}"
    )
    assert "decision_node:no_retrieval" in " ".join(result1.steps_executed) or \
           "decision_node:llm_decision" in " ".join(result1.steps_executed)

    logger.info(
        "Cas 1 OK",
        needs_retrieval=result1.needs_retrieval,
        steps=result1.steps_executed,
        duration_ms=round(result1.total_duration_ms),
    )

    # ------------------------------------------------------------------
    # Cas 2 — Question documentaire → AVEC retrieval
    # ------------------------------------------------------------------
    divider("Cas 2 : Question documentaire (retrieval attendu)")

    result2 = run_agent(
        "Quels sont les principaux défis économiques de Madagascar selon les rapports ?"
    )

    logger.info("\n" + result2.format_full())

    assert result2.answer, "La réponse ne doit pas être vide"
    assert result2.needs_retrieval is True, (
        f"Cette question DEVRAIT déclencher un retrieval.\n"
        f"Raison décision : {result2.decision_reason}"
    )
    assert result2.context_used is True
    assert result2.document_count > 0

    logger.info(
        "Cas 2 OK",
        needs_retrieval=result2.needs_retrieval,
        document_count=result2.document_count,
        confidence=result2.confidence_level,
        steps=result2.steps_executed,
        duration_ms=round(result2.total_duration_ms),
    )

    # ------------------------------------------------------------------
    # Cas 3 — Flux conditionnel vérification étapes
    # ------------------------------------------------------------------
    divider("Cas 3 : Vérification flux conditionnel")

    # Sans retrieval → steps ne doivent PAS contenir retriever_node
    assert "retriever_node" not in " ".join(result1.steps_executed), \
        "Le retriever ne doit pas être appelé pour une question générale"

    # Avec retrieval → steps doivent contenir retriever + reranker
    steps2 = " ".join(result2.steps_executed)
    assert "retriever_node" in steps2, "retriever_node doit être dans les étapes"
    assert "reranker_node" in steps2, "reranker_node doit être dans les étapes"
    assert "llm_node" in steps2, "llm_node doit être dans les étapes"

    logger.info(
        "Flux conditionnel OK",
        steps_sans_retrieval=result1.steps_executed,
        steps_avec_retrieval=result2.steps_executed,
    )

    # ------------------------------------------------------------------
    # Cas 4 — Question climatique → AVEC retrieval
    # ------------------------------------------------------------------
    divider("Cas 4 : Question climatique")

    result3 = run_agent(
        "Comment le changement climatique affecte-t-il le développement économique ?"
    )

    logger.info(
        "Résultat climatique",
        needs_retrieval=result3.needs_retrieval,
        confidence=result3.confidence_level,
        quality_score=round(result3.quality_score, 3),
        duration_ms=round(result3.total_duration_ms),
    )
    logger.info("Réponse :\n" + result3.answer[:600])

    # ------------------------------------------------------------------
    # Cas 5 — Langue française vérification
    # ------------------------------------------------------------------
    divider("Cas 5 : Validation langue française")

    french_markers = [
        "le ", "la ", "les ", "de ", "du ", "des ",
        "est ", "sont ", "dans ", "pour ", "avec ",
    ]

    for result, label in [
        (result1, "générale"),
        (result2, "documentaire"),
        (result3, "climatique"),
    ]:
        answer_lower = result.answer.lower()
        count = sum(1 for m in french_markers if m in answer_lower)
        is_french = count >= 3
        assert is_french, f"Réponse '{label}' devrait être en français (marqueurs: {count})"
        logger.info(f"Langue OK ({label})", french_markers={count})

    # ------------------------------------------------------------------
    # Résumé
    # ------------------------------------------------------------------
    divider("RÉSUMÉ PHASE 9")

    logger.info(
        "Comparaison durées",
        sans_retrieval_ms=round(result1.total_duration_ms),
        avec_retrieval_ms=round(result2.total_duration_ms),
    )

    logger.info("=== VALIDATION PHASE 9 TERMINEE ===")
    logger.info("Agent IA LangGraph opérationnel")
    logger.info("Décision dynamique : retrieval si nécessaire")
    logger.info("Flux conditionnel : START → decision → [retrieval?] → LLM → END")
    logger.info("Réutilisation : RetrieverService + RerankerService + LLMService")
    logger.info("Prêt pour Phase 10 — Tools")


if __name__ == "__main__":
    run_tests()