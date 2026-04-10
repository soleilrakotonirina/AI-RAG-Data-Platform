"""
scripts/test_tools.py

Validation Phase 10 — Tools pour agent IA.

Tests :
1. search_tool isolé
2. api_tool isolé
3. Agent — chemin tool
4. Agent — chemin retrieval
5. Agent — chemin direct
6. Vérification routage à 3 chemins

Lancer :
    python scripts/test_tools.py
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from backend.app.core.logger import configure_logging, get_logger
from backend.app.agents.tools.search_tool import search_tool
from backend.app.agents.tools.api_tool import api_tool
from backend.app.agents.agent import run_agent
from backend.app.db.vector_store import VectorStore

configure_logging()
logger = get_logger(__name__)


def check_prerequisites():
    store = VectorStore()
    count = store.count()
    if count == 0:
        logger.error("ChromaDB vide — lancer : python scripts/ingest_documents.py")
        sys.exit(1)
    logger.info("Documents ChromaDB", count=count)


def divider(title: str):
    logger.info("=" * 55)
    logger.info(f"  {title}")
    logger.info("=" * 55)


def run_tests():
    logger.info("=== Phase 10 — Tools Validation ===")
    check_prerequisites()

    # ------------------------------------------------------------------
    # Test 1 : search_tool isolé
    # ------------------------------------------------------------------
    divider("Test 1 : search_tool isolé")

    result = search_tool(query="économie Madagascar 2024", max_results=2)

    assert result["success"] is True
    assert len(result["results"]) > 0
    assert "source" in result

    logger.info(
        "search_tool OK",
        result_count=result["result_count"],
        source=result["source"],
        duration_ms=result["duration_ms"],
    )
    for r in result["results"]:
        logger.info(f"  → {r['title'][:50]} | {r['source']}")

    # ------------------------------------------------------------------
    # Test 2 : api_tool isolé
    # ------------------------------------------------------------------
    divider("Test 2 : api_tool isolé")

    result_api = api_tool(action="collection_stats")

    assert result_api["success"] is True
    assert result_api["result"] is not None
    assert "document_count" in result_api["result"]

    logger.info(
        "api_tool OK",
        action="collection_stats",
        document_count=result_api["result"]["document_count"],
        duration_ms=result_api["duration_ms"],
    )

    # ------------------------------------------------------------------
    # Test 3 : Agent — chemin TOOL
    # ------------------------------------------------------------------
    divider("Test 3 : Agent — chemin TOOL")

    result_tool = run_agent(
        "Donne-moi les données économiques actuelles de Madagascar"
    )

    logger.info("\n" + result_tool.format_full())

    assert result_tool.answer
    assert result_tool.needs_tool is True, (
        f"Cette question devrait déclencher un tool.\n"
        f"Chemin décidé : retrieval={result_tool.needs_retrieval}, "
        f"tool={result_tool.needs_tool}\n"
        f"Raison : {result_tool.decision_reason}"
    )
    assert "tool_node" in " ".join(result_tool.steps_executed)

    logger.info(
        "Chemin TOOL OK",
        tool_name=result_tool.tool_name,
        steps=result_tool.steps_executed,
    )

    # ------------------------------------------------------------------
    # Test 4 : Agent — chemin RETRIEVAL
    # ------------------------------------------------------------------
    divider("Test 4 : Agent — chemin RETRIEVAL")

    result_ret = run_agent(
        "Quels sont les défis économiques de Madagascar selon les rapports ?"
    )

    assert result_ret.answer
    assert result_ret.needs_retrieval is True
    assert result_ret.needs_tool is False

    logger.info(
        "Chemin RETRIEVAL OK",
        document_count=result_ret.document_count,
        confidence=result_ret.confidence_level,
        steps=result_ret.steps_executed,
    )
    
    logger.info("\n" + result_ret.format_full())   

    # ------------------------------------------------------------------
    # Test 5 : Agent — chemin DIRECT
    # ------------------------------------------------------------------
    divider("Test 5 : Agent — chemin DIRECT")

    result_direct = run_agent("Qu'est-ce que FastAPI ?")

    assert result_direct.answer
    assert result_direct.needs_retrieval is False
    assert result_direct.needs_tool is False

    logger.info(
        "Chemin DIRECT OK",
        steps=result_direct.steps_executed,
    )

    logger.info("\n" + result_direct.format_full()) 

    # ------------------------------------------------------------------
    # Test 6 : Vérification 3 chemins distincts
    # ------------------------------------------------------------------
    divider("Test 6 : Vérification 3 chemins distincts")

    paths = {
        "tool": result_tool.steps_executed,
        "retrieval": result_ret.steps_executed,
        "direct": result_direct.steps_executed,
    }

    for path_name, steps in paths.items():
        steps_str = " ".join(steps)
        logger.info(f"Chemin {path_name.upper()} : {steps}")

    assert "tool_node" in " ".join(result_tool.steps_executed)
    assert "retriever_node" in " ".join(result_ret.steps_executed)
    assert "tool_node" not in " ".join(result_direct.steps_executed)
    assert "retriever_node" not in " ".join(result_direct.steps_executed)

    logger.info("3 chemins distincts confirmés")

    # ------------------------------------------------------------------
    # Test 7 : Langue française
    # ------------------------------------------------------------------
    divider("Test 7 : Validation langue française")

    markers = ["le ", "la ", "les ", "de ", "du ", "est ", "sont "]
    for result, label in [
        (result_tool, "tool"),
        (result_ret, "retrieval"),
        (result_direct, "direct"),
    ]:
        count = sum(1 for m in markers if m in result.answer.lower())
        assert count >= 3, f"Réponse '{label}' devrait être en français"
        logger.info(f"Langue OK ({label})", markers_count=count)

    # ------------------------------------------------------------------
    # Résumé
    # ------------------------------------------------------------------
    divider("RÉSUMÉ PHASE 10")

    logger.info(
        "Durées par chemin",
        tool_ms=round(result_tool.total_duration_ms),
        retrieval_ms=round(result_ret.total_duration_ms),
        direct_ms=round(result_direct.total_duration_ms),
    )

    logger.info("=== VALIDATION PHASE 10 TERMINEE ===")
    logger.info("search_tool : données dynamiques simulées")
    logger.info("api_tool : accès ChromaDB + système")
    logger.info("3 chemins agent : tool / retrieval / direct")
    logger.info("Prêt pour Phase 11 — Pipeline ingestion")


if __name__ == "__main__":
    run_tests()