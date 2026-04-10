"""
backend/app/agents/nodes/reranker_node.py

Node de reranking : filtre et trie les documents par pertinence sémantique.
Réutilise exactement le RerankerService de Phase 8.
"""

from backend.app.agents.state import AgentState
from backend.app.services.reranker_service import RerankerService
from backend.app.core.logger import get_logger

logger = get_logger(__name__)

_reranker = RerankerService(top_n=4, use_cache=True)


def reranker_node(state: AgentState) -> AgentState:
    """
    Reranke les documents selon leur pertinence sémantique.

    Lit depuis l'état  : question, documents
    Écrit dans l'état  : reranked_documents

    Args:
        state: État courant du graphe

    Returns:
        État mis à jour avec les documents rerankés
    """
    question = state["question"]
    documents = state.get("documents", [])
    steps = state.get("steps_executed", [])

    if not documents:
        logger.warning("Reranker node — no documents to rerank")
        steps.append("reranker_node:skipped_empty")
        return {
            **state,
            "reranked_documents": [],
            "steps_executed": steps,
        }

    logger.info(
        "Reranker node started",
        document_count=len(documents),
        question=question[:80],
    )

    try:
        reranked_docs, fallback_used = _reranker.rerank_with_fallback(
            query=question,
            documents=documents,
        )

        steps.append(
            f"reranker_node:{len(reranked_docs)}_docs"
            f"{'_fallback' if fallback_used else ''}"
        )

        logger.info(
            "Reranker node completed",
            original=len(documents),
            reranked=len(reranked_docs),
            fallback_used=fallback_used,
        )

        return {
            **state,
            "reranked_documents": reranked_docs,
            "steps_executed": steps,
        }

    except Exception as e:
        logger.error("Reranker node failed", error=str(e))
        steps.append("reranker_node:error_fallback")
        return {
            **state,
            "reranked_documents": documents[:4],
            "steps_executed": steps,
        }