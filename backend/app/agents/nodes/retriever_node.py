"""
backend/app/agents/nodes/retriever_node.py

Node de retrieval : appelle RetrieverService et stocke les résultats dans l'état.
Réutilise exactement le RetrieverService de Phase 4.
"""

from backend.app.agents.state import AgentState
from backend.app.services.retriever_service import RetrieverService
from backend.app.core.logger import get_logger

logger = get_logger(__name__)

# Instance réutilisée (évite réinitialisation à chaque appel)
_retriever = RetrieverService(top_k=8)


def retriever_node(state: AgentState) -> AgentState:
    """
    Effectue le retrieval sémantique pour la question courante.

    Lit depuis l'état  : question
    Écrit dans l'état  : documents

    Args:
        state: État courant du graphe

    Returns:
        État mis à jour avec les documents récupérés
    """
    question = state["question"]

    logger.info(
        "Retriever node started",
        question=question[:80],
    )

    steps = state.get("steps_executed", [])

    try:
        retrieval_result = _retriever.retrieve(
            query=question,
            use_mmr=True,
            mmr_lambda=0.7,
            adaptive_k=True,
        )

        steps.append(f"retriever_node:{retrieval_result.found}_docs")

        logger.info(
            "Retriever node completed",
            found=retrieval_result.found,
            method=retrieval_result.retrieval_method,
            score_stats=retrieval_result.score_stats,
        )

        return {
            **state,
            "documents": retrieval_result.documents,
            "steps_executed": steps,
        }

    except Exception as e:
        logger.error("Retriever node failed", error=str(e))
        steps.append("retriever_node:error")
        return {
            **state,
            "documents": [],
            "error": f"Retrieval error: {str(e)}",
            "steps_executed": steps,
        }