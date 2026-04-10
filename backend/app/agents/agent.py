"""
backend/app/agents/agent.py

Point d'entrée principal de l'agent IA.
Expose run_agent() pour usage depuis l'API et les scripts.

Ce fichier :
- Initialise le graphe LangGraph (une seule fois)
- Expose run_agent(question) → AgentResult
- Gère le timing et les métriques globales
"""

import time
from dataclasses import dataclass, field

from backend.app.agents.state import AgentState, initial_state
from backend.app.agents.graph import build_agent_graph
from backend.app.core.logger import get_logger

logger = get_logger(__name__)

# Graphe compilé une seule fois (singleton)
_agent_graph = None


def _get_graph():
    """Retourne le graphe compilé (singleton)."""
    global _agent_graph
    if _agent_graph is None:
        _agent_graph = build_agent_graph()
    return _agent_graph


# ---------------------------------------------------------------------------
# Types de retour
# ---------------------------------------------------------------------------

@dataclass
class AgentResult:
    """
    Résultat structuré de l'agent IA.
    Compatible avec le format de réponse de l'API /chat.
    """
    question: str
    answer: str
    needs_retrieval: bool
    decision_reason: str
    context_used: bool
    context_text: str
    confidence_level: str
    quality_score: float
    document_count: int
    model_used: str
    steps_executed: list[str]
    total_duration_ms: float
    error: str = None

    def format_full(self) -> str:
        """Formate le résultat pour affichage/debug."""
        lines = [
            "=" * 60,
            f"QUESTION     : {self.question}",
            f"DÉCISION     : {'RETRIEVAL' if self.needs_retrieval else 'DIRECT LLM'}",
            f"RAISON       : {self.decision_reason}",
            f"CONFIANCE    : {self.confidence_level}",
            f"QUALITÉ      : {self.quality_score:.3f}",
            f"DURÉE        : {self.total_duration_ms:.0f}ms",
            "=" * 60,
            "",
            "RÉPONSE :",
            self.answer,
            "",
            "ÉTAPES EXÉCUTÉES :",
        ]
        for step in self.steps_executed:
            lines.append(f"  → {step}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Point d'entrée principal
# ---------------------------------------------------------------------------

def run_agent(question: str) -> AgentResult:
    """
    Exécute l'agent IA pour une question donnée.

    Flux :
    1. Initialisation de l'état
    2. Exécution du graphe LangGraph
    3. Extraction du résultat depuis l'état final
    4. Retour de AgentResult structuré

    Args:
        question: Question en langage naturel (français)

    Returns:
        AgentResult avec réponse, décision et métriques

    Raises:
        ValueError: Si la question est vide
    """
    if not question or not question.strip():
        raise ValueError("La question ne peut pas être vide.")

    start_time = time.time()

    logger.info(
        "Agent started",
        question=question[:80],
    )

    # Initialisation de l'état
    state = initial_state(question.strip())

    try:
        # Exécution du graphe LangGraph
        graph = _get_graph()
        final_state: AgentState = graph.invoke(state)

        total_duration_ms = (time.time() - start_time) * 1000

        # Extraction des résultats
        result = AgentResult(
            question=question,
            answer=final_state.get("answer", ""),
            needs_retrieval=final_state.get("needs_retrieval", False),
            decision_reason=final_state.get("decision_reason", ""),
            context_used=bool(final_state.get("context_text", "")),
            context_text=final_state.get("context_text", ""),
            confidence_level=final_state.get("confidence_level", "none"),
            quality_score=final_state.get("quality_score", 0.0),
            document_count=len(final_state.get("reranked_documents", [])),
            model_used=final_state.get("model_used", ""),
            steps_executed=final_state.get("steps_executed", []),
            total_duration_ms=total_duration_ms,
            error=final_state.get("error"),
        )

        logger.info(
            "Agent completed",
            question=question[:60],
            needs_retrieval=result.needs_retrieval,
            decision_reason=result.decision_reason,
            context_used=result.context_used,
            confidence_level=result.confidence_level,
            answer_length=len(result.answer),
            steps=result.steps_executed,
            total_duration_ms=round(total_duration_ms),
        )

        return result

    except Exception as e:
        total_duration_ms = (time.time() - start_time) * 1000
        logger.error(
            "Agent failed",
            question=question[:60],
            error=str(e),
            error_type=type(e).__name__,
        )
        return AgentResult(
            question=question,
            answer="Une erreur est survenue dans le pipeline agent.",
            needs_retrieval=False,
            decision_reason="",
            context_used=False,
            context_text="",
            confidence_level="none",
            quality_score=0.0,
            document_count=0,
            model_used="",
            steps_executed=[],
            total_duration_ms=total_duration_ms,
            error=str(e),
        )