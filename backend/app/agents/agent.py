"""
backend/app/agents/agent.py

Point d'entrée de l'agent IA — Phase 10.
"""

import time
from dataclasses import dataclass, field

from backend.app.agents.state import AgentState, initial_state
from backend.app.agents.graph import build_agent_graph
from backend.app.core.logger import get_logger

logger = get_logger(__name__)

_agent_graph = None


def _get_graph():
    global _agent_graph
    if _agent_graph is None:
        _agent_graph = build_agent_graph()
    return _agent_graph


@dataclass
class AgentResult:
    """Résultat structuré de l'agent IA — Phase 10."""
    question: str
    answer: str
    needs_retrieval: bool
    needs_tool: bool
    tool_name: str
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
    tool_output: dict = field(default_factory=dict)

    def format_full(self) -> str:
        path = "TOOL" if self.needs_tool else \
               "RETRIEVAL" if self.needs_retrieval else "DIRECT LLM"
        lines = [
            "=" * 60,
            f"QUESTION     : {self.question}",
            f"CHEMIN       : {path}",
            f"RAISON       : {self.decision_reason}",
            f"TOOL         : {self.tool_name or 'aucun'}",
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


def run_agent(question: str) -> AgentResult:
    """
    Exécute l'agent IA Phase 10.

    Chemins possibles :
    - Tool path    : données dynamiques
    - Retrieval path : documents ChromaDB
    - Direct path  : LLM seul

    Args:
        question: Question en français

    Returns:
        AgentResult structuré
    """
    if not question or not question.strip():
        raise ValueError("La question ne peut pas être vide.")

    start_time = time.time()

    logger.info("Agent started (Phase 10)", question=question[:80])

    state = initial_state(question.strip())

    try:
        graph = _get_graph()
        final_state: AgentState = graph.invoke(state)
        total_duration_ms = (time.time() - start_time) * 1000

        result = AgentResult(
            question=question,
            answer=final_state.get("answer", ""),
            needs_retrieval=final_state.get("needs_retrieval", False),
            needs_tool=final_state.get("needs_tool", False),
            tool_name=final_state.get("tool_name", ""),
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
            tool_output=final_state.get("tool_output", {}),
        )

        logger.info(
            "Agent completed",
            path="tool" if result.needs_tool else
                 "retrieval" if result.needs_retrieval else "direct",
            steps=result.steps_executed,
            duration_ms=round(total_duration_ms),
        )

        return result

    except Exception as e:
        total_duration_ms = (time.time() - start_time) * 1000
        logger.error("Agent failed", error=str(e))
        return AgentResult(
            question=question,
            answer="Une erreur est survenue dans le pipeline agent.",
            needs_retrieval=False,
            needs_tool=False,
            tool_name="",
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