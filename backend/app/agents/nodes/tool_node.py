"""
backend/app/agents/nodes/tool_node.py

Node d'exécution des tools.
Dispatche vers le bon tool selon tool_name dans l'état,
exécute, et injecte le résultat dans tool_output.
"""

from backend.app.agents.state import AgentState
from backend.app.agents.tools.search_tool import search_tool
from backend.app.agents.tools.api_tool import api_tool
from backend.app.core.logger import get_logger

logger = get_logger(__name__)

# Registre des tools disponibles
TOOL_REGISTRY = {
    "search_tool": search_tool,
    "api_tool": api_tool,
}


def tool_node(state: AgentState) -> AgentState:
    """
    Exécute le tool sélectionné et injecte le résultat dans l'état.

    Lit depuis l'état  : tool_name, question
    Écrit dans l'état  : tool_output, tool_input

    Args:
        state: État courant du graphe

    Returns:
        État mis à jour avec tool_output
    """
    tool_name = state.get("tool_name", "search_tool")
    question = state["question"]
    steps = state.get("steps_executed", [])

    logger.info(
        "Tool node started",
        tool_name=tool_name,
        question=question[:60],
    )

    # Vérification que le tool existe
    if tool_name not in TOOL_REGISTRY:
        logger.warning(
            "Unknown tool — falling back to search_tool",
            requested=tool_name,
            available=list(TOOL_REGISTRY.keys()),
        )
        tool_name = "search_tool"

    tool_fn = TOOL_REGISTRY[tool_name]

    try:
        # Préparation input selon le tool
        if tool_name == "search_tool":
            tool_input = {"query": question, "max_results": 3}
            tool_output = tool_fn(**tool_input)

        elif tool_name == "api_tool":
            # Détecter l'action appropriée
            action = _detect_api_action(question)
            tool_input = {"action": action}
            tool_output = tool_fn(**tool_input)

        else:
            tool_input = {"query": question}
            tool_output = tool_fn(**tool_input)

        steps.append(f"tool_node:{tool_name}({'ok' if tool_output.get('success') else 'error'})")

        logger.info(
            "Tool node completed",
            tool_name=tool_name,
            success=tool_output.get("success"),
            result_count=tool_output.get("result_count", 1),
        )

        return {
            **state,
            "tool_input": tool_input,
            "tool_output": tool_output,
            "steps_executed": steps,
        }

    except Exception as e:
        logger.error("Tool node failed", tool_name=tool_name, error=str(e))
        steps.append(f"tool_node:{tool_name}:error")
        return {
            **state,
            "tool_input": {},
            "tool_output": {
                "success": False,
                "error": str(e),
                "results": [],
            },
            "steps_executed": steps,
        }


def _detect_api_action(question: str) -> str:
    """Détecte l'action API appropriée selon la question."""
    q = question.lower()
    if any(kw in q for kw in ["statut", "pipeline", "état"]):
        return "pipeline_status"
    elif any(kw in q for kw in ["collection", "documents", "combien"]):
        return "collection_stats"
    else:
        return "system_info"