"""
backend/app/agents/graph.py

Graphe LangGraph — Phase 10.

Nouveau flow à 4 chemins :

    START
      │
      ▼
  decision_node
      │
      ├── needs_tool=True
      │         │
      │         ▼
      │     tool_node
      │         │
      │         ▼
      │     llm_node → END
      │
      ├── needs_retrieval=True
      │         │
      │         ▼
      │   retriever_node
      │         │
      │         ▼
      │   reranker_node
      │         │
      │         ▼
      │     llm_node → END
      │
      └── direct
                │
                ▼
            llm_node → END
"""

from langgraph.graph import StateGraph, END, START

from backend.app.agents.state import AgentState
from backend.app.agents.nodes.decision_node import decision_node
from backend.app.agents.nodes.retriever_node import retriever_node
from backend.app.agents.nodes.reranker_node import reranker_node
from backend.app.agents.nodes.llm_node import llm_node
from backend.app.agents.nodes.tool_node import tool_node
from backend.app.core.logger import get_logger

logger = get_logger(__name__)


def _route_after_decision(state: AgentState) -> str:
    """
    Routage conditionnel après decision_node.

    Priorité :
    1. needs_tool → tool_node
    2. needs_retrieval → retriever_node
    3. default → llm_node (direct)

    Args:
        state: État après decision_node

    Returns:
        Nom du prochain node
    """
    needs_tool = state.get("needs_tool", False)
    needs_retrieval = state.get("needs_retrieval", False)

    if needs_tool:
        next_node = "tool_node"
    elif needs_retrieval:
        next_node = "retriever_node"
    else:
        next_node = "llm_node"

    logger.info(
        "Routing decision",
        needs_tool=needs_tool,
        needs_retrieval=needs_retrieval,
        next_node=next_node,
        decision_reason=state.get("decision_reason", ""),
    )

    return next_node


def build_agent_graph() -> StateGraph:
    """
    Construit et compile le graphe LangGraph Phase 10.

    Nodes : decision, tool, retriever, reranker, llm
    Edges conditionnels : decision → tool|retriever|llm
    Edges normaux :
        tool → llm
        retriever → reranker → llm
        llm → END

    Returns:
        Graphe LangGraph compilé
    """
    graph = StateGraph(AgentState)

    # Enregistrement des nodes
    graph.add_node("decision_node", decision_node)
    graph.add_node("tool_node", tool_node)
    graph.add_node("retriever_node", retriever_node)
    graph.add_node("reranker_node", reranker_node)
    graph.add_node("llm_node", llm_node)

    # Point d'entrée
    graph.add_edge(START, "decision_node")

    # Edge conditionnel depuis decision
    graph.add_conditional_edges(
        "decision_node",
        _route_after_decision,
        {
            "tool_node": "tool_node",
            "retriever_node": "retriever_node",
            "llm_node": "llm_node",
        },
    )

    # Chemin tool → llm
    graph.add_edge("tool_node", "llm_node")

    # Chemin retrieval → reranker → llm
    graph.add_edge("retriever_node", "reranker_node")
    graph.add_edge("reranker_node", "llm_node")

    # Fin
    graph.add_edge("llm_node", END)

    compiled = graph.compile()

    logger.info(
        "Agent graph compiled (Phase 10)",
        nodes=["decision_node", "tool_node", "retriever_node", "reranker_node", "llm_node"],
        paths=["tool", "retrieval", "direct"],
    )

    return compiled