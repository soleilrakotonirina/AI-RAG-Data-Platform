"""
backend/app/agents/graph.py

Définition du graphe LangGraph de l'agent IA.

Structure du graphe :

    START
      │
      ▼
  decision_node
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
      │     llm_node
      │         │
      │         ▼
      │        END
      │
      └── needs_retrieval=False
                │
                ▼
            llm_node
                │
                ▼
               END
"""

from langgraph.graph import StateGraph, END, START

from backend.app.agents.state import AgentState
from backend.app.agents.nodes.decision_node import decision_node
from backend.app.agents.nodes.retriever_node import retriever_node
from backend.app.agents.nodes.reranker_node import reranker_node
from backend.app.agents.nodes.llm_node import llm_node
from backend.app.core.logger import get_logger

logger = get_logger(__name__)


def _route_after_decision(state: AgentState) -> str:
    """
    Fonction de routage conditionnel après decision_node.

    Lit needs_retrieval depuis l'état et retourne le nom
    du prochain node à exécuter.

    Args:
        state: État courant après decision_node

    Returns:
        Nom du prochain node : "retriever_node" ou "llm_node"
    """
    needs_retrieval = state.get("needs_retrieval", False)
    next_node = "retriever_node" if needs_retrieval else "llm_node"

    logger.info(
        "Routing decision",
        needs_retrieval=needs_retrieval,
        next_node=next_node,
        decision_reason=state.get("decision_reason", ""),
    )

    return next_node


def build_agent_graph() -> StateGraph:
    """
    Construit et compile le graphe LangGraph de l'agent.

    Structure :
    - Nodes : decision, retriever, reranker, llm
    - Edges conditionnels : decision → retriever ou llm
    - Edges normaux : retriever → reranker → llm → END

    Returns:
        Graphe LangGraph compilé et prêt à exécuter
    """
    graph = StateGraph(AgentState)

    # Enregistrement des nodes
    graph.add_node("decision_node", decision_node)
    graph.add_node("retriever_node", retriever_node)
    graph.add_node("reranker_node", reranker_node)
    graph.add_node("llm_node", llm_node)

    # Point d'entrée
    graph.add_edge(START, "decision_node")

    # Edge conditionnel après decision
    graph.add_conditional_edges(
        "decision_node",
        _route_after_decision,
        {
            "retriever_node": "retriever_node",
            "llm_node": "llm_node",
        },
    )

    # Edges du chemin avec retrieval
    graph.add_edge("retriever_node", "reranker_node")
    graph.add_edge("reranker_node", "llm_node")

    # Fin du graphe
    graph.add_edge("llm_node", END)

    compiled = graph.compile()

    logger.info(
        "Agent graph compiled",
        nodes=["decision_node", "retriever_node", "reranker_node", "llm_node"],
        conditional_edges=["decision_node → retriever_node|llm_node"],
    )

    return compiled