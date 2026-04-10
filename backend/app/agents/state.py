"""
backend/app/agents/state.py

Définition de l'état partagé du graphe LangGraph.
L'état est le seul canal de communication entre les nodes.

Chaque node :
- lit des champs depuis l'état
- retourne un dict partiel pour mettre à jour l'état

Règle stricte : aucune variable globale entre nodes.
Tout passe par AgentState.
"""

from typing import Optional, TypedDict
from backend.app.db.vector_store import SearchResult


class AgentState(TypedDict, total=False):
    """
    État partagé du graphe LangGraph.

    Champs :
        question          : Question originale de l'utilisateur
        needs_retrieval   : Décision du decision_node
        decision_reason   : Raison de la décision (pour logs)
        documents         : Documents retournés par le retriever
        reranked_documents: Documents après reranking
        context_text      : Contexte formaté pour le LLM
        confidence_level  : Niveau de confiance du contexte
        quality_score     : Score qualité du contexte
        answer            : Réponse finale générée par le LLM
        error             : Message d'erreur si échec d'un node
        steps_executed    : Liste des étapes exécutées (pour logs)
        model_used        : Modèle LLM utilisé
        total_duration_ms : Durée totale de traitement
    """
    question: str
    needs_retrieval: bool
    decision_reason: str
    documents: list[SearchResult]
    reranked_documents: list[SearchResult]
    context_text: str
    confidence_level: str
    quality_score: float
    answer: str
    error: Optional[str]
    steps_executed: list[str]
    model_used: str
    total_duration_ms: float


def initial_state(question: str) -> AgentState:
    """
    Crée l'état initial pour une nouvelle question.

    Args:
        question: Question de l'utilisateur

    Returns:
        AgentState avec valeurs par défaut
    """
    return AgentState(
        question=question,
        needs_retrieval=False,
        decision_reason="",
        documents=[],
        reranked_documents=[],
        context_text="",
        confidence_level="none",
        quality_score=0.0,
        answer="",
        error=None,
        steps_executed=[],
        model_used="",
        total_duration_ms=0.0,
    )