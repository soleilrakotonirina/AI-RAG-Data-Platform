"""
backend/app/agents/nodes/decision_node.py

Node de décision : détermine si la question nécessite un retrieval.

Stratégie de décision en deux niveaux :
1. Règles lexicales rapides (mots-clés → pas de retrieval)
2. LLM décisionnel si ambigu (appel léger à OpenRouter)

Exemples :
- "Qu'est-ce que FastAPI ?" → needs_retrieval=False (connaissance générale)
- "Que disent les rapports sur Madagascar ?" → needs_retrieval=True
- "Quel est le taux de pauvreté selon les données ?" → needs_retrieval=True
"""

import httpx
from backend.app.agents.state import AgentState
from backend.app.core.settings import get_settings
from backend.app.core.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Mots-clés indiquant qu'un retrieval est nécessaire
# ---------------------------------------------------------------------------

RETRIEVAL_KEYWORDS = [
    # Références explicites aux documents
    "rapport", "rapports", "document", "données", "étude",
    "selon", "d'après", "source", "analyse", "statistique",
    # Contexte Madagascar / économie
    "madagascar", "malgache", "pib", "croissance", "pauvreté",
    "exportation", "urbanisation", "inflation", "économie",
    # Contexte climat
    "climatique", "climat", "température", "carbone", "émission",
    # Questions de recherche
    "quelle est la situation", "quels sont les chiffres",
    "combien", "quel pourcentage", "quelle proportion",
    "comment évolue", "quels facteurs",
    # Références temporelles précises
    "en 2020", "en 2021", "en 2022", "en 2023", "en 2024",
]

# Mots-clés indiquant une question générale (pas de retrieval)
NO_RETRIEVAL_KEYWORDS = [
    "qu'est-ce que", "c'est quoi", "définition", "explique",
    "comment fonctionne", "à quoi sert", "différence entre",
    "bonjour", "merci", "aide", "aide-moi",
]

# Prompt de décision LLM (utilisé si règles ambiguës)
DECISION_PROMPT = """Tu es un assistant expert en classification de questions.

Question : {question}

Cette question nécessite-t-elle une recherche dans une base documentaire
contenant des rapports économiques sur Madagascar, le changement climatique
et le développement ?

Réponds UNIQUEMENT par :
- OUI si la question porte sur des faits spécifiques, statistiques, données, rapports
- NON si la question est générale, conceptuelle ou ne nécessite pas de données

Réponse (OUI ou NON) :"""


def decision_node(state: AgentState) -> AgentState:
    """
    Analyse la question et décide si un retrieval est nécessaire.

    Stratégie :
    1. Vérifier les mots-clés NO_RETRIEVAL (→ False immédiatement)
    2. Vérifier les mots-clés RETRIEVAL (→ True immédiatement)
    3. Si ambigu → appel LLM léger pour décider

    Args:
        state: État courant du graphe

    Returns:
        État mis à jour avec needs_retrieval et decision_reason
    """
    question = state["question"]
    question_lower = question.lower()

    logger.info(
        "Decision node started",
        question=question[:80],
    )

    # Étape 1 : Vérifier mots-clés "pas de retrieval"
    for keyword in NO_RETRIEVAL_KEYWORDS:
        if keyword in question_lower:
            # Vérifier qu'il n'y a pas de sur-qualification
            has_retrieval_kw = any(
                kw in question_lower for kw in RETRIEVAL_KEYWORDS
            )
            if not has_retrieval_kw:
                reason = f"Question générale détectée (mot-clé: '{keyword}')"
                logger.info(
                    "Decision: NO retrieval (lexical rule)",
                    reason=reason,
                )
                steps = state.get("steps_executed", [])
                steps.append("decision_node:no_retrieval_lexical")
                return {
                    **state,
                    "needs_retrieval": False,
                    "decision_reason": reason,
                    "steps_executed": steps,
                }

    # Étape 2 : Vérifier mots-clés "retrieval nécessaire"
    matched_keywords = [
        kw for kw in RETRIEVAL_KEYWORDS if kw in question_lower
    ]
    if matched_keywords:
        reason = f"Mots-clés de recherche détectés : {matched_keywords[:3]}"
        logger.info(
            "Decision: YES retrieval (lexical rule)",
            matched_keywords=matched_keywords[:3],
        )
        steps = state.get("steps_executed", [])
        steps.append("decision_node:retrieval_lexical")
        return {
            **state,
            "needs_retrieval": True,
            "decision_reason": reason,
            "steps_executed": steps,
        }

    # Étape 3 : Ambigu → LLM décisionnel
    logger.info("Decision: ambiguous — calling LLM for decision")
    needs_retrieval, reason = _llm_decision(question)

    steps = state.get("steps_executed", [])
    steps.append(f"decision_node:llm_decision({'yes' if needs_retrieval else 'no'})")

    logger.info(
        "Decision completed",
        needs_retrieval=needs_retrieval,
        reason=reason,
    )

    return {
        **state,
        "needs_retrieval": needs_retrieval,
        "decision_reason": reason,
        "steps_executed": steps,
    }


def _llm_decision(question: str) -> tuple[bool, str]:
    """
    Utilise le LLM pour décider si un retrieval est nécessaire.
    Appel léger (max_tokens=5, temperature=0).

    Args:
        question: Question à analyser

    Returns:
        Tuple (needs_retrieval, reason)
    """
    settings = get_settings()

    try:
        response = httpx.post(
            url=f"{settings.openrouter_base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.openrouter_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "mistralai/mistral-small-3.1-24b-instruct",
                "messages": [
                    {
                        "role": "user",
                        "content": DECISION_PROMPT.format(question=question),
                    }
                ],
                "temperature": 0.0,
                "max_tokens": 5,
            },
            timeout=15.0,
        )
        response.raise_for_status()
        raw = response.json()["choices"][0]["message"]["content"].strip().upper()

        needs_retrieval = "OUI" in raw
        reason = f"Décision LLM : {'OUI' if needs_retrieval else 'NON'} (réponse: {raw})"
        return needs_retrieval, reason

    except Exception as e:
        logger.warning(
            "LLM decision failed — defaulting to retrieval",
            error=str(e),
        )
        return True, f"Fallback → retrieval par défaut (erreur LLM: {type(e).__name__})"