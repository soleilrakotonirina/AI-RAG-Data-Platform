"""
backend/app/agents/nodes/decision_node.py

Node de décision — Phase 10.

Décide maintenant entre TROIS chemins :

1. Retrieval (documents indexés)
2. Tool (données dynamiques/actuelles)
3. LLM direct (question générale)
"""

import re
import httpx
from backend.app.agents.state import AgentState
from backend.app.core.settings import get_settings
from backend.app.core.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Mots-clés de classification
# ---------------------------------------------------------------------------

# Chemin : Tool (données dynamiques)
TOOL_KEYWORDS = [
    "actuel", "actuels", "actuelle", "actuelles", "actuellement", "aujourd'hui", "maintenant",
    "dernières données", "données récentes", "mise à jour",
    "latest", "current",
    "statistiques actuelles", "chiffres récents",
    "en ce moment", "2025", "2026",
    "combien sont", "quel est le taux actuel",
    "statut", "état du système", "pipeline status",
]

# Chemin : Retrieval (documents indexés)
RETRIEVAL_KEYWORDS = [
    "rapport", "rapports", "document", "étude", "analyse",
    "selon", "d'après", "source", "données historiques",
    "madagascar", "malgache", "pib", "croissance", "pauvreté",
    "exportation", "urbanisation", "inflation", "économie",
    "climatique", "climat", "température", "carbone", "émission",
    "banque mondiale", "world bank", "fmi",
    "quelle est la situation", "quels sont les chiffres",
]

# Chemin : LLM direct (questions générales)
NO_RETRIEVAL_KEYWORDS = [
    "qu'est-ce que", "c'est quoi", "définition", "explique",
    "comment fonctionne", "à quoi sert", "différence entre",
    "bonjour", "merci", "aide-moi",
]

# Prompt de décision LLM
DECISION_PROMPT = """Tu es un classificateur de questions.

Question : {question}

Classe cette question dans l'une des trois catégories :
- RETRIEVAL : question sur des faits dans des rapports/documents indexés
- TOOL : question nécessitant des données actuelles ou dynamiques
- DIRECT : question générale ne nécessitant pas de source externe

Exemples :
- "Qu'est-ce que FastAPI ?" → DIRECT
- "Défis économiques Madagascar ?" → RETRIEVAL
- "Données économiques actuelles ?" → TOOL

Réponds UNIQUEMENT par : RETRIEVAL, TOOL ou DIRECT

Réponse :"""


# ---------------------------------------------------------------------------
# Utilitaire — correspondance par mots entiers
# ---------------------------------------------------------------------------

def _keyword_match(keyword: str, text: str) -> bool:
    """
    Vérifie si un mot-clé apparaît dans le texte comme mot entier
    ou comme expression complète.

    Évite les faux positifs :
    - "api" ne matche PAS "fastapi"
    - "api" matche "appel api" ou "via api"

    Args:
        keyword: Mot-clé à chercher
        text: Texte en minuscules

    Returns:
        True si le mot-clé est présent comme mot/expression entier(e)
    """
    # Pour les expressions multi-mots, recherche directe
    if " " in keyword:
        return keyword in text

    # Pour les mots simples, correspondance par frontière de mot
    pattern = r'\b' + re.escape(keyword) + r'\b'
    return bool(re.search(pattern, text))


def decision_node(state: AgentState) -> AgentState:
    """
    Analyse la question et décide du chemin d'exécution.

    Trois chemins :
    - needs_tool=True       : données dynamiques (search_tool / api_tool)
    - needs_retrieval=True  : documents indexés (ChromaDB)
    - direct                : LLM seul

    Stratégie :
    1. Mots-clés TOOL (mots entiers) → chemin tool
    2. Mots-clés NO_RETRIEVAL → chemin direct
    3. Mots-clés RETRIEVAL → chemin retrieval
    4. Ambigu → LLM classificateur

    Args:
        state: État courant

    Returns:
        État mis à jour avec chemin décidé
    """
    question = state["question"]
    question_lower = question.lower()
    steps = state.get("steps_executed", [])

    logger.info("Decision node started", question=question[:80])

    # Étape 1 : Mots-clés TOOL (correspondance mots entiers)
    matched_tool_kw = [
        kw for kw in TOOL_KEYWORDS
        if _keyword_match(kw, question_lower)
    ]
    if matched_tool_kw:
        reason = f"Données actuelles/dynamiques : {matched_tool_kw[:2]}"
        steps.append("decision_node:tool_lexical")
        logger.info("Decision: TOOL (lexical)", matched=matched_tool_kw[:2])
        return {
            **state,
            "needs_retrieval": False,
            "needs_tool": True,
            "tool_name": _select_tool(question_lower),
            "decision_reason": reason,
            "steps_executed": steps,
        }

    # Étape 2 : Mots-clés NO_RETRIEVAL
    matched_no_ret = [
        kw for kw in NO_RETRIEVAL_KEYWORDS
        if _keyword_match(kw, question_lower)
    ]
    if matched_no_ret:
        has_ret_kw = any(
            _keyword_match(kw, question_lower)
            for kw in RETRIEVAL_KEYWORDS
        )
        if not has_ret_kw:
            reason = f"Question générale : '{matched_no_ret[0]}'"
            steps.append("decision_node:direct_lexical")
            logger.info("Decision: DIRECT (lexical)", keyword=matched_no_ret[0])
            return {
                **state,
                "needs_retrieval": False,
                "needs_tool": False,
                "decision_reason": reason,
                "steps_executed": steps,
            }

    # Étape 3 : Mots-clés RETRIEVAL
    matched_ret_kw = [
        kw for kw in RETRIEVAL_KEYWORDS
        if _keyword_match(kw, question_lower)
    ]
    if matched_ret_kw:
        reason = f"Documents indexés pertinents : {matched_ret_kw[:2]}"
        steps.append("decision_node:retrieval_lexical")
        logger.info("Decision: RETRIEVAL (lexical)", matched=matched_ret_kw[:2])
        return {
            **state,
            "needs_retrieval": True,
            "needs_tool": False,
            "decision_reason": reason,
            "steps_executed": steps,
        }

    # Étape 4 : LLM classificateur
    logger.info("Decision: ambiguous — calling LLM classifier")
    path, reason = _llm_decision(question)
    steps.append(f"decision_node:llm_decision({path})")

    return {
        **state,
        "needs_retrieval": path == "retrieval",
        "needs_tool": path == "tool",
        "tool_name": _select_tool(question_lower) if path == "tool" else "",
        "decision_reason": reason,
        "steps_executed": steps,
    }


def _select_tool(question_lower: str) -> str:
    """Sélectionne le tool approprié selon la question."""
    api_keywords = ["statut", "pipeline", "système", "collection", "base de données"]
    if any(_keyword_match(kw, question_lower) for kw in api_keywords):
        return "api_tool"
    return "search_tool"


def _llm_decision(question: str) -> tuple[str, str]:
    """Utilise le LLM pour classer la question."""
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
                "max_tokens": 10,
            },
            timeout=15.0,
        )
        response.raise_for_status()
        raw = response.json()["choices"][0]["message"]["content"].strip().upper()

        if "TOOL" in raw:
            return "tool", f"Classificateur LLM → TOOL (réponse: {raw})"
        elif "RETRIEVAL" in raw:
            return "retrieval", f"Classificateur LLM → RETRIEVAL (réponse: {raw})"
        else:
            return "direct", f"Classificateur LLM → DIRECT (réponse: {raw})"

    except Exception as e:
        logger.warning("LLM decision failed — defaulting to retrieval", error=str(e))
        return "retrieval", f"Fallback → retrieval (erreur: {type(e).__name__})"