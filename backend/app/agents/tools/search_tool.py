"""
backend/app/agents/tools/search_tool.py

Tool de recherche externe.
Simule une recherche web enrichie avec des données économiques
et peut être connecté à une vraie API de recherche (SerpAPI, Tavily, etc.)

Structure de retour standardisée :
{
    "results": [...],
    "source": "web|internal|simulation",
    "query": str,
    "success": bool,
    "error": str | None
}

Phase 11+ : ce tool sera enrichi avec des données
issues du pipeline d'ingestion Dagster.
"""

import time
from datetime import datetime
from backend.app.core.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Données simulées — remplacées en production par vraie API
# ---------------------------------------------------------------------------

SIMULATED_DATA = {
    "madagascar": {
        "pib_2024": "15.2 milliards USD",
        "croissance_2024": "4.2%",
        "inflation_2024": "8.1%",
        "taux_pauvrete": "75.2% (2022)",
        "source": "Banque Mondiale 2024",
        "date_maj": "2024-Q4",
    },
    "climat": {
        "temperature_hausse": "+1.5°C depuis 1970",
        "cyclones_2024": "3 cyclones majeurs",
        "secheresse": "Sud Madagascar en alerte rouge",
        "source": "GIEC + Météo Madagascar",
        "date_maj": "2024-Q3",
    },
    "economie_mondiale": {
        "croissance_mondiale": "3.1% (FMI 2024)",
        "inflation_mondiale": "5.8%",
        "commerce_mondial": "+2.4%",
        "source": "FMI World Economic Outlook 2024",
        "date_maj": "2024-Q4",
    },
    "afrique_subsaharienne": {
        "croissance_region": "3.8%",
        "ide_entrants": "45 milliards USD",
        "taux_pauvrete_extreme": "35%",
        "source": "Banque Africaine de Développement",
        "date_maj": "2024-Q3",
    },
}


def search_tool(query: str, max_results: int = 3) -> dict:
    """
    Effectue une recherche externe pour des données actuelles.

    En production : connecter à SerpAPI, Tavily, ou OpenRouter web search.
    En développement : retourne des données simulées pertinentes.

    Args:
        query: Requête de recherche en langage naturel
        max_results: Nombre maximum de résultats

    Returns:
        Dict structuré avec résultats et métadonnées
    """
    start_time = time.time()
    query_lower = query.lower()

    logger.info(
        "Search tool called",
        query=query[:80],
        max_results=max_results,
    )

    try:
        results = _search_simulated(query_lower, max_results)

        duration_ms = (time.time() - start_time) * 1000

        output = {
            "query": query,
            "results": results,
            "source": "simulated_web_search",
            "result_count": len(results),
            "success": True,
            "error": None,
            "duration_ms": round(duration_ms),
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(
            "Search tool completed",
            query=query[:60],
            result_count=len(results),
            duration_ms=round(duration_ms),
        )

        return output

    except Exception as e:
        logger.error("Search tool failed", query=query[:60], error=str(e))
        return {
            "query": query,
            "results": [],
            "source": "simulated_web_search",
            "result_count": 0,
            "success": False,
            "error": str(e),
            "duration_ms": 0,
            "timestamp": datetime.now().isoformat(),
        }


def _search_simulated(query_lower: str, max_results: int) -> list[dict]:
    """
    Recherche dans les données simulées selon les mots-clés de la requête.

    Args:
        query_lower: Requête en minuscules
        max_results: Limite de résultats

    Returns:
        Liste de résultats pertinents
    """
    results = []

    # Correspondance par mots-clés
    keyword_map = {
        "madagascar": "madagascar",
        "malgache": "madagascar",
        "climat": "climat",
        "climatique": "climat",
        "cyclone": "climat",
        "sécheresse": "climat",
        "économie mondiale": "economie_mondiale",
        "fmi": "economie_mondiale",
        "croissance mondiale": "economie_mondiale",
        "afrique": "afrique_subsaharienne",
        "subsaharienne": "afrique_subsaharienne",
        "bad": "afrique_subsaharienne",
    }

    matched_keys = set()
    for keyword, data_key in keyword_map.items():
        if keyword in query_lower:
            matched_keys.add(data_key)

    # Si aucun mot-clé trouvé, retourner données générales
    if not matched_keys:
        matched_keys = {"economie_mondiale"}

    for key in list(matched_keys)[:max_results]:
        if key in SIMULATED_DATA:
            data = SIMULATED_DATA[key]
            result = {
                "title": f"Données économiques — {key.replace('_', ' ').title()}",
                "content": _format_data_as_text(data),
                "source": data.get("source", "Source inconnue"),
                "date": data.get("date_maj", "N/A"),
                "relevance": "high",
            }
            results.append(result)

    return results


def _format_data_as_text(data: dict) -> str:
    """Formate un dict de données en texte lisible."""
    lines = []
    skip_keys = {"source", "date_maj"}
    for key, value in data.items():
        if key not in skip_keys:
            label = key.replace("_", " ").title()
            lines.append(f"{label} : {value}")
    return " | ".join(lines)