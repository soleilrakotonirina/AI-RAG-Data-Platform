"""
indexing/enrichment.py

Enrichissement des métadonnées des chunks.

Détections automatiques :
- Langue (fr/en/unknown)
- Domaine thématique (economics, climate, finance, etc.)
- Pays mentionnés (détection basique)
- Timestamp d'indexation

Générique : fonctionne pour tous les domaines, pas seulement Madagascar.
Préparation Dagster : fonction pure, transformable en op.
"""

import re
import sys
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from backend.app.core.logger import get_logger
from indexing.chunking import TextChunk

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Tables de référence
# ---------------------------------------------------------------------------

DOMAIN_KEYWORDS = {
    "economics": [
        "gdp", "pib", "economy", "économie", "growth", "croissance",
        "inflation", "trade", "export", "import", "poverty", "pauvreté",
        "fiscal", "budget", "investment", "revenue", "productivity",
        "manufacturing", "agriculture", "sector",
    ],
    "climate": [
        "climate", "climat", "carbon", "carbone", "emission", "cyclone",
        "drought", "flood", "rainfall", "temperature", "greenhouse",
        "fossil", "renewable", "adaptation", "resilience",
    ],
    "urbanization": [
        "urban", "urbain", "city", "ville", "population", "migration",
        "rural", "housing", "logement", "infrastructure", "density",
    ],
    "finance": [
        "finance", "bank", "banque", "credit", "loan", "debt", "dette",
        "interest", "monetary", "currency", "monnaie", "microfinance",
        "fiscal", "tax", "revenue",
    ],
    "development": [
        "development", "développement", "human", "health", "santé",
        "education", "poverty", "pauvreté", "inequality", "inégalité",
        "social", "welfare", "nutrition",
    ],
}

COUNTRY_KEYWORDS = {
    "madagascar": ["madagascar", "malagasy", "malgache", "antananarivo"],
    "west_bank_gaza": ["west bank", "gaza", "palestine", "palestinian"],
    "global": ["global", "worldwide", "international", "world bank", "imf"],
}

FRENCH_MARKERS = {"le", "la", "les", "de", "du", "des", "est", "sont", "dans", "pour", "avec"}
ENGLISH_MARKERS = {"the", "of", "and", "in", "to", "is", "are", "for", "with", "that", "from"}


@dataclass
class EnrichedChunk:
    """Chunk enrichi avec métadonnées complètes — prêt pour ChromaDB."""
    id: str
    text: str
    metadata: dict = field(default_factory=dict)


def detect_language(text: str) -> str:
    """Détecte la langue dominante du texte (fr/en/unknown)."""
    words = set(text.lower().split()[:150])
    fr_score = len(words & FRENCH_MARKERS)
    en_score = len(words & ENGLISH_MARKERS)
    if fr_score > en_score:
        return "fr"
    elif en_score > fr_score:
        return "en"
    return "unknown"


def detect_domain(text: str) -> str:
    """Identifie le domaine thématique principal."""
    text_lower = text.lower()
    scores = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[domain] = score
    return max(scores, key=scores.get) if scores else "general"


def detect_countries(text: str) -> list[str]:
    """Détecte les pays mentionnés dans le texte."""
    text_lower = text.lower()
    found = []
    for country, keywords in COUNTRY_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            found.append(country)
    return found if found else ["unknown"]


def generate_chunk_id(source: str, chunk_index: int) -> str:
    """Génère un ID unique pour un chunk."""
    clean = re.sub(r'[^a-zA-Z0-9_-]', '_', Path(source).stem)[:50]
    return f"{clean}_chunk_{chunk_index}"


def enrich_chunk(chunk: TextChunk, domain_override: str = None) -> EnrichedChunk:
    """
    Enrichit un chunk avec métadonnées détectées automatiquement.

    Args:
        chunk: TextChunk à enrichir
        domain_override: Forcer un domaine spécifique

    Returns:
        EnrichedChunk avec métadonnées complètes
    """
    language = detect_language(chunk.text)
    domain = domain_override or detect_domain(chunk.text)
    countries = detect_countries(chunk.text)
    chunk_id = generate_chunk_id(chunk.source, chunk.chunk_index)

    metadata = {
        **chunk.metadata,
        "language": language,
        "domain": domain,
        "countries": ",".join(countries),
        "word_count": len(chunk.text.split()),
        "char_count": len(chunk.text),
        "indexed_at": datetime.now().isoformat(),
    }

    return EnrichedChunk(
        id=chunk_id,
        text=chunk.text,
        metadata=metadata,
    )


def enrich_chunks(
    chunks: list[TextChunk],
    domain_override: str = None,
) -> list[EnrichedChunk]:
    """
    Enrichit une liste de chunks.

    Args:
        chunks: Chunks à enrichir
        domain_override: Domaine forcé pour tous

    Returns:
        Liste de EnrichedChunk
    """
    logger.info("Enriching chunks", count=len(chunks))

    enriched = [enrich_chunk(c, domain_override) for c in chunks]

    # Statistiques
    langs = {}
    domains = {}
    countries_flat = []
    for e in enriched:
        l = e.metadata.get("language", "unknown")
        d = e.metadata.get("domain", "general")
        c = e.metadata.get("countries", [])
        langs[l] = langs.get(l, 0) + 1
        domains[d] = domains.get(d, 0) + 1
        countries_flat.extend(c)

    logger.info(
        "Enrichment completed",
        total=len(enriched),
        languages=langs,
        domains=domains,
    )

    return enriched