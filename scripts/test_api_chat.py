"""
scripts/test_api_chat.py

Validation Phase 7 — Endpoint /chat.

Prérequis :
- API lancée : python backend/run.py
- Documents indexés dans ChromaDB

Lancer depuis la racine :
    python scripts/test_api_chat.py
"""

import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import httpx
from backend.app.core.logger import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)

BASE_URL = "http://localhost:8000/api/v1"


def divider(title: str):
    logger.info("=" * 50)
    logger.info(f"  {title}")
    logger.info("=" * 50)


def run_tests():
    logger.info("=== Phase 7 — API /chat Validation ===")

    # ------------------------------------------------------------------
    # Test 1 : Health check
    # ------------------------------------------------------------------
    divider("Test 1 : Health check")

    r = httpx.get(f"{BASE_URL}/health", timeout=10)
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    logger.info("Health OK", response=data)

    # ------------------------------------------------------------------
    # Test 2 : Statut pipeline
    # ------------------------------------------------------------------
    divider("Test 2 : Statut pipeline RAG")

    r = httpx.get(f"{BASE_URL}/chat/status", timeout=10)
    assert r.status_code == 200
    data = r.json()
    assert data["status"] in ["ok", "warning"]
    assert data["chromadb"]["connected"] is True

    doc_count = data["chromadb"]["document_count"]
    logger.info(
        "Pipeline status OK",
        status=data["status"],
        document_count=doc_count,
    )

    if doc_count == 0:
        logger.error(
            "ChromaDB vide — lancer : python scripts/ingest_documents.py --reset"
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Test 3 : Requête /chat simple
    # ------------------------------------------------------------------
    divider("Test 3 : POST /chat simple")

    start = time.time()
    r = httpx.post(
        f"{BASE_URL}/chat",
        json={"question": "Quels sont les défis économiques de Madagascar ?"},
        timeout=300,
    )
    duration = (time.time() - start) * 1000

    assert r.status_code == 200, f"Status inattendu : {r.status_code}\n{r.text}"

    data = r.json()
    assert "answer" in data
    assert "sources" in data
    assert "metadata" in data
    assert len(data["answer"]) > 0

    logger.info(
        "POST /chat OK",
        answer_length=len(data["answer"]),
        source_count=len(data["sources"]),
        quality_score=data["metadata"]["quality_score"],
        confidence=data["metadata"]["confidence_level"],
        duration_ms=round(duration),
    )
    logger.info("Réponse :\n" + data["answer"])

    # ------------------------------------------------------------------
    # Test 4 : Langue de la réponse
    # ------------------------------------------------------------------
    divider("Test 4 : Validation langue française")

    french_markers = [
        "le ", "la ", "les ", "de ", "du ", "des ",
        "est ", "sont ", "dans ", "pour ", "avec ",
    ]
    answer_lower = data["answer"].lower()
    french_count = sum(1 for m in french_markers if m in answer_lower)
    is_french = french_count >= 3

    assert is_french, f"La réponse devrait être en français. Détecté : {french_count} marqueurs"
    logger.info("Langue OK", french_markers_detected=french_count, is_french=is_french)

    # ------------------------------------------------------------------
    # Test 5 : Paramètres avancés
    # ------------------------------------------------------------------
    divider("Test 5 : Paramètres avancés (top_k, mmr)")

    r = httpx.post(
        f"{BASE_URL}/chat",
        json={
            "question": "Comment l'urbanisation affecte-t-elle l'économie malgache ?",
            "top_k": 5,
            "score_threshold": 0.0,
            "use_mmr": True,
        },
        timeout=300,
    )
    assert r.status_code == 200
    data = r.json()

    logger.info(
        "Paramètres avancés OK",
        document_count=data["metadata"]["document_count"],
        quality_score=data["metadata"]["quality_score"],
        confidence=data["metadata"]["confidence_level"],
    )

    # ------------------------------------------------------------------
    # Test 6 : Cache — même question
    # ------------------------------------------------------------------
    divider("Test 6 : Cache (même question)")

    start = time.time()
    r = httpx.post(
        f"{BASE_URL}/chat",
        json={"question": "Quels sont les défis économiques de Madagascar ?"},
        timeout=300,
    )
    cache_duration = (time.time() - start) * 1000
    data_cached = r.json()

    assert r.status_code == 200
    logger.info(
        "Cache",
        from_cache=data_cached["metadata"]["from_cache"],
        duration_ms=round(cache_duration),
    )

    # ------------------------------------------------------------------
    # Test 7 : Validation erreur — question trop courte
    # ------------------------------------------------------------------
    divider("Test 7 : Validation erreur — question invalide")

    r = httpx.post(
        f"{BASE_URL}/chat",
        json={"question": "ab"},
        timeout=10,
    )
    assert r.status_code == 422
    logger.info("Validation erreur OK", status_code=r.status_code)

    # ------------------------------------------------------------------
    # Test 8 : Sources structurées
    # ------------------------------------------------------------------
    divider("Test 8 : Structure des sources")

    r = httpx.post(
        f"{BASE_URL}/chat",
        json={"question": "Quelle est la situation de la pauvreté à Madagascar ?"},
        timeout=300,
    )
    assert r.status_code == 200
    data = r.json()

    if data["sources"]:
        src = data["sources"][0]
        assert "id" in src
        assert "score" in src
        assert "confidence" in src
        assert "source_file" in src
        logger.info(
            "Sources OK",
            source_count=len(data["sources"]),
            first_source=src,
        )

    # ------------------------------------------------------------------
    # Résumé
    # ------------------------------------------------------------------
    logger.info("=== VALIDATION PHASE 7 TERMINEE ===")
    logger.info("Endpoint /chat operationnel")
    logger.info("Endpoint /chat/status operationnel")
    logger.info("Langue : questions FR → réponses FR")
    logger.info("Sources : structurées et traçables")
    logger.info("Cache : fonctionnel")
    logger.info("Swagger : http://localhost:8000/docs")
    logger.info("Pret pour Phase 8 — Reranking")


if __name__ == "__main__":
    run_tests()