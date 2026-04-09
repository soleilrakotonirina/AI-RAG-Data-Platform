"""
scripts/test_rag_real.py

Test du pipeline RAG avec les vrais documents PDF.
Documents : rapports économiques (anglais)
Questions  : français
Réponses   : français

Prérequis :
- PDFs dans data/raw/
- python scripts/ingest_documents.py --reset (à faire UNE fois)
- OPENROUTER_API_KEY configurée dans .env

Lancer :
    python scripts/test_rag_real.py
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from backend.app.core.logger import configure_logging, get_logger
from backend.app.rag.chain import RAGPipeline
from backend.app.db.vector_store import VectorStore

configure_logging()
logger = get_logger(__name__)


def check_prerequisites():
    """Vérifie que ChromaDB contient des documents."""
    store = VectorStore()
    count = store.count()
    if count == 0:
        logger.error(
            "ChromaDB est vide. "
            "Lancer d'abord : python scripts/ingest_documents.py --reset"
        )
        sys.exit(1)
    logger.info("Documents dans ChromaDB", count=count)
    return count


def divider(title: str):
    logger.info("=" * 55)
    logger.info(f"  {title}")
    logger.info("=" * 55)


def run_tests():
    logger.info("=== Test RAG — Documents réels (EN→FR) ===")

    doc_count = check_prerequisites()

    pipeline = RAGPipeline(
        top_k=4,
        max_chars_per_doc=700,
        max_total_chars=5000,
        use_mmr=True,
        mmr_lambda=0.7,
        adaptive_k=True,
        use_cache=True,
        temperature=0.2,
        max_tokens=1024,
    )

    # ------------------------------------------------------------------
    # Questions sur les documents réels
    # ------------------------------------------------------------------
    questions = [
        {
            "question": "Quels sont les principaux défis économiques "
                        "de Madagascar selon les rapports ?",
            "expected_topics": ["madagascar", "economic", "development"],
        },
        {
            "question": "Comment le changement climatique affecte-t-il "
                        "le développement économique selon les rapports ?",
            "expected_topics": ["climate", "development", "impact"],
        },
        {
            "question": "Quels indicateurs économiques sont utilisés "
                        "pour mesurer la croissance du PIB ?",
            "expected_topics": ["gdp", "indicators", "growth"],
        },
        {
            "question": "Quelle est la situation de la pauvreté "
                        "dans les pays en développement ?",
            "expected_topics": ["poverty", "developing", "countries"],
        },
        {
            "question": "Comment l'urbanisation affecte-t-elle "
                        "l'économie de Madagascar ?",
            "expected_topics": ["urbanization", "madagascar", "economy"],
        },
    ]

    results_summary = []

    for i, q_data in enumerate(questions, 1):
        question = q_data["question"]
        divider(f"Question {i}/{len(questions)}")
        logger.info(f"Question : {question}")

        try:
            result = pipeline.run(question)

            # Validation langue de réponse
            french_markers = [
                "le ", "la ", "les ", "de ", "du ", "des ",
                "est ", "sont ", "dans ", "pour ", "avec ",
                "qui ", "que ", "une ", "un ",
            ]
            french_count = sum(
                1 for marker in french_markers
                if marker in result.answer.lower()
            )
            is_french = french_count >= 3

            logger.info(
                "Résultat",
                document_count=result.document_count,
                quality_score=round(result.quality_score, 3),
                confidence=result.confidence_level,
                answer_length=len(result.answer),
                is_french=is_french,
                duration_ms=round(result.total_duration_ms),
            )

            logger.info("\nRéponse :\n" + result.answer)

            if result.sources:
                logger.info("\nSources utilisées :")
                for src in result.sources:
                    logger.info(
                        f"  {src['id'][:50]} | "
                        f"source={src['metadata'].get('source', '?')[:40]} | "
                        f"score={src['score']:.4f}"
                    )

            results_summary.append({
                "question": question[:60],
                "document_count": result.document_count,
                "quality_score": result.quality_score,
                "confidence": result.confidence_level,
                "is_french": is_french,
                "duration_ms": result.total_duration_ms,
            })

            # Assertion langue
            assert is_french, (
                f"La réponse devrait être en français.\n"
                f"Réponse : {result.answer[:200]}"
            )

        except Exception as e:
            logger.error(f"Erreur question {i}", error=str(e))
            results_summary.append({
                "question": question[:60],
                "error": str(e),
            })

    # ------------------------------------------------------------------
    # Résumé
    # ------------------------------------------------------------------
    divider("RÉSUMÉ DES TESTS")

    logger.info(f"Documents dans ChromaDB : {doc_count}")
    logger.info(f"Questions testées : {len(questions)}")

    successful = [r for r in results_summary if "error" not in r]
    french_ok = [r for r in successful if r.get("is_french")]

    logger.info(f"Questions réussies : {len(successful)}/{len(questions)}")
    logger.info(f"Réponses en français : {len(french_ok)}/{len(successful)}")

    if successful:
        avg_quality = sum(r["quality_score"] for r in successful) / len(successful)
        avg_duration = sum(r["duration_ms"] for r in successful) / len(successful)
        logger.info(f"Score qualité moyen : {avg_quality:.3f}")
        logger.info(f"Durée moyenne : {avg_duration:.0f}ms")

    logger.info("=== TEST RAG RÉEL TERMINÉ ===")


if __name__ == "__main__":
    run_tests()