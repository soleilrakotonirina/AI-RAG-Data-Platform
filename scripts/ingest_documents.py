"""
scripts/ingest_documents.py

Ingestion des documents PDF depuis data/raw/ vers ChromaDB.

Utilisation :
    python scripts/ingest_documents.py
    python scripts/ingest_documents.py --reset      # vide la collection avant
    python scripts/ingest_documents.py --chunk-size 800
    python scripts/ingest_documents.py --domain economics

Prérequis :
    pip install pypdf
    Placer les PDFs dans data/raw/
"""

import sys
import argparse
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from backend.app.core.logger import configure_logging, get_logger
from backend.app.db.vector_store import VectorStore
from indexing.embeddings import load_documents_from_raw, index_documents

configure_logging()
logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ingestion des PDFs depuis data/raw/ vers ChromaDB"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Vider la collection ChromaDB avant l'ingestion",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Taille des chunks en caractères (défaut: 500)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=50,
        help="Overlap entre chunks (défaut: 50)",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="economics",
        help="Domaine des documents (défaut: economics)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("=== Ingestion Pipeline ===")
    logger.info(
        "Configuration",
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        domain=args.domain,
        reset=args.reset,
    )

    # Reset si demandé
    store = VectorStore()
    if args.reset:
        logger.warning("Resetting ChromaDB collection")
        store.reset()
        logger.info("Collection reset OK")
    else:
        current_count = store.count()
        logger.info("Current documents in DB", count=current_count)

    # Chargement et chunking des PDFs
    logger.info("Loading PDFs from data/raw/")
    raw_documents = load_documents_from_raw(
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        domain=args.domain,
    )

    logger.info(
        "Documents ready for indexing",
        total=len(raw_documents),
        sample_id=raw_documents[0]["id"] if raw_documents else "none",
    )

    # Indexation
    logger.info("Starting indexation (this may take several minutes)...")
    index_documents(raw_documents)

    # Validation finale
    final_count = store.count()
    logger.info(
        "=== Ingestion terminée ===",
        documents_indexed=final_count,
    )


if __name__ == "__main__":
    main()