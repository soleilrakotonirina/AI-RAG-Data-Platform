"""
scripts/ingest_data.py

CLI du pipeline d'ingestion complet.

Pipeline :
data/raw/ → Docling → data/processed/ → Chunking → Enrichment
         → data/embeddings/ → ChromaDB

Usage :
    python scripts/ingest_data.py
    python scripts/ingest_data.py --reset
    python scripts/ingest_data.py --force
    python scripts/ingest_data.py --chunk-size 800 --overlap 100
    python scripts/ingest_data.py --from-processed  # skip Docling
    python scripts/ingest_data.py --domain economics

Options :
    --reset           Vider ChromaDB avant ingestion
    --force           Retraiter même les fichiers déjà traités
    --from-processed  Charger depuis data/processed/ (skip Docling)
    --chunk-size N    Taille max des chunks (défaut: 600)
    --overlap N       Overlap entre chunks (défaut: 100)
    --domain STR      Domaine forcé pour tous les documents
    --batch-size N    Taille des batches d'indexation (défaut: 50)
"""

import sys
import argparse
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from backend.app.core.logger import configure_logging, get_logger
from backend.app.db.vector_store import VectorStore
from ingestion.docling_pipeline import scan_and_process, load_all_processed
from indexing.chunking import chunk_documents
from indexing.enrichment import enrich_chunks
from indexing.embeddings import generate_embeddings
from indexing.index_builder import build_index

configure_logging()
logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pipeline d'ingestion RAG — Docling + ChromaDB"
    )
    parser.add_argument("--reset", action="store_true",
                        help="Vider ChromaDB avant ingestion")
    parser.add_argument("--force", action="store_true",
                        help="Retraiter tous les fichiers (ignore cache)")
    parser.add_argument("--from-processed", action="store_true",
                        help="Charger depuis data/processed/ (skip Docling)")
    parser.add_argument("--chunk-size", type=int, default=600)
    parser.add_argument("--overlap", type=int, default=100)
    parser.add_argument("--domain", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=50)
    return parser.parse_args()


def main():
    args = parse_args()
    pipeline_start = time.time()

    logger.info("=" * 55)
    logger.info("  INGESTION PIPELINE — RAG Agent System")
    logger.info("=" * 55)
    logger.info(
        "Configuration",
        reset=args.reset,
        force=args.force,
        from_processed=args.from_processed,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        domain=args.domain,
    )

    # ------------------------------------------------------------------
    # Reset ChromaDB si demandé
    # ------------------------------------------------------------------
    if args.reset:
        store = VectorStore()
        store.reset()
        logger.warning("ChromaDB collection reset")

    # ------------------------------------------------------------------
    # Étape 1 : Ingestion (Docling ou cache processed)
    # ------------------------------------------------------------------
    logger.info("--- Étape 1/5 : Ingestion ---")

    if args.from_processed:
        documents = load_all_processed()
        logger.info("Loaded from data/processed/", count=len(documents))
    else:
        result = scan_and_process(force=args.force)
        documents = result.documents
        logger.info(
            "Ingestion completed",
            scanned=result.total_files_scanned,
            processed=result.files_processed,
            skipped=result.files_skipped,
            failed=result.files_failed,
        )

    if not documents:
        logger.error(
            "Aucun document chargé. "
            "Vérifier data/raw/ et installer docling : pip install docling"
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Étape 2 : Chunking
    # ------------------------------------------------------------------
    logger.info("--- Étape 2/5 : Chunking ---")

    chunks = chunk_documents(
        documents=documents,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )

    logger.info("Chunking done", total_chunks=len(chunks))

    # ------------------------------------------------------------------
    # Étape 3 : Enrichissement
    # ------------------------------------------------------------------
    logger.info("--- Étape 3/5 : Enrichissement ---")

    enriched = enrich_chunks(chunks, domain_override=args.domain)
    logger.info("Enrichment done", count=len(enriched))

    # ------------------------------------------------------------------
    # Étape 4 : Embeddings
    # ------------------------------------------------------------------
    logger.info("--- Étape 4/5 : Embeddings ---")

    embedded = generate_embeddings(
        enriched_chunks=enriched,
        use_disk_cache=True,
        batch_size=args.batch_size,
    )

    logger.info("Embeddings done", count=len(embedded))

    # ------------------------------------------------------------------
    # Étape 5 : Indexation ChromaDB
    # ------------------------------------------------------------------
    logger.info("--- Étape 5/5 : Indexation ChromaDB ---")

    report = build_index(
        embedded_chunks=embedded,
        batch_size=args.batch_size,
    )

    # ------------------------------------------------------------------
    # Rapport final
    # ------------------------------------------------------------------
    total_duration = (time.time() - pipeline_start) * 1000

    logger.info("=" * 55)
    logger.info("  RAPPORT FINAL")
    logger.info("=" * 55)
    logger.info(f"Documents traités    : {len(documents)}")
    logger.info(f"Chunks générés       : {len(chunks)}")
    logger.info(f"Chunks enrichis      : {len(enriched)}")
    logger.info(f"Chunks vectorisés    : {len(embedded)}")
    logger.info(f"Chunks indexés       : {report.indexed}")
    logger.info(f"Erreurs indexation   : {report.failed}")
    logger.info(f"Total ChromaDB       : {report.final_db_count}")
    logger.info(f"Durée totale         : {total_duration:.0f}ms")
    logger.info(f"Fichiers traités     : {[d.source for d in documents]}")
    logger.info("=" * 55)

    if report.indexed == 0:
        logger.error("Aucun chunk indexé — vérifier les erreurs ci-dessus")
        sys.exit(1)

    logger.info("Pipeline ingestion terminé avec succès")


if __name__ == "__main__":
    main()