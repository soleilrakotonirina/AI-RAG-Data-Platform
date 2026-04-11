"""
ingestion/docling_pipeline.py

Pipeline d'ingestion principal basé sur Docling.
Version optimisée :
- Mode rapide (sans OCR, sans tableaux) pour PDFs textuels
- Sauvegarde en Markdown (.md) dans data/processed/
- Skip des fichiers déjà traités (hash-based)
"""

import sys
import re
import hashlib
import time
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from backend.app.core.logger import get_logger

logger = get_logger(__name__)

DATA_RAW = ROOT_DIR / "data" / "raw"
DATA_PROCESSED = ROOT_DIR / "data" / "processed"

SUPPORTED_FORMATS = {".pdf", ".docx", ".html", ".htm", ".txt", ".md"}


# ---------------------------------------------------------------------------
# Types de données
# ---------------------------------------------------------------------------

@dataclass
class RawDocument:
    """Document brut extrait par Docling."""
    id: str
    text: str
    source: str
    file_path: str
    file_type: str
    char_count: int
    word_count: int
    metadata: dict = field(default_factory=dict)


@dataclass
class IngestionResult:
    """Résultat de l'étape d'ingestion."""
    documents: list[RawDocument]
    total_files_scanned: int
    files_processed: int
    files_skipped: int
    files_failed: int
    duration_ms: float
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Utilitaires fichiers
# ---------------------------------------------------------------------------

def _get_file_hash(file_path: Path) -> str:
    """Hash SHA256 court d'un fichier."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()[:16]


def _get_processed_md_path(file_path: Path) -> Path:
    """Retourne le chemin .md dans data/processed/."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    clean_name = re.sub(r'[^a-zA-Z0-9_-]', '_', file_path.stem)[:60]
    return DATA_PROCESSED / f"{clean_name}.md"


def _get_processed_meta_path(file_path: Path) -> Path:
    """Retourne le chemin du fichier de métadonnées (.meta) associé."""
    md_path = _get_processed_md_path(file_path)
    return md_path.with_suffix(".meta")


def _is_already_processed(file_path: Path) -> bool:
    """
    Vérifie si un fichier a déjà été traité et n'a pas changé.
    Compare le hash stocké dans le .meta avec le hash actuel.
    """
    md_path = _get_processed_md_path(file_path)
    meta_path = _get_processed_meta_path(file_path)

    if not md_path.exists() or not meta_path.exists():
        return False

    try:
        stored_hash = meta_path.read_text(encoding="utf-8").strip()
        current_hash = _get_file_hash(file_path)
        return stored_hash == current_hash
    except Exception:
        return False


def _save_processed_md(doc: RawDocument) -> None:
    """
    Sauvegarde un document traité en Markdown dans data/processed/.

    Deux fichiers :
    - nom_fichier.md   : contenu texte en markdown
    - nom_fichier.meta : hash du fichier source (pour détecter changements)
    """
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    md_path = _get_processed_md_path(Path(doc.file_path))
    meta_path = _get_processed_meta_path(Path(doc.file_path))

    # En-tête YAML frontmatter
    frontmatter = f"""---
source: {doc.source}
file_type: {doc.file_type}
file_path: {doc.file_path}
char_count: {doc.char_count}
word_count: {doc.word_count}
processed_at: {doc.metadata.get('processed_at', '')}
file_hash: {doc.metadata.get('file_hash', '')}
---

"""

    # Sauvegarde markdown
    md_path.write_text(frontmatter + doc.text, encoding="utf-8")

    # Sauvegarde hash pour cache
    meta_path.write_text(doc.metadata.get("file_hash", ""), encoding="utf-8")

    logger.info(
        "Processed document saved (markdown)",
        file=doc.source,
        path=str(md_path),
        chars=doc.char_count,
    )


def _load_processed_md(file_path: Path) -> RawDocument:
    """Charge un document depuis son fichier .md dans data/processed/."""
    md_path = _get_processed_md_path(file_path)

    content = md_path.read_text(encoding="utf-8")

    # Extraction frontmatter YAML
    metadata = {}
    text = content
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            for line in parts[1].strip().split("\n"):
                if ":" in line:
                    k, v = line.split(":", 1)
                    metadata[k.strip()] = v.strip()
            text = parts[2].strip()

    return RawDocument(
        id=re.sub(r'[^a-zA-Z0-9_-]', '_', file_path.stem)[:50],
        text=text,
        source=metadata.get("source", file_path.name),
        file_path=metadata.get("file_path", str(file_path)),
        file_type=metadata.get("file_type", file_path.suffix.lstrip(".")),
        char_count=int(metadata.get("char_count", len(text))),
        word_count=int(metadata.get("word_count", len(text.split()))),
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Nettoie le texte extrait."""
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    lines = [line.strip() for line in text.split('\n')]
    return '\n'.join(lines).strip()


# ---------------------------------------------------------------------------
# Extraction Docling — MODE RAPIDE
# ---------------------------------------------------------------------------

def extract_with_docling(file_path: Path) -> str:
    """
    Extrait le texte via Docling en mode rapide.

    Optimisations CPU :
    - OCR désactivé (inutile pour PDFs textuels)
    - Analyse tableaux désactivée (lente)
    - Export Markdown (préserve structure)

    Pour activer OCR si nécessaire :
        pipeline_options.do_ocr = True
    """
    try:
        from docling.document_converter import DocumentConverter
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.datamodel.base_models import InputFormat
        from docling.document_converter import PdfFormatOption
    except ImportError:
        raise ImportError("pip install docling")

    logger.info("Extracting with Docling (fast/no-ocr)", file=file_path.name)

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = False
    pipeline_options.generate_page_images = False

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
            )
        }
    )

    result = converter.convert(str(file_path))
    text = result.document.export_to_markdown()

    if not text or not text.strip():
        raise ValueError(f"Empty text from {file_path.name}")

    return text


def extract_txt_fallback(file_path: Path) -> str:
    """Lecture directe pour TXT/MD."""
    for enc in ["utf-8", "latin-1", "cp1252"]:
        try:
            return file_path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Cannot decode {file_path.name}")


# ---------------------------------------------------------------------------
# Traitement fichier unique
# ---------------------------------------------------------------------------

def process_single_file(file_path: Path, force: bool = False) -> RawDocument | None:
    """
    Traite un fichier : extraction → cleaning → sauvegarde .md

    Skip si déjà traité et inchangé (sauf force=True).
    """
    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        return None

    # Skip si déjà traité
    if not force and _is_already_processed(file_path):
        logger.info("Already processed — loading from cache", file=file_path.name)
        return _load_processed_md(file_path)

    logger.info("Processing file", file=file_path.name)

    try:
        if suffix in {".txt", ".md"}:
            raw_text = extract_txt_fallback(file_path)
        else:
            raw_text = extract_with_docling(file_path)

        clean = clean_text(raw_text)

        if len(clean) < 100:
            logger.warning("Too short after cleaning — skipping", file=file_path.name)
            return None

        file_hash = _get_file_hash(file_path)
        doc_id = re.sub(r'[^a-zA-Z0-9_-]', '_', file_path.stem)[:50]

        doc = RawDocument(
            id=doc_id,
            text=clean,
            source=file_path.name,
            file_path=str(file_path),
            file_type=suffix.lstrip("."),
            char_count=len(clean),
            word_count=len(clean.split()),
            metadata={
                "source": file_path.name,
                "file_type": suffix.lstrip("."),
                "file_path": str(file_path),
                "file_hash": file_hash,
                "processed_at": datetime.now().isoformat(),
                "char_count": len(clean),
                "word_count": len(clean.split()),
            },
        )

        _save_processed_md(doc)

        logger.info(
            "File processed successfully",
            file=file_path.name,
            chars=len(clean),
            words=len(clean.split()),
        )

        return doc

    except Exception as e:
        logger.error("File processing failed", file=file_path.name, error=str(e))
        return None


# ---------------------------------------------------------------------------
# Pipeline scan complet
# ---------------------------------------------------------------------------

def scan_and_process(
    data_dir: Path = None,
    force: bool = False,
) -> IngestionResult:
    """
    Scanne data/raw/ et traite tous les fichiers supportés.

    Comportement :
    - Skip fichiers déjà traités (vérification hash)
    - Continue en cas d'erreur (log + skip)
    - Sauvegarde chaque doc en .md dans data/processed/
    """
    start_time = time.time()
    data_dir = data_dir or DATA_RAW

    if not data_dir.exists():
        raise FileNotFoundError(f"Répertoire source introuvable : {data_dir}")

    all_files = [
        f for f in data_dir.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_FORMATS
    ]

    if not all_files:
        logger.warning("No supported files found", directory=str(data_dir))
        return IngestionResult(
            documents=[], total_files_scanned=0,
            files_processed=0, files_skipped=0,
            files_failed=0, duration_ms=0.0,
        )

    logger.info(
        "Scanning data/raw/",
        total_files=len(all_files),
        files=[f.name for f in all_files],
        force_reprocess=force,
    )

    documents = []
    files_processed = 0
    files_skipped = 0
    files_failed = 0
    errors = []

    for file_path in all_files:
        was_cached = not force and _is_already_processed(file_path)
        result = process_single_file(file_path, force=force)

        if result is None:
            files_failed += 1
            errors.append(file_path.name)
        elif was_cached:
            documents.append(result)
            files_skipped += 1
        else:
            documents.append(result)
            files_processed += 1

    duration_ms = (time.time() - start_time) * 1000

    logger.info(
        "Scan completed",
        total=len(all_files),
        processed=files_processed,
        skipped_cached=files_skipped,
        failed=files_failed,
        duration_ms=round(duration_ms),
    )

    return IngestionResult(
        documents=documents,
        total_files_scanned=len(all_files),
        files_processed=files_processed,
        files_skipped=files_skipped,
        files_failed=files_failed,
        duration_ms=duration_ms,
        errors=errors,
    )


def load_all_processed() -> list[RawDocument]:
    """Charge tous les documents depuis data/processed/*.md"""
    if not DATA_PROCESSED.exists():
        return []

    documents = []
    for md_file in DATA_PROCESSED.glob("*.md"):
        try:
            content = md_file.read_text(encoding="utf-8")
            metadata = {}
            text = content
            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    for line in parts[1].strip().split("\n"):
                        if ":" in line:
                            k, v = line.split(":", 1)
                            metadata[k.strip()] = v.strip()
                    text = parts[2].strip()

            doc_id = re.sub(r'[^a-zA-Z0-9_-]', '_', md_file.stem)[:50]
            documents.append(RawDocument(
                id=doc_id,
                text=text,
                source=metadata.get("source", md_file.name),
                file_path=metadata.get("file_path", str(md_file)),
                file_type=metadata.get("file_type", "md"),
                char_count=int(metadata.get("char_count", len(text))),
                word_count=int(metadata.get("word_count", len(text.split()))),
                metadata=metadata,
            ))
        except Exception as e:
            logger.warning("Failed to load processed md", file=md_file.name, error=str(e))

    logger.info("Loaded processed documents", count=len(documents))
    return documents