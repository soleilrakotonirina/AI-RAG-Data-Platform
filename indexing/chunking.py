"""
indexing/chunking.py

Découpage intelligent de texte en chunks avec overlap.

Stratégie :
1. Découpe en paragraphes (double newline)
2. Regroupe jusqu'à chunk_size
3. Paragraphes longs → découpe par phrases
4. Overlap entre chunks consécutifs

Paramètres production :
- chunk_size : 600 chars (recommandé)
- overlap    : 100 chars (recommandé)
- min_length : 80 chars

Préparation Dagster : fonction pure, input/output typés.
"""

import re
import sys
from pathlib import Path
from dataclasses import dataclass, field

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from backend.app.core.logger import get_logger

logger = get_logger(__name__)

DEFAULT_CHUNK_SIZE = 600
DEFAULT_OVERLAP = 100
MIN_CHUNK_LENGTH = 80


@dataclass
class TextChunk:
    """Chunk de texte prêt pour embedding et indexation."""
    chunk_index: int
    text: str
    source: str
    page_number: int
    char_start: int
    char_end: int
    metadata: dict = field(default_factory=dict)


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    page_number: int = 1,
    extra_metadata: dict = None,
) -> list[TextChunk]:
    """
    Découpe un texte en chunks avec overlap, en respectant
    les limites naturelles (paragraphes, puis phrases).

    Args:
        text: Texte nettoyé à découper
        source: Nom du fichier source
        chunk_size: Taille max d'un chunk en caractères
        overlap: Chevauchement entre chunks consécutifs
        page_number: Numéro de page (si applicable)
        extra_metadata: Métadonnées additionnelles

    Returns:
        Liste de TextChunk
    """
    if not text or not text.strip():
        return []

    extra_metadata = extra_metadata or {}
    paragraphs = [p.strip() for p in re.split(r'\n\n+', text) if p.strip()]
    chunks = []
    current_text = ""
    current_start = 0
    global_pos = 0

    for para in paragraphs:
        # Paragraphe trop long → découpe par phrases
        if len(para) > chunk_size:
            if current_text.strip() and len(current_text.strip()) >= MIN_CHUNK_LENGTH:
                chunks.append(_build_chunk(
                    len(chunks), current_text.strip(),
                    current_start, global_pos,
                    source, page_number, extra_metadata,
                ))
            current_text = ""

            sentence_chunks = _split_by_sentences(
                text=para,
                source=source,
                page_number=page_number,
                chunk_size=chunk_size,
                overlap=overlap,
                extra_metadata=extra_metadata,
                start_index=len(chunks),
                offset=global_pos,
            )
            chunks.extend(sentence_chunks)
            global_pos += len(para) + 2
            continue

        candidate = (current_text + "\n\n" + para).strip() if current_text else para

        if len(candidate) > chunk_size and current_text:
            if len(current_text.strip()) >= MIN_CHUNK_LENGTH:
                chunks.append(_build_chunk(
                    len(chunks), current_text.strip(),
                    current_start, global_pos,
                    source, page_number, extra_metadata,
                ))
            overlap_text = current_text[-overlap:] if overlap > 0 else ""
            current_text = (overlap_text + "\n\n" + para).strip() if overlap_text else para
            current_start = max(0, global_pos - overlap)
        else:
            if not current_text:
                current_start = global_pos
            current_text = candidate

        global_pos += len(para) + 2

    # Flush dernier chunk
    if current_text.strip() and len(current_text.strip()) >= MIN_CHUNK_LENGTH:
        chunks.append(_build_chunk(
            len(chunks), current_text.strip(),
            current_start, global_pos,
            source, page_number, extra_metadata,
        ))

    logger.info(
        "Text chunked",
        source=source,
        chunks=len(chunks),
        avg_len=sum(len(c.text) for c in chunks) // max(len(chunks), 1),
        chunk_size=chunk_size,
        overlap=overlap,
    )

    return chunks


def _split_by_sentences(
    text: str,
    source: str,
    page_number: int,
    chunk_size: int,
    overlap: int,
    extra_metadata: dict,
    start_index: int,
    offset: int,
) -> list[TextChunk]:
    """Découpe un paragraphe long par frontières de phrases."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = ""
    current_start = offset

    for sentence in sentences:
        candidate = (current + " " + sentence).strip() if current else sentence

        if len(candidate) > chunk_size and current:
            if len(current.strip()) >= MIN_CHUNK_LENGTH:
                chunks.append(_build_chunk(
                    start_index + len(chunks),
                    current.strip(),
                    current_start,
                    current_start + len(current),
                    source, page_number, extra_metadata,
                ))
            current_start = max(0, current_start + len(current) - overlap)
            current = sentence
        else:
            current = candidate

    if current.strip() and len(current.strip()) >= MIN_CHUNK_LENGTH:
        chunks.append(_build_chunk(
            start_index + len(chunks),
            current.strip(),
            current_start,
            current_start + len(current),
            source, page_number, extra_metadata,
        ))

    return chunks


def _build_chunk(
    index: int, text: str, start: int, end: int,
    source: str, page: int, extra_metadata: dict,
) -> TextChunk:
    """Construit un objet TextChunk."""
    return TextChunk(
        chunk_index=index,
        text=text,
        source=source,
        page_number=page,
        char_start=start,
        char_end=end,
        metadata={
            "source": source,
            "page": page,
            "chunk_index": index,
            "char_start": start,
            "char_end": end,
            **extra_metadata,
        },
    )


def chunk_documents(
    documents: list,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[TextChunk]:
    """
    Découpe une liste de RawDocument en chunks.

    Args:
        documents: Liste de RawDocument
        chunk_size: Taille des chunks
        overlap: Overlap entre chunks

    Returns:
        Liste complète de TextChunk renumérotés globalement
    """
    all_chunks = []

    for doc in documents:
        doc_chunks = chunk_text(
            text=doc.text,
            source=doc.source,
            chunk_size=chunk_size,
            overlap=overlap,
            extra_metadata=doc.metadata,
        )
        all_chunks.extend(doc_chunks)

    # Renumérotation globale
    for i, chunk in enumerate(all_chunks):
        chunk.chunk_index = i
        chunk.metadata["chunk_index"] = i
        chunk.metadata["total_chunks"] = len(all_chunks)

    logger.info(
        "All documents chunked",
        total_docs=len(documents),
        total_chunks=len(all_chunks),
        avg_chunk_len=sum(len(c.text) for c in all_chunks) // max(len(all_chunks), 1),
    )

    return all_chunks