"""
indexing/embeddings.py

Pipeline d'embedding offline.
Phase 6 : support des vrais documents PDF depuis data/raw/

Structure des documents :
{
    "id": "...",
    "text": "...",
    "metadata": {
        "source": "nom_fichier.pdf",
        "language": "en",
        "domain": "economics",
        "page": 1
    }
}
"""

import sys
import re
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from backend.app.services.embedding_service import embed_batch
from backend.app.db.vector_store import VectorStore, Document
from backend.app.core.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Chargement et chunking des PDFs
# ---------------------------------------------------------------------------

def load_pdf_text(pdf_path: Path) -> str:
    """
    Extrait le texte brut d'un PDF.

    Args:
        pdf_path: Chemin vers le fichier PDF

    Returns:
        Texte extrait du PDF
    """
    try:
        import pypdf
        reader = pypdf.PdfReader(str(pdf_path))
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text.strip())
        full_text = "\n\n".join(pages)
        logger.info(
            "PDF loaded",
            file=pdf_path.name,
            pages=len(reader.pages),
            chars=len(full_text),
        )
        return full_text
    except ImportError:
        raise ImportError(
            "pypdf non installé. Lancer : pip install pypdf"
        )
    except Exception as e:
        logger.error("PDF load error", file=str(pdf_path), error=str(e))
        raise


def clean_text(text: str) -> str:
    """
    Nettoie le texte extrait d'un PDF.
    - Supprime les espaces multiples
    - Supprime les caractères de contrôle
    - Normalise les sauts de ligne
    """
    # Supprime caractères de contrôle sauf newlines
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    # Normalise espaces multiples
    text = re.sub(r' +', ' ', text)
    # Normalise sauts de ligne multiples
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Supprime lignes vides en début/fin
    text = text.strip()
    return text


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[str]:
    """
    Découpe le texte en chunks avec overlap.

    Stratégie :
    - Découpe par paragraphes en priorité
    - Respecte chunk_size en caractères
    - Overlap pour conserver le contexte entre chunks

    Args:
        text: Texte à découper
        chunk_size: Taille max d'un chunk en caractères
        overlap: Chevauchement entre chunks

    Returns:
        Liste de chunks texte
    """
    # Découpe par paragraphes
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

    chunks = []
    current_chunk = ""

    for para in paragraphs:
        # Si le paragraphe seul dépasse chunk_size, le découper
        if len(para) > chunk_size:
            words = para.split()
            temp = ""
            for word in words:
                if len(temp) + len(word) + 1 > chunk_size:
                    if temp:
                        chunks.append(temp.strip())
                    temp = word
                else:
                    temp = temp + " " + word if temp else word
            if temp:
                current_chunk = temp
            continue

        # Ajouter le paragraphe au chunk courant
        if len(current_chunk) + len(para) + 2 > chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
            # Overlap : reprendre les derniers caractères
            if overlap > 0 and current_chunk:
                current_chunk = current_chunk[-overlap:] + "\n\n" + para
            else:
                current_chunk = para
        else:
            current_chunk = current_chunk + "\n\n" + para if current_chunk else para

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    # Filtrer les chunks trop courts
    chunks = [c for c in chunks if len(c) > 50]

    logger.info(
        "Text chunked",
        total_chunks=len(chunks),
        chunk_size=chunk_size,
        overlap=overlap,
        avg_chunk_len=sum(len(c) for c in chunks) // max(len(chunks), 1),
    )

    return chunks


def load_documents_from_raw(
    data_dir: Path = None,
    chunk_size: int = 500,
    overlap: int = 50,
    domain: str = "general",
) -> list[dict]:
    """
    Charge tous les PDFs depuis data/raw/ et les transforme
    en documents structurés avec chunks.

    Structure de sortie :
    {
        "id": "climate_report_chunk_0",
        "text": "...",
        "metadata": {
            "source": "Climate-Development-Report.pdf",
            "language": "en",
            "domain": "economics",
            "chunk_index": 0,
            "total_chunks": 42
        }
    }

    Args:
        data_dir: Répertoire data/raw/ (défaut : auto-détecté)
        chunk_size: Taille des chunks
        overlap: Overlap entre chunks
        domain: Domaine des documents

    Returns:
        Liste de documents prêts pour indexation
    """
    if data_dir is None:
        data_dir = ROOT_DIR / "data" / "raw"

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Répertoire data/raw/ introuvable : {data_dir}\n"
            "Créer le répertoire et y placer les PDFs."
        )

    pdf_files = list(data_dir.glob("*.pdf"))
    if not pdf_files:
        raise ValueError(
            f"Aucun PDF trouvé dans {data_dir}\n"
            "Placer les PDFs dans data/raw/"
        )

    logger.info(
        "Loading PDFs from data/raw/",
        pdf_count=len(pdf_files),
        files=[f.name for f in pdf_files],
    )

    all_documents = []

    for pdf_path in pdf_files:
        try:
            # Extraction texte
            raw_text = load_pdf_text(pdf_path)
            clean = clean_text(raw_text)

            # Chunking
            chunks = chunk_text(clean, chunk_size=chunk_size, overlap=overlap)

            # Identifiant base (nom fichier sans extension, nettoyé)
            base_id = re.sub(r'[^a-zA-Z0-9_-]', '_', pdf_path.stem)[:40]

            # Construction documents structurés
            for i, chunk in enumerate(chunks):
                doc = {
                    "id": f"{base_id}_chunk_{i}",
                    "text": chunk,
                    "metadata": {
                        "source": pdf_path.name,
                        "language": "en",
                        "domain": domain,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "file_path": str(pdf_path),
                    },
                }
                all_documents.append(doc)

            logger.info(
                "PDF processed",
                file=pdf_path.name,
                chunks=len(chunks),
            )

        except Exception as e:
            logger.error(
                "Failed to process PDF",
                file=pdf_path.name,
                error=str(e),
            )
            continue

    logger.info(
        "All PDFs loaded",
        total_documents=len(all_documents),
        pdf_files=len(pdf_files),
    )

    return all_documents


# ---------------------------------------------------------------------------
# Pipeline d'indexation
# ---------------------------------------------------------------------------

def build_documents_with_embeddings(
    raw_documents: list[dict],
) -> list[Document]:
    """
    Transforme des documents bruts en objets Document avec embeddings.

    Args:
        raw_documents: Liste de dicts avec id, text, metadata

    Returns:
        Liste de Document avec embeddings réels
    """
    for i, doc in enumerate(raw_documents):
        if "id" not in doc:
            raise ValueError(f"Document {i} manque le champ 'id'")
        if "text" not in doc:
            raise ValueError(f"Document {i} manque le champ 'text'")
        if not doc["text"].strip():
            raise ValueError(f"Document '{doc['id']}' a un texte vide")

    texts = [doc["text"] for doc in raw_documents]

    logger.info(
        "Generating embeddings",
        document_count=len(texts),
    )

    embeddings = embed_batch(texts, use_cache=True, show_progress=True)

    documents = []
    for raw_doc, embedding in zip(raw_documents, embeddings):
        documents.append(Document(
            id=raw_doc["id"],
            text=raw_doc["text"],
            metadata=raw_doc.get("metadata", {}),
            embedding=embedding,
        ))

    logger.info(
        "Documents with embeddings ready",
        count=len(documents),
        embedding_dim=len(embeddings[0]) if embeddings else 0,
    )

    return documents


def index_documents(raw_documents: list[dict]) -> None:
    """
    Pipeline complet : documents bruts → embeddings → ChromaDB.

    Args:
        raw_documents: Liste de dicts avec id, text, metadata
    """
    logger.info(
        "Starting indexing pipeline",
        document_count=len(raw_documents),
    )

    documents = build_documents_with_embeddings(raw_documents)
    store = VectorStore()
    store.add_documents(documents)

    logger.info(
        "Indexing pipeline completed",
        indexed=len(documents),
        total_in_db=store.count(),
    )


def search_documents(
    query_text: str,
    top_k: int = 5,
) -> list:
    """
    Recherche sémantique complète.

    Args:
        query_text: Requête en langage naturel (français ou anglais)
        top_k: Nombre de résultats

    Returns:
        Liste de SearchResult
    """
    from backend.app.services.embedding_service import embed_text

    logger.info("Semantic search", query=query_text, top_k=top_k)
    query_embedding = embed_text(query_text, use_cache=True)
    store = VectorStore()
    return store.search(
        query_text=query_text,
        top_k=top_k,
        query_embedding=query_embedding,
    )