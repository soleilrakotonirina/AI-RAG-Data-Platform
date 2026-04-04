"""
backend/app/db/chroma_client.py

Couche de connexion ChromaDB.
Responsabilités :
- Initialiser le client ChromaDB en mode embedded (local, sans Docker)
- Créer ou récupérer la collection cible
- Exposer un singleton ChromaClient réutilisable

Ce fichier ne contient AUCUNE logique métier.
Toutes les opérations sur les données sont dans vector_store.py.
"""

from pathlib import Path
from functools import lru_cache

import chromadb
from chromadb.config import Settings as ChromaSettings

from backend.app.core.settings import get_settings
from backend.app.core.logger import get_logger

logger = get_logger(__name__)


def get_chroma_client() -> chromadb.ClientAPI:
    """
    Initialise et retourne un client ChromaDB en mode embedded (persistant).

    Le client est configuré pour persister les données dans data/chromadb/
    à la racine du projet. Ce répertoire est créé automatiquement si absent.

    Returns:
        Instance chromadb.ClientAPI prête à l'emploi
    """
    settings = get_settings()

    # Chemin de persistance : racine_projet/data/chromadb/
    root_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
    persist_dir = root_dir / "data" / "chromadb"
    persist_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Initializing ChromaDB client",
        mode="embedded",
        persist_path=str(persist_dir),
    )

    client = chromadb.PersistentClient(
        path=str(persist_dir),
        settings=ChromaSettings(
            anonymized_telemetry=False,
            allow_reset=True,
        ),
    )

    logger.info("ChromaDB client initialized successfully")
    return client


def get_or_create_collection(
    client: chromadb.ClientAPI,
    collection_name: str = None,
) -> chromadb.Collection:
    """
    Récupère une collection existante ou la crée si elle n'existe pas.
    Comportement idempotent : safe à appeler plusieurs fois.

    Args:
        client: Instance ChromaDB active
        collection_name: Nom de la collection (défaut : depuis .env)

    Returns:
        Instance chromadb.Collection prête à l'emploi
    """
    settings = get_settings()
    name = collection_name or settings.chroma_collection_name

    collection = client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},  # similarité cosine (standard RAG)
    )

    logger.info(
        "Collection ready",
        collection_name=name,
        document_count=collection.count(),
    )

    return collection


class ChromaDBClient:
    """
    Wrapper singleton autour du client ChromaDB et de sa collection principale.

    Usage :
        db = ChromaDBClient()
        collection = db.collection
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.client = get_chroma_client()
        self.collection = get_or_create_collection(self.client)
        self._initialized = True

        logger.info("ChromaDBClient singleton ready")

    def reset_collection(self, collection_name: str = None) -> None:
        """
        Supprime et recrée la collection. Utile pour les tests.

        Args:
            collection_name: Nom de la collection (défaut : depuis .env)
        """
        settings = get_settings()
        name = collection_name or settings.chroma_collection_name

        self.client.delete_collection(name=name)
        logger.warning("Collection deleted", collection_name=name)

        self.collection = get_or_create_collection(self.client, name)
        logger.info("Collection recreated", collection_name=name)