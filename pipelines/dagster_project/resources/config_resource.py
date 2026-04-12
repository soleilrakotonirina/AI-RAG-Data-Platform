"""
pipelines/dagster_project/resources/config_resource.py

Ressource de configuration globale du pipeline.
Centralise tous les paramètres configurables.
"""

from dagster import ConfigurableResource


class PipelineConfigResource(ConfigurableResource):
    """
    Configuration centrale du pipeline.

    Paramètres injectables via Dagster config system.
    Permet de modifier le comportement sans changer le code.
    """

    # Mode d'exécution
    mode: str = "incremental"          # "full", "processed", "incremental"
    reset_chromadb: bool = False       # Vider ChromaDB avant indexation

    # Chemins
    data_raw_path: str = "data/raw"
    data_processed_path: str = "data/processed"
    data_embeddings_path: str = "data/embeddings"

    # Chunking
    chunk_size: int = 600
    overlap: int = 100
    min_chunk_length: int = 80

    # Embeddings
    embedding_model: str = "nvidia/llama-nemotron-embed-vl-1b-v2:free"
    embedding_batch_size: int = 50
    use_disk_cache: bool = True

    # Indexation
    indexing_batch_size: int = 50

    # Domaine (optionnel)
    domain_override: str = ""          # "" = détection automatique