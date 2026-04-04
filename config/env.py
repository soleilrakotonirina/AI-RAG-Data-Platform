"""
config/env.py

Utilitaire de chargement des variables d'environnement depuis .env.
Ce module est séparé de settings.py pour permettre un chargement
explicite et précoce avant toute initialisation applicative.
"""

import os
from pathlib import Path
from dotenv import load_dotenv


def load_env(env_file: str = ".env") -> None:
    """
    Charge les variables d'environnement depuis le fichier .env
    situé à la racine du projet.

    Args:
        env_file: Nom du fichier d'environnement (défaut: .env)
    """
    root_dir = Path(__file__).resolve().parent.parent
    env_path = root_dir / env_file

    if not env_path.exists():
        raise FileNotFoundError(
            f"Fichier d'environnement introuvable : {env_path}\n"
            "Assurez-vous que le fichier .env existe à la racine du projet."
        )

    load_dotenv(dotenv_path=env_path, override=False)


def get_env(key: str, default: str = None, required: bool = False) -> str:
    """
    Récupère une variable d'environnement.

    Args:
        key: Nom de la variable
        default: Valeur par défaut si non définie
        required: Si True, lève une exception si absente

    Returns:
        Valeur de la variable d'environnement
    """
    value = os.getenv(key, default)

    if required and value is None:
        raise EnvironmentError(
            f"Variable d'environnement requise non définie : {key}"
        )

    return value