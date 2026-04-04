"""
backend/run.py

Point de lancement du serveur uvicorn.
Ajoute la racine du projet au PYTHONPATH avant tout import.
"""

import sys
from pathlib import Path

# Ajoute la racine du projet (rag-agent-system/) au PYTHONPATH
# Nécessaire pour résoudre les imports du type : from backend.app.xxx import ...
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import uvicorn
from backend.app.core.settings import get_settings


def main() -> None:
    settings = get_settings()

    uvicorn.run(
        "backend.app.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()