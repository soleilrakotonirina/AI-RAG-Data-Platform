# RAG Agent System

Système RAG + Agentic AI — Backend FastAPI + ChromaDB + LangGraph + Dagster

---

## Stack technique

- **API** : FastAPI + Uvicorn
- **Vector DB** : ChromaDB
- **LLM** : OpenRouter
- **Agent** : LangGraph
- **RAG** : LangChain
- **Pipeline** : Dagster
- **UI** : OpenWebUI

---

## Prérequis

- Python 3.11+
- pip

---

## Installation
```bash
# Cloner le projet
git clone <repo>
cd rag-agent-system

# Créer environnement virtuel
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Installer les dépendances
pip install -r requirements.txt

# Configurer les variables d'environnement
cp .env.example .env
# Editer .env avec vos valeurs
```

---

## Lancer l'API
```bash
python backend/run.py
```

L'API est disponible sur : http://localhost:8000

Documentation Swagger : http://localhost:8000/docs

---

## Tester les endpoints

### Health check
```bash
curl http://localhost:8000/api/v1/health
```

Résultat attendu :
```json
{
  "status": "ok",
  "app_name": "rag-agent-system",
  "version": "0.1.0",
  "environment": "development"
}
```

---
## Phase 2 — ChromaDB

### Lancer le test ChromaDB
```bash
python scripts/test_chromadb.py
```

### Données persistées

Les données ChromaDB sont stockées dans :

```data/chromadb/```
Pour réinitialiser complètement :
```bash
rm -rf data/chromadb/
```

### Endpoints disponibles

| Méthode | Endpoint        | Description      |
|---------|----------------|------------------|
| GET     | /api/v1/health | Statut de l'API  |


---

## Structure du projet
rag-agent-system/
├── backend/            # API FastAPI + logique IA
│   ├── app/
│   │   ├── api/        # Routes HTTP (FastAPI)
│   │   ├── core/       # Config, settings, logger
│   │   └── db/         # Intégration ChromaDB
│   └── run.py
├── config/             # Configuration globale
├── scripts/           # Scripts utilitaires (tests, ingestion, etc.)
├── .env                # Variables d'environnement (ne pas commiter)
├── Readme.md
└── requirements.txt

---

## Phases de développement


- [x] Phase 0 — Initialisation projet
- [x] Phase 1 — Backend FastAPI minimal
- [x] Phase 2 — ChromaDB
- [ ] Phase 3 — Embeddings
- [ ] Phase 4 — Retriever
- [ ] Phase 5 — LLM
- [ ] Phase 6 — RAG pipeline
- [ ] Phase 7 — Endpoint /chat
- [ ] Phase 8 — Reranking
- [ ] Phase 9 — Agent LangGraph
- [ ] Phase 10 — Tools
- [ ] Phase 11 — Ingestion pipeline
- [ ] Phase 12 — Dagster
- [ ] Phase 13 — OpenWebUI
- [ ] Phase 14 — Tests
- [ ] Phase 15 — Optimisation
- [ ] Phase 16 — Production

---


## Configuration

### Fichier .env
```env
# Application
APP_NAME=rag-agent-system
APP_VERSION=0.1.0
APP_ENV=development
DEBUG=true

# API Server
API_HOST=0.0.0.0
API_PORT=8000

# OpenRouter
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

# ChromaDB
CHROMA_HOST=localhost
CHROMA_PORT=8001
CHROMA_COLLECTION_NAME=rag_documents

# Logging
LOG_LEVEL=INFO
```

### Obtenir une API Key OpenRouter

1. Aller sur https://openrouter.ai
2. Créer un compte ou se connecter
3. Aller dans la section Keys / API Keys
4. Cliquer sur Create Key
5. Copier la clé et la coller dans `.env`

---

## Instructions d'exécution complètes

Exécuter dans cet ordre depuis la racine `rag-agent-system/` :

### 1. Créer l'environnement virtuel
```bash
python -m venv .venv
source .venv/bin/activate
```

### Séquence complète à exécuter dans l'ordre

* Installer les headers Python 3.13
```bash
sudo apt update
sudo apt install python3.13-dev -y
```

### 2. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 3. Configurer .env

Vérifier que les valeurs `APP_ENV`, `API_HOST`, `API_PORT` sont correctes.
Les clés API (OpenRouter) ne sont pas requises avant la Phase 5.

### 4. Lancer l'API
```bash
python backend/run.py
```

### 5. Tester le endpoint health
```bash
curl http://localhost:8000/api/v1/health
```

### 6. Réinitialiser l'environnement (si besoin)
```bash
deactivate
rm -rf .venv
```

---

## Dépendances principales

| Package           | Version  | Usage                  |
|-------------------|----------|------------------------|
| fastapi           | 0.115.0  | Framework API          |
| uvicorn           | 0.29.0   | Serveur ASGI           |
| pydantic          | 2.9.2    | Validation données     |
| pydantic-settings | 2.5.2    | Gestion configuration  |
| python-dotenv     | 1.0.1    | Chargement .env        |
| structlog         | 24.4.0   | Logging structuré      |
| chromadb          | 0.5.3    | Base vectorielle       |
| httpx             | 0.27.0   | Client HTTP            |
| PyYAML            | 6.0.2    | Lecture settings.yaml  |