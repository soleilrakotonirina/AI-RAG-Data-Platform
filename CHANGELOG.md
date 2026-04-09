## README.md — Mise à jour complète Phase 6

```markdown
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
OPENROUTER_API_KEY=sk-or-v1-your-key-here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

# ChromaDB
CHROMA_HOST=localhost
CHROMA_PORT=8001
CHROMA_COLLECTION_NAME=rag_documents

# Retrieval
RETRIEVAL_TOP_K=5
RETRIEVAL_SCORE_THRESHOLD=0.0

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

## Lancer l'API

```bash
python backend/run.py
```

L'API est disponible sur : http://localhost:8000

Documentation Swagger : http://localhost:8000/docs

---

## Endpoints disponibles

| Méthode | Endpoint        | Description     |
|---------|----------------|-----------------|
| GET     | /api/v1/health | Statut de l'API |

---

## Architecture du système

### Récapitulatif Phase 0 → 6

```
Question utilisateur (français)
        │
        ▼
embed_text()
  ├── Modèle  : openai/text-embedding-3-small
  ├── Dim     : 1536
  ├── Cache   : LRU (évite appels API redondants)
  └── Retry   : automatique (backoff exponentiel)
        │
        ▼
ChromaDB.query()
  ├── Documents : 1128 chunks (PDFs réels)
  ├── Mode      : embedded local persistant
  └── Metric    : similarité cosine
        │
        ▼
Adaptive top-k
  ├── Candidats : top-12 bruts
  ├── Seuil     : best_score × 0.6
  └── Résultat  : 4 documents pertinents
        │
        ▼
MMR (Maximal Marginal Relevance)
  ├── Lambda    : 0.7 (pertinence + diversité)
  ├── Objectif  : éviter redondance des sources
  └── Résultat  : 4 documents diversifiés
        │
        ▼
ContextBuilder
  ├── Déduplication  : similarité Jaccard
  ├── Score qualité  : 0.0 → 1.0
  ├── Confiance      : high / medium / low
  └── Limite         : 5000 chars total
        │
        ▼
PromptBuilder
  ├── Template adaptatif selon confiance
  ├── Instructions bilingues EN → FR
  └── Règle : reformulation intelligente
        │
        ▼
OpenRouter LLM
  ├── Modèle    : mistralai/mistral-small-3.1-24b-instruct
  ├── Fallback  : multi-modèles automatique
  ├── Temp      : 0.2
  └── Max tokens: 1024
        │
        ▼
Réponse finale (français)
  ├── Ancrée dans les documents
  ├── Sources citées [Document N]
  └── Score qualité affiché
```

---

## Structure du projet

```
rag-agent-system/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── main.py           # Point d'entrée FastAPI
│   │   │   ├── deps.py           # Injection dépendances
│   │   │   └── routes/
│   │   │       └── health.py     # GET /api/v1/health
│   │   ├── core/
│   │   │   ├── config.py         # Constantes globales
│   │   │   ├── settings.py       # Lecture .env (pydantic)
│   │   │   └── logger.py         # Logging structuré
│   │   ├── db/
│   │   │   ├── chroma_client.py  # Connexion ChromaDB
│   │   │   └── vector_store.py   # CRUD vectoriel
│   │   ├── services/
│   │   │   ├── embedding_service.py   # Embeddings + cache
│   │   │   ├── retriever_service.py   # Retrieval + MMR
│   │   │   ├── llm_service.py         # Génération LLM
│   │   │   └── openrouter_client.py   # Client HTTP OpenRouter
│   │   └── rag/
│   │       ├── chain.py          # Pipeline RAG orchestré
│   │       ├── prompts.py        # Templates bilingues EN→FR
│   │       └── context.py        # Construction contexte
│   └── run.py                    # Lancement uvicorn
├── indexing/
│   └── embeddings.py             # Chunking + indexation PDF
├── data/
│   ├── raw/                      # PDFs source (anglais)
│   └── chromadb/                 # Index vectoriel persistant
├── scripts/
│   ├── ingest_documents.py       # Ingestion PDF → ChromaDB
│   ├── test_chromadb.py          # Validation Phase 2
│   ├── test_embeddings.py        # Validation Phase 3
│   ├── test_retriever.py         # Validation Phase 4
│   ├── test_llm.py               # Validation Phase 5
│   ├── test_rag_pipeline.py      # Validation Phase 6
│   └── test_rag_real.py          # Test documents réels
├── config/
│   ├── settings.yaml
│   └── env.py
├── .env                          # Variables d'environnement
├── .env.example                  # Template .env
├── requirements.txt
└── README.md
```

---

## Documents supportés

Placer les PDFs dans `data/raw/` :

```
data/raw/
├── Climate-Development-Report-World Bank.pdf
├── GDP-Prediction-using-Machine-Learning.pdf
├── MADAGASCAR-URBANIZATION-REVIEW.pdf
├── Rapports économiques(croissance-exportations-...).pdf
└── Rapports économiques(indicateurs-finance-...).pdf
```

---

## Ingestion des documents

**À faire une seule fois (ou après ajout de nouveaux PDFs) :**

```bash
pip install pypdf
python scripts/ingest_documents.py --reset --chunk-size 500 --domain economics
```

Options disponibles :

| Option | Défaut | Description |
|---|---|---|
| --reset | false | Vider ChromaDB avant ingestion |
| --chunk-size | 500 | Taille des chunks en caractères |
| --overlap | 50 | Chevauchement entre chunks |
| --domain | economics | Domaine des documents |

Résultat attendu :

```
[info] PDF loaded  file=Climate-Development-Report.pdf  pages=45
[info] PDF processed  file=Climate-Development-Report.pdf  chunks=142
[info] === Ingestion terminée ===  documents_indexed=1128
```

---

## Ordre d'exécution des scripts

```
# Une seule fois — ingestion PDFs
python scripts/ingest_documents.py --reset

# Tests à répéter librement
python scripts/test_rag_real.py
python scripts/test_rag_pipeline.py
python scripts/test_retriever.py
python scripts/test_embeddings.py

# ATTENTION — reset la collection ChromaDB
# Ne pas lancer après ingestion des PDFs
# python scripts/test_chromadb.py
```

---

## Phase 2 — ChromaDB

### Lancer le test ChromaDB

```bash
# ATTENTION : ce script resets la collection
# Ne lancer qu'avant l'ingestion des vrais documents
python scripts/test_chromadb.py
```

### Données persistées

```
data/chromadb/
```

Pour réinitialiser :

```bash
rm -rf data/chromadb/
```

---

## Phase 3 — Embeddings

### Modèle utilisé

| Paramètre | Valeur |
|---|---|
| Provider | OpenRouter |
| Modèle | openai/text-embedding-3-small |
| Dimension | 1536 |
| Cache | LRU en mémoire |
| Fallback | Hash déterministe (sans clé) |

### Lancer le test

```bash
python scripts/test_embeddings.py
```

---

## Phase 4 — Retriever

### Composants

| Fichier | Rôle |
|---|---|
| retriever_service.py | Retrieval sémantique + MMR |
| context.py | Construction contexte LLM |

### Paramètres configurables (.env)

```env
RETRIEVAL_TOP_K=5
RETRIEVAL_SCORE_THRESHOLD=0.0
```

### Lancer le test

```bash
python scripts/test_retriever.py
```

---

## Phase 5 — LLM

### Modèle LLM utilisé

| Paramètre | Valeur |
|---|---|
| Provider | OpenRouter |
| Modèle | mistralai/mistral-small-3.1-24b-instruct |
| Endpoint | /chat/completions |
| Timeout | 60 secondes |
| Fallback | Multi-modèles automatique |

### Lancer le test

```bash
python scripts/test_llm.py
```

### Changer de modèle LLM

Modifier dans `backend/app/services/openrouter_client.py` :

```python
DEFAULT_LLM_MODEL = "openai/gpt-4o-mini"
# ou
DEFAULT_LLM_MODEL = "anthropic/claude-3-haiku"
# ou
DEFAULT_LLM_MODEL = "mistralai/mistral-small-3.1-24b-instruct"
```

---

## Phase 6 — Pipeline RAG complet

### Architecture

```
run_rag_pipeline(question)
    ├── RetrieverService    → embed + ChromaDB + MMR
    ├── ContextBuilder      → déduplication + qualité
    ├── PromptBuilder       → template adaptatif EN→FR
    └── LLMService          → OpenRouter
```

### Améliorations implémentées

| Amélioration | Impact |
|---|---|
| Cache embeddings | Évite appels API redondants |
| Cache requêtes | Réponse instantanée si même question |
| MMR | Documents diversifiés |
| Top-k adaptatif | Filtre automatique chunks peu pertinents |
| Déduplication | Supprime fragments quasi-identiques |
| Prompts adaptatifs | Instructions selon confiance (HIGH/MED/LOW) |
| Score qualité | Métrique objective du retrieval |
| Retry embeddings | Résilience aux erreurs réseau |

### Lancer le test pipeline

```bash
python scripts/test_rag_pipeline.py
```

### Lancer le test documents réels

```bash
python scripts/test_rag_real.py
```

### Résultats Phase 6 — Documents réels

```
| Question | Score | Confiance | Sources |
|---|---|---|---|
| Défis économiques Madagascar | 0.770 | HIGH | Rapports WB + Climate |
| Changement climatique | 0.669 | MEDIUM | Climate Report x4 |
| Indicateurs PIB | 0.585 | MEDIUM | GDP ML + Rapports |
| Pauvreté pays développement | 0.678 | MEDIUM | Rapports WB x3 |
| Urbanisation Madagascar | 0.828 | HIGH | URBANIZATION x4 |
| **Moyenne** | **0.706** | | |
```

### Comportement multilingue

```
Documents  : anglais (PDFs Banque Mondiale)
Questions  : français (utilisateur)
Réponses   : français (reformulation intelligente)
```

### Point d'entrée unique

```python
from backend.app.rag.chain import run_rag_pipeline

result = run_rag_pipeline("Quels sont les défis économiques de Madagascar ?")
print(result.answer)
print(result.confidence_level)
print(result.quality_score)
```
---
## Phase 7 — Endpoint /chat

### Lancer l'API

```bash
python backend/run.py
```

### Endpoints disponibles

| Méthode | Endpoint              | Description              |
|---------|-----------------------|--------------------------|
| GET     | /api/v1/health        | Statut de l'API          |
| GET     | /api/v1/chat/status   | Statut du pipeline RAG   |
| POST    | /api/v1/chat          | Question → Réponse RAG   |

### Exemples curl

```bash
# Statut pipeline
curl http://localhost:8000/api/v1/chat/status

# Question simple
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Quels sont les défis économiques de Madagascar ?"}'

# Avec paramètres avancés
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Comment le changement climatique affecte-t-il Madagascar ?",
    "top_k": 5,
    "use_mmr": true
  }'
```

### Documentation Swagger
Lancer l'API et aller sur : http://localhost:8000/docs

### Format de réponse

```json
{
  "answer": "Réponse en français...",
  "sources": [
    {
      "id": "chunk_id",
      "score": 0.6768,
      "confidence": "high",
      "source_file": "rapport.pdf",
      "topic": "economics",
      "chunk_index": 98
    }
  ],
  "metadata": {
    "model": "mistralai/mistral-small-3.1-24b-instruct",
    "language": "fr",
    "document_count": 4,
    "quality_score": 0.77,
    "confidence_level": "high",
    "context_used": true,
    "from_cache": false,
    "duration_ms": 34293
  }
}
```

### Tester l'API

```bash
# API doit être lancée dans un terminal
python backend/run.py

# Dans un second terminal
python scripts/test_api_chat.py
```

---

## Dépendances principales

| Package | Version | Usage |
|---|---|---|
| fastapi | 0.115.0 | Framework API |
| uvicorn | 0.29.0 | Serveur ASGI |
| pydantic | 2.9.2 | Validation données |
| pydantic-settings | 2.5.2 | Gestion configuration |
| python-dotenv | 1.0.1 | Chargement .env |
| structlog | 24.4.0 | Logging structuré |
| chromadb | 0.5.23 | Base vectorielle |
| httpx | 0.27.0 | Client HTTP |
| PyYAML | 6.0.2 | Lecture settings.yaml |
| pypdf | 4.2.0 | Extraction texte PDF |

---

## Phases de développement

- [x] Phase 0  — Initialisation projet
- [x] Phase 1  — Backend FastAPI minimal
- [x] Phase 2  — ChromaDB (base vectorielle)
- [x] Phase 3  — Embeddings (documents)
- [x] Phase 4  — Retriever (RAG simple)
- [x] Phase 5  — LLM (OpenRouter)
- [x] Phase 6  — Pipeline RAG complet (documents réels EN→FR)
- [x] Phase 7  — Endpoint /chat
- [ ] Phase 8  — Reranking
- [ ] Phase 9  — Agent LangGraph
- [ ] Phase 10 — Tools (Agent)
- [ ] Phase 11 — Pipeline ingestion (Docling)
- [ ] Phase 12 — Dagster (orchestration)
- [ ] Phase 13 — OpenWebUI (interface)
- [ ] Phase 14 — Tests automatisés
- [ ] Phase 15 — Optimisation
- [ ] Phase 16 — Production readiness

---

## Instructions d'exécution complètes

```bash
# 1. Créer l'environnement virtuel
python -m venv .venv
source .venv/bin/activate

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Configurer .env
cp .env.example .env
# Editer .env et ajouter OPENROUTER_API_KEY

# 4. Placer les PDFs dans data/raw/

# 5. Ingérer les documents (une seule fois)
python scripts/ingest_documents.py --reset

# 6. Lancer l'API
python backend/run.py

# 7. Tester le pipeline RAG
python scripts/test_rag_real.py

# 8. Réinitialiser l'environnement si besoin
deactivate
rm -rf .venv
```
```