# RAG Agent System

Système complet de **Retrieval-Augmented Generation (RAG) + Agentic AI**, conçu avec une architecture modulaire et orientée production.

Ce projet vise à construire un assistant intelligent capable de :

* lire et exploiter des documents (RAG)
* raisonner avec un modèle de langage (LLM)
* agir dynamiquement via des agents IA

---

# Description

RAG Agent System est un projet d’ingénierie IA avancé visant à concevoir un système complet combinant **RAG** et **Agentic AI**.

L’objectif est de dépasser les limites des LLM classiques en intégrant :

* un accès dynamique à des connaissances externes via une base vectorielle (ChromaDB)
* une orchestration intelligente via un agent (LangGraph)
* une génération de réponses contextualisées via des modèles LLM (OpenRouter)

---

# Architecture

Le système repose sur deux pipelines complémentaires :

## Pipeline Offline (Data Processing)

```
Docling → Parsing → Cleaning → Chunking → Enrichment → Embeddings → ChromaDB
```

### Rôle

Transformer des documents bruts en base de connaissances exploitable.

---

## Pipeline Online (Serving)

```
Utilisateur → OpenWebUI
→ Embedding (query)
→ API FastAPI
→ Agent (LangGraph)
→ Retriever (ChromaDB)
→ Reranker
→ RAG (LangChain)
→ LLM (OpenRouter / Ollama)
→ Réponse finale
```

### Rôle

Traiter les requêtes utilisateurs en combinant recherche + raisonnement + décision.

---

# Stack technique

| Composant | Technologie       |
| --------- | ----------------- |
| API       | FastAPI + Uvicorn |
| Vector DB | ChromaDB          |
| LLM       | OpenRouter        |
| Agent     | LangGraph         |
| RAG       | LangChain         |
| Pipeline  | Dagster           |
| UI        | OpenWebUI         |

---

# Prérequis

* Python 3.11+
* pip
* environnement virtuel recommandé

---

# Installation

```bash
git clone <repo-url>
cd rag-agent-system

python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows

pip install -r requirements.txt
```

---

# Configuration

Créer un fichier `.env` :

```env
APP_NAME=rag-agent-system
APP_VERSION=0.1.0
APP_ENV=development
DEBUG=true

API_HOST=0.0.0.0
API_PORT=8000

OPENROUTER_API_KEY=your_api_key
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

CHROMA_HOST=localhost
CHROMA_PORT=8001
CHROMA_COLLECTION_NAME=rag_documents

LOG_LEVEL=INFO
```

---

# Lancer l’API

```bash
python backend/run.py
```

Accès :

* API : http://localhost:8000
* Swagger : http://localhost:8000/docs

---

# Test rapide

```bash
curl http://localhost:8000/api/v1/health
```

Résultat attendu :

```json
{
  "status": "ok"
}
```

# Phase 2 — ChromaDB

## Lancer le test ChromaDB
```bash
python scripts/test_chromadb.py
```

# Phase 3 — Embeddings

## Modèle utilisé

| Paramètre  | Valeur                          |
|------------|---------------------------------|
| Provider   | OpenRouter                      |
| Modèle     | nvidia/llama-nemotron-embed-vl-1b-v2:free   |
| Dimension  | 1536                            |
| Fallback   | Hash déterministe (sans clé)    |

## Lancer le test Embeddings
```bash
python scripts/test_embeddings.py
```

## Résultat attendu (avec clé OpenRouter)
```
[info] Embedding generated via OpenRouter  dim=1536
[info] Indexation OK  document_count=7
[info] Requête : 'base de données vectorielle'
[info]   #1  id=doc_002  score=0.91  topic=chromadb
[info] === VALIDATION PHASE 3 TERMINEE ===
```

## Phase 4 — Retriever

### Composants

| Fichier                              | Rôle                                      |
|--------------------------------------|-------------------------------------------|
| backend/app/services/retriever_service.py | Retrieval sémantique (embedding + search) |
| backend/app/rag/context.py           | Construction du contexte pour le LLM      |

### Lancer le test Retriever
```bash
python scripts/test_retriever.py
```

### Résultat attendu

[info] Retrieval OK  found=3  top_k=3  total_in_db=7
[info] Context OK  document_count=3  context_length=487
[info] === VALIDATION PHASE 4 TERMINEE ===



## Phase 5 — LLM (Génération de réponse)

### Modèle LLM utilisé

| Paramètre   | Valeur                              		   |
|-------------|------------------------------------------------|
| Provider    | OpenRouter                                     |
| Modèle      | mistralai/mistral-small-3.1-24b-instruct       |
| Endpoint    | /chat/completions                   	       |
| Timeout     | 60 secondes                         		   |

### Lancer le test LLM
```bash
python scripts/test_llm.py
```

### Résultat attendu
```
[info     ] Réponse AVEC RAG :
Pour construire un agent IA, le contexte mentionne LangGraph comme un framework spécifique pour construire des agents IA avec gestion d'état et workflows.
[info     ] Comparaison                    avec_rag_length=154 sans_rag_length=117 sources_used=3
[info     ] --- Test 5 : Gestion contexte vide ---
[info     ] Starting LLM generation        context_empty=True document_count=0 model=mistralai/mistral-small-3.1-24b-instruct question=Qu'est-ce que Kubernetes ?
[warning  ] No context available — generating without RAG question=Qu'est-ce que Kubernetes ?
[info     ] Sending request to OpenRouter  max_tokens=1024 message_count=2 model=mistralai/mistral-small-3.1-24b-instruct temperature=0.2
HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 200 OK"
[info     ] OpenRouter response received   completion_tokens=14 model=mistralai/mistral-small-3.1-24b-instruct prompt_tokens=139 response_preview=Je ne peux pas répondre à cette question avec les informations disponibles. total_tokens=153
[info     ] LLM generation completed       answer_length=75 context_used=False document_count=0 question=Qu'est-ce que Kubernetes ?
[info     ] Réponse contexte vide :
Je ne peux pas répondre à cette question avec les informations disponibles.
[info     ] === VALIDATION PHASE 5 TERMINEE ===
[info     ] OpenRouterClient operationnel 
[info     ] LLMService operationnel       
[info     ] Pipeline RAG complet : question → retrieval → contexte → LLM → réponse
```
# Phase 6 — Pipeline RAG complet

## Architecture
```
run_rag_pipeline(question)
        │
        ├── RetrieverService.retrieve() 
        │       └── embed_text()          → OpenRouter embeddings
        │       └── VectorStore.search()  → ChromaDB
        │
        ├── ContextBuilder.build() 		  → formatage
        │       └── format_document_block()
        │
        ├── PromptBuilder.build_rag_prompt() → prompts.py
        │       └── prompts.py templates
        │
        └── LLMService.generate() → OpenRouter
                └── OpenRouterClient.generate_completion()
                        └── OpenRouter /chat/completions
```

## Améliorations implémentées

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

## Lancer le test pipeline

```bash
python scripts/test_rag_pipeline.py
```

## Lancer le test documents réels

```bash
python scripts/test_rag_real.py
```

## Résultats Phase 6 — Documents réels

| Question | Score | Confiance | Sources |
|---|---|---|---|
| Défis économiques Madagascar | 0.770 | HIGH | Rapports WB + Climate |
| Changement climatique | 0.669 | MEDIUM | Climate Report x4 |
| Indicateurs PIB | 0.585 | MEDIUM | GDP ML + Rapports |
| Pauvreté pays développement | 0.678 | MEDIUM | Rapports WB x3 |
| Urbanisation Madagascar | 0.828 | HIGH | URBANIZATION x4 |
| **Moyenne** | **0.706** | | |

## Comportement multilingue

```
Documents  : anglais (PDFs Banque Mondiale)
Questions  : français (utilisateur)
Réponses   : français (reformulation intelligente)
```

## Point d'entrée unique

```python
from backend.app.rag.chain import run_rag_pipeline

result = run_rag_pipeline("Quels sont les défis économiques de Madagascar ?")
print(result.answer)
print(result.confidence_level)
print(result.quality_score)
```

---
# Phase 7 — Endpoint /chat

## Lancer l'API

```bash
python backend/run.py
```

##  Endpoints disponibles

| Méthode | Endpoint              | Description              |
|---------|-----------------------|--------------------------|
| GET     | /api/v1/health        | Statut de l'API          |
| GET     | /api/v1/chat/status   | Statut du pipeline RAG   |
| POST    | /api/v1/chat          | Question → Réponse RAG   |

## Documentation Swagger
Lancer l'API et aller sur : http://localhost:8000/docs

## Format de réponse

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
---

# Phase 8 — Reranking sémantique

## Principe
```
AVANT Phase 8 :

Question → ChromaDB → top-4 documents (cosine)
                              ↓
                    Contexte potentiellement bruité
                              ↓
                           LLM

APRÈS Phase 8 :

Question → ChromaDB → top-12 documents (cosine)
                              ↓
                    Reranker LLM (pertinence fine)
                              ↓
                    top-4 documents rerankés
                              ↓
                    Contexte de haute qualité
                              ↓
                           LLM
```

## Pipeline complet (5 étapes)

| Étape | Composant | Rôle |
|---|---|---|
| 1 | RetrieverService | Embedding + ChromaDB + MMR |
| 2 | RerankerService | Score sémantique LLM (NOUVEAU) |
| 3 | ContextBuilder | Déduplication + qualité |
| 4 | PromptBuilder | Template adaptatif EN→FR |
| 5 | LLMService | Génération réponse |

## Tester le reranking

```bash
python scripts/test_reranking.py
```

## Paramètres

```python
pipeline = RAGPipeline(
    top_k=8,           # Documents récupérés par le retriever
    use_reranking=True, # Activer le reranking
    rerank_top_n=3,    # Documents conservés après reranking
)
```

## Impact

- Moins de bruit dans le contexte
- Documents plus pertinents sémantiquement
- Meilleure qualité de réponse LLM
- Fallback automatique si reranker indisponible


---
# Phase 9 — Agent IA (LangGraph)

```
PIPELINE STATIQUE (Phases 6-8) :
Question → Retrieval → Reranking → Context → LLM → Réponse
           TOUJOURS    TOUJOURS

AGENT IA (Phase 9) :
Question → Decision Node
               │
               ├── besoin documents ?
               │        OUI → Retrieval → Reranking → LLM → Réponse
               │        NON → LLM direct → Réponse
               │
               └── logique extensible (Phase 10 : tools)
```
## Architecture agent

```
START
  │
  ▼
decision_node
  ├── needs_retrieval=True → retriever_node → reranker_node → llm_node → END
  │
  │
  └── needs_retrieval=False → llm_node → END 
```
## Installation LangGraph
Ajouter à requirements.txt :

```
langgraph==0.2.28
``` 
## Tester l'agent

```bash
pip install langgraph==0.2.28
python scripts/test_agent.py
```

## Utilisation directe

```python
from backend.app.agents.agent import run_agent

result = run_agent("Quels sont les défis économiques de Madagascar ?")
print(result.answer)
print(result.needs_retrieval)
print(result.steps_executed)
```

### Comportement

| Question | Décision | Flux |
|---|---|---|
| "Qu'est-ce que FastAPI ?" | NON | decision → llm |
| "Défis économiques Madagascar ?" | OUI | decision → retriever → reranker → llm |

---
# Phase 10 — Tools (Agent capable d'agir)

## Concept Explanation
```
AGENT PHASE 9 (sans tools) :
Question → decision → [RAG] → LLM → Réponse
                       ↑
              Base fixe (1128 docs)

AGENT PHASE 10 (avec tools) :
Question → decision → [RAG]          → LLM → Réponse
                    → [search_tool]  → LLM → Réponse
                    → [api_tool]     → LLM → Réponse
                    → [LLM direct]        → Réponse
```
## Architecture agent complète
```
START
  │
  ▼
decision_node
  │
  ├── needs_tool=True
  │     → tool_node → llm_node → END
  │
  ├── needs_retrieval=True
  │     → retriever_node → reranker_node → llm_node → END
  │
  └──direct
        →llm_node→END                                                             
```
## Tools disponibles

| Tool | Rôle | Déclencheur |
|---|---|---|
| search_tool | Données économiques dynamiques | "actuel", "aujourd'hui", "2025" |
| api_tool | Statut système / ChromaDB | "statut", "pipeline", "collection" |

## Tester les tools

```bash
python scripts/test_tools.py
```

## Utilisation directe

```python
from backend.app.agents.agent import run_agent

# Chemin tool
r = run_agent("Données économiques actuelles de Madagascar")
print(r.needs_tool, r.tool_name)

# Chemin retrieval
r = run_agent("Défis économiques Madagascar selon les rapports ?")
print(r.needs_retrieval, r.document_count)

# Chemin direct
r = run_agent("Qu'est-ce que FastAPI ?")
print(r.needs_retrieval, r.needs_tool)
``` 

---

## Phase 11 — Pipeline Ingestion (Docling)

### Architecture pipeline

```
data/raw/          → Docling (extraction multi-format)
    ↓
data/processed/    → Markdown + frontmatter YAML
    ↓
Chunking (600 chars, overlap 100)
    ↓
Enrichment (langue, domaine, pays détectés automatiquement)
    ↓
data/embeddings/   → cache JSON (évite recalcul)
    ↓
ChromaDB           → index vectoriel final
```


### Formats supportés

| Format | Extracteur | Notes |
|--------|-----------|-------|
| PDF    | Docling   | Sans OCR (rapide) |
| DOCX   | Docling   | Structure préservée |
| HTML   | Docling   | Nettoyage automatique |
| TXT/MD | Lecture directe | Encodage auto |

### Modes d'exécution

| Situation de départ | Commande | Durée |
|---------------------|----------|-------|
| Seulement `data/raw/` | `--reset` | Lente (Docling + API) |
| `data/raw/` + `data/processed/` | `--reset --from-processed` | Moyenne (API embeddings) |
| `data/raw/` + `data/processed/` + `data/embeddings/` | `--reset --from-processed` | Très rapide (tout en cache) |
| Nouveau PDF ajouté | _(aucune option)_ | Rapide (seul le nouveau traité) |
| Bug fix chunking/enrichment | `--reset --from-processed` | Rapide si embeddings cachés |
| Réinstallation complète | `--reset --force` | Lente (tout retraiter) |

### Checkpoints persistants

| Répertoire | Contenu | Créé automatiquement |
|-----------|---------|---------------------|
| `data/raw/` | Documents source originaux | Non (à placer manuellement) |
| `data/processed/` | Markdown par document + fichier .meta (hash) | Oui |
| `data/embeddings/` | Cache JSON des vecteurs | Oui |
| `data/chromadb/` | Index vectoriel ChromaDB | Oui |

### Format data/processed/

Chaque document génère deux fichiers :

```
data/processed/
├── nom_document.md      ← texte en Markdown avec frontmatter YAML
└── nom_document.meta    ← hash SHA256 du fichier source (détection changements)
```

Format `.md` :

```markdown
---
source: document.pdf
file_type: pdf
char_count: 46320
word_count: 7312
processed_at: 2026-04-11T04:50:19
file_hash: a3f2b1c4d5e6f7a8
---

# Contenu du document...
```

### Enrichissement automatique

Le pipeline détecte automatiquement pour chaque chunk :

| Métadonnée | Valeurs possibles | Méthode |
|-----------|-------------------|---------|
| `language` | fr, en, unknown | Mots fréquents |
| `domain` | economics, climate, finance, urbanization, development, general | Mots-clés thématiques |
| `countries` | madagascar, global, west_bank_gaza, unknown | Mots-clés géographiques |
| `indexed_at` | ISO timestamp | Automatique |

### Vérification

```bash
python -c "
import sys; sys.path.insert(0, '.')
from backend.app.db.vector_store import VectorStore
print('Documents ChromaDB:', VectorStore().count())
"
```
---

## Phase 12 — Orchestration Dagster

### Concept

Dagster automatise le pipeline de données complet.
Plus besoin de lancer manuellement `python scripts/ingest_data.py`.

```
AVANT Phase 12 (manuel) :
python scripts/ingest_data.py --reset

APRÈS Phase 12 (automatique) :
Dagster schedule → détecte nouveaux fichiers → traite → ChromaDB mis à jour
```

### Ce que Dagster automatise

```
data/raw/ (nouveau PDF déposé)
        ↓
load_raw_docs_op   → détecte les fichiers nouveaux/modifiés
        ↓
docling_op         → extraction texte (Docling, sans OCR)
        ↓
chunking_op        → découpage (600 chars, overlap 100)
        ↓
enrichment_op      → métadonnées (langue, domaine, pays)
        ↓
embedding_op       → vecteurs OpenRouter (cache disque)
        ↓
chromadb_op        → indexation ChromaDB
```

### Installation

```bash
pip install dagster dagster-webserver
```

### Lancer Dagster

```bash
# Depuis la racine du projet
dagster dev -f pipelines/dagster_project/repository.py

# Avec persistance des données entre sessions
export DAGSTER_HOME=/home/user/dagster_home
dagster dev -f pipelines/dagster_project/repository.py
```

UI accessible sur : `http://localhost:3000`

### Jobs disponibles

| Job | Description | Équivalent script |
|-----|-------------|------------------|
| `ingestion_job` | data/raw/ → data/processed/ | Docling seul |
| `indexing_job` | data/processed/ → ChromaDB | --from-processed |
| `full_pipeline_job` | data/raw/ → ChromaDB (complet) | Pipeline entier |

### Modes de configuration

| Mode Dagster | Équivalent script | Description |
|-------------|-------------------|-------------|
| `mode=full, reset_chromadb=true` | `--reset --force` | Tout retraiter depuis zéro |
| `mode=processed, reset_chromadb=true` | `--reset --from-processed` | Skip Docling, reset ChromaDB |
| `mode=processed, reset_chromadb=false` | `--from-processed` | Skip Docling, ajout incrémental |
| `mode=incremental, reset_chromadb=false` | _(aucune option)_ | Seulement nouveaux fichiers |


### Schedules automatiques

| Schedule | Cron | Job | Mode | Description |
|----------|------|-----|------|-------------|
| `daily_full_pipeline` | `0 0 * * *` | `full_pipeline_job` | full + reset | Reindexation complète chaque nuit |
| `hourly_incremental_pipeline` | `0 * * * *` | `full_pipeline_job` | incremental | Nouveaux fichiers chaque heure |


### Structure ops Dagster

```
Dagster Op            Logique réutilisée
──────────────────────────────────────────────────────
load_raw_docs_op   →  détection fichiers (hash-based)
docling_op         →  ingestion/docling_pipeline.py
chunking_op        →  indexing/chunking.py
enrichment_op      →  indexing/enrichment.py
embedding_op       →  indexing/embeddings.py
chromadb_op        →  backend/app/db/vector_store.py
```

### Workflow quotidien après Phase 12

```bash
# Terminal 1 — Serveur API (une fois au démarrage)
python backend/run.py

# Terminal 2 — Orchestration Dagster (optionnel si schedules activés)
dagster dev -f pipelines/dagster_project/repository.py

# Notre seul travail ensuite :
# → Déposer des PDFs dans data/raw/
# → Dagster les détecte et traite automatiquement
# → ChromaDB se met à jour
# → L'API /chat répond avec les nouveaux documents
```

---

## Sans clé OpenRouter

Le système bascule automatiquement sur un embedding déterministe.
La pipeline fonctionne mais la pertinence sémantique n'est pas garantie.

---

# ARCHITECTURE  (RAG + AGENTS + OpenRouter)
```

                    ┌──────────────────────────────┐
                    │        DAGSTER               │
                    │   Data Pipeline (Offline)    │
                    └─────────────┬────────────────┘
                                  │
     ┌────────────────────────────┼────────────────────────────┐
     │                            │                            │

 INGESTION                  INDEXING                   SERVING (AGENTIC AI)

	Docling                Chunking (smart)         	OpenWebUI (Chat UI)
	   ↓                      	↓                          		↓
	Parsing               Enrichissement          		Choix modèle (LLM)
	   ↓                        ↓                           	↓
	Cleaning              Embeddings (docs)     		Embedding (query)[via OpenRouter]
	   ↓                      	↓               			    ↓
	                     ChromaDB (index)             	   API Backend
																↓
		                                          		  AGENT (LangGraph)
																↓
								                  ┌─────────────┬─────────────┐
								                  │             │             │
								           Retriever        Reranker        Tools/API
								           (ChromaDB)     (OpenRouter)     (external)
								                  │             │             │
								                  └──────┬──────┴──────┬──────┘
																↓
				                                  			LangChain (RAG)
																↓
				                           			   LLM (OpenRouter / Ollama)
																↓
			                                  			   Final Answer
```
---

# Structure du projet

```
rag-agent-system/
│
├── backend/                      #SERVING (toute l’IA runtime)
│   ├── app/
│   │   ├── api/                  # couche HTTP (FastAPI)
│   │   │   ├── main.py           # point d’entrée API (uvicorn)
│   │   │   ├── routes/           # endpoints REST
│   │   │   │   ├── chat.py       # endpoint principal (OpenWebUI)
│   │   │   │   └── health.py     # vérification API
│   │   │   └── deps.py           # injection dépendances (DB, services)
│   │	│
│   │   ├── core/                 # config globale
│   │   │   ├── config.py         # variables globales
│   │   │   ├── settings.py       # lecture .env
│   │   │   └── logger.py         # logs
│   │   │
│   │   ├── services/             # logique métier (connecteurs IA, OpenRouter, etc.)
│   │   │   ├── embedding_service.py   # embedding query (OpenRouter)
│   │   │   ├── retriever_service.py   # recherche ChromaDB
│   │   │   ├── reranker_service.py    # reranking OpenRouter
│   │   │   ├── llm_service.py         # appel LLM
│   │   │   └── openrouter_client.py   # client API OpenRouter
│   │   │
│   │   ├── agents/               # AGENT IA (LangGraph)
│   │   │   ├── agent.py          # logique principale agent
│   │   │   ├── graph.py          # définition workflow (graph LangGraph)
│   │   │   ├── state.py          # état partagé agent
│   │   │   │
│   │   │   ├── nodes/            # étapes du raisonnement
│   │   │   │   ├── retriever_node.py   # recherche docs (appel retrieval)
│   │   │   │   ├── reranker_node.py    # filtrage résultats (appel reranking)
│   │   │   │   ├── llm_node.py         # génération réponse (appel LLM)
│   │   │   │   └── decision_node.py    # logique décision
│   │   │   │
│   │   │   └── tools/            # outils utilisables par agent
│   │   │       ├── search_tool.py
│   │   │       └── api_tool.py
│   │	|
│   │   ├── rag/                  # logique RAG
│   │   │   ├── chain.py          # pipeline LangChain
│   │   │   ├── prompts.py        # templates prompts (prompts LLM)
│   │   │   └── context.py        # construction contexte docs
│   │	|
│   │	|
│   │   └── db/                   # accès base vectorielle
│   │	    ├── chroma_client.py  # connexion ChromaDB
│   │       └── vector_store.py   # wrapper retrieval
│   │
│   └── run.py                    # lancer FastAPI
│
├── pipelines/                    # DAGSTER (offline)
│   ├── dagster_project/
│   │   ├── repository.py        # enregistre jobs
│   │
│   │   ├── jobs/                # pipelines complets
│   │   │   ├── ingestion_job.py # ingestion documents
│   │   │   └── indexing_job.py  # indexation RAG
│   │	|
│   │   ├── ops/                 # étapes pipeline
│   │   │   ├── docling_op.py    # extraction Docling
│   │   │   ├── parsing_op.py    # parsing
│   │   │   ├── cleaning_op.py   # nettoyage
│   │   │   ├── chunking_op.py   # découpage
│   │   │   ├── enrichment_op.py # enrichissement metadata
│   │   │   ├── embedding_op.py  # embeddings docs
│   │   │   └── chromadb_op.py   # stockage DB vectorielle
│   │	|
│   │   ├── resources/           # connexions externes
│   │   │   ├── docling_resource.py
│   │   │   └── chromadb_resource.py
│   │	|
│   │   └── schedules/           # automatisation
│   │       └── daily_job.py     # exécution automatique
│
├── ui/                          # INTERFACE
│   └── openwebui/
│       └── config.yaml          # URL backend + modèles
│
├── data/                        # DONNÉES
│   ├── raw/                     # fichiers bruts
│   ├── processed/               # texte nettoyé
│   ├── embeddings/              # vecteurs
│   └── chromadb/                # index vector DB
│
├── ingestion/                   # traitement fichiers
│   ├── loaders/
│   │   ├── pdf_loader.py        # lire PDF
│   │   ├── excel_loader.py      # lire Excel
│   │   └── web_loader.py        # scraping web
│   │
│   └── docling_pipeline.py      # pipeline Docling
│
├── indexing/                    # logique offline pure
│   ├── chunking.py              # découpage intelligent
│   ├── enrichment.py            # enrichissement données
│   ├── embeddings.py            # génération embeddings
│   └── index_builder.py         # push vers ChromaDB
│
├── scripts/                     # utilitaires
│   ├── run_api.py               # lancer API
│   ├── run_dagster.py           # lancer Dagster
│   └── ingest_data.py           # ingestion manuelle
│
├── config/                      # config globale
│   ├── settings.yaml
│   └── env.py
│
├── tests/                       # tests
│   ├── test_api.py
│   ├── test_rag.py
│   └── test_agent.py
│
├── requirements.txt             # dépendances Python
├── .env                         # clés API (OpenRouter, etc.)
└── README.md                    # documentation projet
```

---

# ROADMAP de construction du système RAG + Agents (validation par étapes)

----

## Phase 0 - Initialisation du projet

* Objectif
Mettre en place un environnement propre et exécutable.

* Actions
- Créer la structure de dossiers complète
- Initialiser environnement Python
- Ajouter `requirements.txt`
- Configurer `.env`
- Créer `README.md` minimal

* Validation
- Le projet s’exécute sans erreur (import modules OK)
- Variables d’environnement accessibles
- Structure claire et cohérente

----

## Phase 1 - Backend minimal (API FastAPI)

* Objectif
Avoir une API fonctionnelle sans logique IA.

* Fichiers concernés
- backend/app/api/main.py
- backend/app/api/routes/health.py
- backend/run.py

* Actions
- Créer serveur FastAPI
- Ajouter endpoint `/health`
- Lancer serveur local

* Validation
- API démarre sans erreur
- Endpoint `/health` retourne OK
- Logs fonctionnels

----

## Phase 2 - Connexion ChromaDB (base vectorielle)

* Objectif
Mettre en place la base de données vectorielle.

* Fichiers concernés
- backend/app/db/chroma_client.py
- backend/app/db/vector_store.py

* Actions
- Initialiser ChromaDB local
- Créer collection
- Ajouter documents manuellement (test)

* Validation
- Connexion DB OK
- Insertion documents OK
- Recherche simple fonctionne

----

## Phase 3 - Embeddings (documents)

* Objectif
Transformer documents en vecteurs.

* Fichiers concernés
- indexing/embeddings.py
- backend/app/services/embedding_service.py

* Actions
- Intégrer modèle embedding (via OpenRouter ou local)
- Générer embeddings docs
- Stocker dans ChromaDB

* Validation
- Chaque document a un embedding
- Recherche vectorielle retourne résultats cohérents

----

## Phase 4 - Retriever (RAG simple)

* Objectif
Construire le premier pipeline RAG minimal.

* Fichiers concernés
- backend/app/services/retriever_service.py
- backend/app/rag/context.py

* Actions
- Implémenter recherche top-k
- Construire contexte à partir documents
- Retourner contexte brut

* Validation
- Une question retourne documents pertinents
- Résultats cohérents avec la requête

----

## Phase 5 - LLM (génération réponse)

* Objectif
Connecter un modèle pour générer des réponses.

* Fichiers concernés
- backend/app/services/llm_service.py
- backend/app/services/openrouter_client.py

* Actions
- Intégrer OpenRouter
- Envoyer prompt + contexte
- Générer réponse

* Validation
- Question → réponse générée
- Réponse utilise le contexte récupéré

----

## Phase 6 - Pipeline RAG complet

* Objectif
Assembler Retriever + LLM

* Fichiers concernés
- backend/app/rag/chain.py
- backend/app/rag/prompts.py

* Actions
- Créer chaîne RAG complète :
  - question → retrieval → contexte → LLM
- Structurer prompt

* Validation
- Réponse correcte avec contexte
- Amélioration visible vs LLM seul

----

## Phase 7 - Endpoint /chat

* Objectif
Exposer le RAG via API

* Fichiers concernés
- backend/app/api/routes/chat.py

* Actions
- Créer endpoint POST `/chat`
- Connecter au pipeline RAG

* Validation
- Appel API retourne réponse RAG
- Compatible avec OpenWebUI

----

## Phase 8 - Reranking

* Objectif
Améliorer la qualité des résultats

* Fichiers concernés
- backend/app/services/reranker_service.py

* Actions
- Ajouter reranking après retrieval
- Filtrer top-k résultats

* Validation
- Résultats plus pertinents
- Réduction bruit documents

----

## Phase 9 - Agent IA (LangGraph)

* Objectif
Passer d’un pipeline fixe à un système intelligent

* Fichiers concernés
- backend/app/agents/graph.py
- backend/app/agents/agent.py
- backend/app/agents/nodes/

* Actions
- Créer agent simple :
  - décision → retrieval → LLM
- Implémenter nodes

* Validation
- L’agent décide quand appeler retrieval
- Pipeline dynamique fonctionnel

----

## Phase 10 - Tools (Agent)

* Objectif
Ajouter capacités d’action

* Fichiers concernés
- backend/app/agents/tools/

* Actions
- Ajouter tools :
  - recherche externe
  - API interne
- Connecter au graph

* Validation
- Agent peut appeler un tool
- Résultat tool utilisé dans réponse

----

## Phase 11 - Pipeline ingestion (offline)

* Objectif
Automatiser traitement des documents

* Fichiers concernés
- ingestion/
- indexing/

* Actions
- Charger documents (PDF, Excel, Web)
- Nettoyer texte
- Chunking intelligent
- Générer embeddings

* Validation
- Documents traités automatiquement
- Données prêtes pour indexation

----

## Phase 12 - Dagster (orchestration)

* Objectif
Automatiser pipeline data

* Fichiers concernés
- pipelines/dagster_project/

* Actions
- Créer jobs ingestion + indexing
- Définir ops
- Ajouter schedule

* Validation
- Pipeline exécutable automatiquement
- Données mises à jour sans intervention

----

## Phase 13 - OpenWebUI

* Objectif
Interface utilisateur

* Fichiers concernés
- ui/openwebui/config.yaml

* Actions
- Connecter API backend
- Configurer modèles

* Validation
- Interface fonctionnelle
- Chat connecté au backend

----

## Phase 14 - Tests

* Objectif
Stabiliser le système

* Fichiers concernés
- tests/

* Actions
- Tester API
- Tester RAG
- Tester agent

* Validation
- Tests passent
- Pas de régression

----

## Phase 15 - Optimisation

* Objectif
Améliorer performance et qualité

* Actions
- Ajuster chunking
- Optimiser prompts
- Ajouter cache
- Ajuster top-k

* Validation
- Réponses plus rapides
- Réponses plus pertinentes

----

## Phase 16 - Production readiness

* Objectif
Système prêt pour usage réel

* Actions
- Logging complet
- Gestion erreurs
- Monitoring
- Sécurité API

* Validation
- Système stable
- Gestion des erreurs robuste

----

## Résumé final
* Phase 0 - Initialisation
* Phase 1 - API FastAPI
* Phase 2 - ChromaDB
* Phase 3 - Embeddings
* Phase 4 - Retriever
* Phase 5 - LLM (OpenRouter)
* Phase 6 - RAG pipeline
* Phase 7 - Endpoint /chat
* Phase 8 - Reranking
* Phase 9 - Agent LangGraph
* Phase 10 - Tools
* Phase 11 - Ingestion
* Phase 12 - Dagster
* Phase 13 - UI
* Phase 14 - Tests
* Phase 15 - Optimisation
* Phase 16 - Production

---

# Concepts clés

## LLM = le cerveau

* raisonnement
* génération de texte

## RAG = cerveau + bibliothèque

* accès à des données externes
* amélioration de la précision

## Agent = cerveau + mains

* prise de décision
* utilisation d’outils
* exécution d’actions

---

# Bonnes pratiques

* Ne jamais exposer les clés API
* Tester chaque phase avant d’avancer
* Séparer ingestion / indexing / serving
* Implémenter un logging structuré

---

# État du projet

Projet en cours de développement, structuré par phases progressives.

---

# Licence

MIT (à confirmer)

---

# Contribution

Les contributions sont bienvenues.

---

# Auteur

**Rakotonirina Soleil**

AI & Data Engineer spécialisé en systèmes intelligents (RAG, LLM, Agentic AI) et en automatisation (n8n Certified).

Je conçois des systèmes IA complets allant du traitement de données (ingestion, indexing) jusqu’au déploiement d’agents intelligents capables de raisonner, rechercher et agir.

* GitHub : https://github.com/soleilrakotonirina
* LinkedIn : https://www.linkedin.com/in/soleil-rakotonirina-351789230/

