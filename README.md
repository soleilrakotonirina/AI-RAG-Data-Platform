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
| Modèle     | openai/text-embedding-3-small   |
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

