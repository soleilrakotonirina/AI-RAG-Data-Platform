"""
backend/app/api/routes/chat.py

Endpoint REST du pipeline RAG.
Responsabilités :
- Exposer POST /api/v1/chat
- Valider les inputs avec Pydantic
- Déléguer au pipeline RAG (chain.py Phase 6)
- Retourner une réponse JSON structurée
- Gérer toutes les erreurs proprement
- Logger chaque interaction

Ce fichier ne contient AUCUNE logique RAG.
Toute la logique est dans backend/app/rag/chain.py.
"""

import time
from typing import Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from backend.app.rag.chain import run_rag_pipeline, RAGPipelineResult
from backend.app.core.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Schémas de données
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    """
    Schéma de requête du endpoint /chat.

    Champs :
        question        : Question de l'utilisateur (obligatoire)
        top_k           : Nombre de documents à récupérer (optionnel)
        score_threshold : Score minimal de similarité (optionnel)
        use_mmr         : Activer MMR pour diversité (optionnel)
    """
    question: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="Question de l'utilisateur en langage naturel",
        examples=["Quels sont les défis économiques de Madagascar ?"],
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=20,
        description="Nombre de documents à récupérer (défaut : settings)",
    )
    score_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Score minimal de similarité (défaut : 0.0)",
    )
    use_mmr: Optional[bool] = Field(
        default=True,
        description="Activer MMR pour diversifier les résultats",
    )

    @field_validator("question")
    @classmethod
    def validate_question(cls, v: str) -> str:
        """Nettoie et valide la question."""
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("La question ne peut pas être vide ou composée uniquement d'espaces.")
        return cleaned

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "question": "Quels sont les principaux défis économiques de Madagascar ?",
                    "top_k": 4,
                    "score_threshold": 0.0,
                    "use_mmr": True,
                }
            ]
        }
    }


class SourceInfo(BaseModel):
    """Informations sur une source documentaire utilisée."""
    id: str = Field(description="Identifiant du chunk source")
    score: float = Field(description="Score de similarité cosine")
    confidence: str = Field(description="Niveau de confiance : high / medium / low")
    source_file: str = Field(description="Nom du fichier PDF source")
    topic: Optional[str] = Field(default=None, description="Sujet du document")
    chunk_index: Optional[int] = Field(default=None, description="Index du chunk dans le PDF")


class ChatMetadata(BaseModel):
    """Métadonnées de la réponse générée."""
    model: str = Field(description="Modèle LLM utilisé")
    language: str = Field(default="fr", description="Langue de la réponse")
    document_count: int = Field(description="Nombre de documents utilisés")
    quality_score: float = Field(description="Score qualité du contexte (0.0 → 1.0)")
    confidence_level: str = Field(description="Niveau de confiance global")
    context_used: bool = Field(description="Si un contexte documentaire a été utilisé")
    from_cache: bool = Field(description="Si la réponse vient du cache")
    duration_ms: float = Field(description="Durée totale de traitement en ms")
    steps: list[dict] = Field(
        default_factory=list,
        description="Détail des étapes du pipeline",
    )


class ChatResponse(BaseModel):
    """
    Schéma de réponse du endpoint /chat.

    Champs :
        answer          : Réponse générée par le LLM (en français)
        sources         : Documents utilisés pour générer la réponse
        metadata        : Informations de traitement
    """
    answer: str = Field(description="Réponse générée en français")
    sources: list[SourceInfo] = Field(
        default_factory=list,
        description="Sources documentaires utilisées",
    )
    metadata: ChatMetadata = Field(description="Métadonnées de traitement")


class ErrorResponse(BaseModel):
    """Schéma d'erreur standardisé."""
    error: str = Field(description="Type d'erreur")
    message: str = Field(description="Message d'erreur détaillé")
    question: Optional[str] = Field(default=None, description="Question originale")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_sources(result: RAGPipelineResult) -> list[SourceInfo]:
    """
    Construit la liste des sources depuis le résultat du pipeline.

    Args:
        result: Résultat complet du pipeline RAG

    Returns:
        Liste de SourceInfo structurés
    """
    sources = []
    for src in result.sources:
        metadata = src.get("metadata", {})
        sources.append(SourceInfo(
            id=src.get("id", "unknown"),
            score=round(src.get("score", 0.0), 4),
            confidence=src.get("confidence", "unknown"),
            source_file=metadata.get("source", "unknown"),
            topic=metadata.get("topic"),
            chunk_index=metadata.get("chunk_index"),
        ))
    return sources


def _build_metadata(result: RAGPipelineResult) -> ChatMetadata:
    """
    Construit les métadonnées depuis le résultat du pipeline.

    Args:
        result: Résultat complet du pipeline RAG

    Returns:
        ChatMetadata structuré
    """
    steps = [
        {
            "name": step.name,
            "duration_ms": round(step.duration_ms),
            "success": step.success,
        }
        for step in result.steps
    ]

    return ChatMetadata(
        model=result.model,
        language="fr",
        document_count=result.document_count,
        quality_score=round(result.quality_score, 3),
        confidence_level=result.confidence_level,
        context_used=result.context_used,
        from_cache=result.from_cache,
        duration_ms=round(result.total_duration_ms),
        steps=steps,
    )


# ---------------------------------------------------------------------------
# Endpoint principal
# ---------------------------------------------------------------------------

@router.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Pipeline RAG — Question / Réponse",
    description=(
        "Reçoit une question en français, interroge la base documentaire "
        "(PDFs en anglais via ChromaDB), et retourne une réponse en français "
        "générée par le LLM en se basant uniquement sur les documents pertinents."
    ),
    tags=["RAG"],
    responses={
        200: {"description": "Réponse générée avec succès"},
        400: {"description": "Question invalide", "model": ErrorResponse},
        422: {"description": "Erreur de validation Pydantic"},
        500: {"description": "Erreur interne du pipeline RAG", "model": ErrorResponse},
        503: {"description": "Service LLM indisponible", "model": ErrorResponse},
    },
)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Endpoint principal du système RAG.

    Pipeline exécuté :
    1. Validation de la question
    2. Embedding de la requête (OpenRouter)
    3. Retrieval sémantique (ChromaDB + MMR)
    4. Construction du contexte (déduplication + qualité)
    5. Construction du prompt adaptatif (EN→FR)
    6. Génération LLM (OpenRouter)
    7. Retour de la réponse structurée

    Args:
        request: ChatRequest avec question et paramètres optionnels

    Returns:
        ChatResponse avec réponse, sources et métadonnées
    """
    request_start = time.time()

    logger.info(
        "Chat request received",
        question=request.question,
        question_length=len(request.question),
        top_k=request.top_k,
        use_mmr=request.use_mmr,
    )

    try:
        # Appel au pipeline RAG Phase 6
        result = run_rag_pipeline(
            question=request.question,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
            use_mmr=request.use_mmr if request.use_mmr is not None else True,
            adaptive_k=True,
            use_cache=True,
        )

        # Construction de la réponse
        sources = _build_sources(result)
        metadata = _build_metadata(result)

        total_ms = (time.time() - request_start) * 1000

        logger.info(
            "Chat request completed",
            question=request.question[:80],
            answer_length=len(result.answer),
            document_count=result.document_count,
            quality_score=round(result.quality_score, 3),
            confidence_level=result.confidence_level,
            from_cache=result.from_cache,
            total_ms=round(total_ms),
        )

        return ChatResponse(
            answer=result.answer,
            sources=sources,
            metadata=metadata,
        )

    except ValueError as e:
        logger.warning(
            "Chat request validation error",
            question=request.question[:80],
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "validation_error",
                "message": str(e),
                "question": request.question,
            },
        )

    except RuntimeError as e:
        # Tous les modèles LLM ont échoué
        error_msg = str(e)
        logger.error(
            "LLM service unavailable",
            question=request.question[:80],
            error=error_msg[:200],
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "llm_unavailable",
                "message": "Le service LLM est temporairement indisponible. Réessayez dans quelques instants.",
                "question": request.question,
            },
        )

    except Exception as e:
        logger.error(
            "Chat request failed",
            question=request.question[:80],
            error=str(e),
            error_type=type(e).__name__,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "pipeline_error",
                "message": f"Erreur interne du pipeline RAG : {type(e).__name__}",
                "question": request.question,
            },
        )


# ---------------------------------------------------------------------------
# Endpoint de statut du pipeline
# ---------------------------------------------------------------------------

@router.get(
    "/chat/status",
    summary="Statut du pipeline RAG",
    description="Vérifie que ChromaDB est accessible et contient des documents.",
    tags=["RAG"],
)
async def chat_status() -> dict:
    """
    Vérifie l'état du pipeline RAG.

    Contrôles effectués :
    - Connexion ChromaDB
    - Nombre de documents indexés
    - Disponibilité du pipeline

    Returns:
        Dict avec statut et métriques
    """
    try:
        from backend.app.db.vector_store import VectorStore
        store = VectorStore()
        doc_count = store.count()

        status_info = {
            "status": "ok",
            "pipeline": "ready",
            "chromadb": {
                "connected": True,
                "document_count": doc_count,
                "collection": "rag_documents",
            },
            "capabilities": {
                "language_input": "fr",
                "language_documents": "en",
                "language_output": "fr",
                "mmr": True,
                "cache": True,
                "adaptive_k": True,
            },
        }

        if doc_count == 0:
            status_info["status"] = "warning"
            status_info["pipeline"] = "empty"
            status_info["message"] = (
                "ChromaDB est vide. "
                "Lancer : python scripts/ingest_documents.py --reset"
            )

        logger.info(
            "Pipeline status checked",
            document_count=doc_count,
            status=status_info["status"],
        )

        return status_info

    except Exception as e:
        logger.error("Pipeline status check failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "chromadb_unavailable",
                "message": f"ChromaDB inaccessible : {str(e)}",
            },
        )