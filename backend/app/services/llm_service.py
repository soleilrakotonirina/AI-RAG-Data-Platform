"""
backend/app/services/llm_service.py

Couche d'orchestration LLM pour le pipeline RAG.
Responsabilités :
- Construire le prompt final (système + contexte + question)
- Appeler OpenRouterClient
- Structurer et retourner la réponse

Ce fichier ne fait PAS d'appels HTTP directs (rôle de openrouter_client.py).
Ce fichier ne fait PAS de retrieval (rôle de retriever_service.py).
"""

from dataclasses import dataclass

from backend.app.services.openrouter_client import OpenRouterClient, DEFAULT_LLM_MODEL
from backend.app.rag.context import Context
from backend.app.core.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """Tu es un assistant expert qui répond aux questions
en te basant UNIQUEMENT sur le contexte fourni.

Règles strictes :
- Réponds uniquement à partir des informations présentes dans le contexte
- Si le contexte ne contient pas la réponse, dis-le clairement
- Sois précis, concis et structuré
- Ne fabrique pas d'informations
- Réponds dans la même langue que la question"""

RAG_PROMPT_TEMPLATE = """{context_block}

Question : {question}

Réponds de manière claire et précise en te basant uniquement sur
le contexte ci-dessus."""

NO_CONTEXT_PROMPT_TEMPLATE = """Question : {question}

Aucun document pertinent n'a été trouvé dans la base de connaissances
pour répondre à cette question. Indique clairement que tu ne peux pas
répondre avec les informations disponibles."""


# ---------------------------------------------------------------------------
# Types de données
# ---------------------------------------------------------------------------

@dataclass
class LLMResponse:
    """
    Réponse structurée du LLM incluant les métadonnées de génération.
    """
    question: str
    answer: str
    model: str
    context_used: bool
    document_count: int
    sources: list[dict]

    def format_full(self) -> str:
        """
        Formate la réponse complète avec sources pour affichage.

        Returns:
            Chaîne formatée avec réponse et sources
        """
        lines = [
            f"Question : {self.question}",
            "",
            f"Réponse :",
            self.answer,
        ]

        if self.sources:
            lines.append("")
            lines.append("Sources utilisées :")
            for i, src in enumerate(self.sources, 1):
                topic = src.get("metadata", {}).get("topic", "?")
                score = src.get("score", 0)
                lines.append(f"  [{i}] {src['id']} | topic={topic} | score={score:.4f}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM Service
# ---------------------------------------------------------------------------

class LLMService:
    """
    Service de génération de réponse RAG.

    Orchestre :
    1. Construction du prompt (contexte + question)
    2. Appel au LLM via OpenRouterClient
    3. Retour de la réponse structurée

    Usage :
        service = LLMService()
        response = service.generate(
            question="Qu'est-ce que ChromaDB ?",
            context=context_object,
        )
        print(response.answer)
    """

    def __init__(self, model: str = None):
        """
        Args:
            model: Modèle LLM à utiliser via OpenRouter.
                   Défaut : mistralai/mistral-7b-instruct
        """
        self._model = model or DEFAULT_LLM_MODEL
        self._client = OpenRouterClient(model=self._model)

        logger.info(
            "LLMService initialized",
            model=self._model,
        )

    def generate(
        self,
        question: str,
        context: Context,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """
        Génère une réponse RAG complète.

        Pipeline :
        1. Vérifie si le contexte est disponible
        2. Construit le prompt approprié
        3. Appelle le LLM
        4. Retourne LLMResponse structuré

        Args:
            question: Question de l'utilisateur
            context: Objet Context construit par ContextBuilder (Phase 4)
            temperature: Créativité du modèle
            max_tokens: Tokens maximum en sortie

        Returns:
            LLMResponse avec réponse et métadonnées

        Raises:
            ValueError: Si la question est vide
        """
        if not question or not question.strip():
            raise ValueError("La question ne peut pas être vide.")

        logger.info(
            "Starting LLM generation",
            question=question,
            context_empty=context.is_empty,
            document_count=context.document_count,
            model=self._model,
        )

        # Construction du prompt selon disponibilité du contexte
        if context.is_empty:
            user_prompt = NO_CONTEXT_PROMPT_TEMPLATE.format(
                question=question,
            )
            logger.warning(
                "No context available — generating without RAG",
                question=question,
            )
        else:
            user_prompt = RAG_PROMPT_TEMPLATE.format(
                context_block=context.to_prompt_block(),
                question=question,
            )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        # Appel LLM
        answer = self._client.generate_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Construction sources pour traçabilité
        sources = [
            {
                "id": src.id,
                "score": src.score,
                "metadata": src.metadata,
            }
            for src in context.sources
        ]

        response = LLMResponse(
            question=question,
            answer=answer,
            model=self._model,
            context_used=not context.is_empty,
            document_count=context.document_count,
            sources=sources,
        )

        logger.info(
            "LLM generation completed",
            question=question,
            answer_length=len(answer),
            context_used=response.context_used,
            document_count=response.document_count,
        )

        return response

    def generate_without_context(
        self,
        question: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        """
        Génère une réponse LLM pure sans contexte RAG.
        Utile pour comparer la qualité avec/sans retrieval.

        Args:
            question: Question de l'utilisateur
            temperature: Créativité
            max_tokens: Tokens maximum

        Returns:
            Réponse texte brute
        """
        logger.info(
            "LLM generation without RAG context",
            question=question,
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question : {question}"},
        ]

        return self._client.generate_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )