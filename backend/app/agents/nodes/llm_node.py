"""
backend/app/agents/nodes/llm_node.py

Node LLM : génère la réponse finale.

Deux cas :
- Cas 1 : avec contexte (documents rerankés disponibles)
- Cas 2 : sans contexte (question générale, pas de retrieval)

Réutilise ContextBuilder et PromptBuilder de Phase 6.
"""

from backend.app.agents.state import AgentState
from backend.app.rag.context import ContextBuilder
from backend.app.rag.prompts import PromptBuilder
from backend.app.services.openrouter_client import OpenRouterClient
from backend.app.services.retriever_service import RetrievalResult
from backend.app.core.logger import get_logger

logger = get_logger(__name__)

_context_builder = ContextBuilder(max_chars_per_doc=700, max_total_chars=4000)
_prompt_builder = PromptBuilder()
_llm_client = OpenRouterClient()


def llm_node(state: AgentState) -> AgentState:
    """
    Génère la réponse finale à partir du contexte disponible.

    Lit depuis l'état  : question, reranked_documents (si disponibles)
    Écrit dans l'état  : answer, context_text, confidence_level, quality_score

    Cas 1 — avec contexte :
        reranked_documents → ContextBuilder → PromptBuilder (RAG) → LLM
    Cas 2 — sans contexte :
        PromptBuilder (no_context) → LLM

    Args:
        state: État courant du graphe

    Returns:
        État mis à jour avec la réponse finale
    """
    question = state["question"]
    reranked_docs = state.get("reranked_documents", [])
    steps = state.get("steps_executed", [])

    logger.info(
        "LLM node started",
        question=question[:80],
        has_documents=len(reranked_docs) > 0,
        document_count=len(reranked_docs),
    )

    try:
        # ------------------------------------------------------------------
        # Cas 1 : avec contexte documentaire
        # ------------------------------------------------------------------
        if reranked_docs:
            mock_retrieval = RetrievalResult(
                query=question,
                documents=reranked_docs,
                top_k=len(reranked_docs),
                embedding_dim=1536,
                total_in_db=len(reranked_docs),
            )
            context = _context_builder.build(mock_retrieval)

            prompt = _prompt_builder.build_rag_prompt(
                question=question,
                context_block=context.to_prompt_block(),
                confidence_level=context.confidence_level,
            )

            answer = _llm_client.generate_completion(
                messages=prompt.to_messages(),
                temperature=0.2,
                max_tokens=1024,
            )

            steps.append(f"llm_node:rag_response(confidence={context.confidence_level})")

            logger.info(
                "LLM node completed (with context)",
                answer_length=len(answer),
                confidence_level=context.confidence_level,
                quality_score=round(context.quality_score, 3),
                document_count=context.document_count,
            )

            return {
                **state,
                "answer": answer,
                "context_text": context.text,
                "confidence_level": context.confidence_level,
                "quality_score": context.quality_score,
                "model_used": _llm_client._model,
                "steps_executed": steps,
            }

        # ------------------------------------------------------------------
        # Cas 2 : sans contexte (question générale)
        # ------------------------------------------------------------------
        else:
            prompt = _prompt_builder.build_comparison_prompt(question=question)

            answer = _llm_client.generate_completion(
                messages=prompt.to_messages(),
                temperature=0.3,
                max_tokens=1024,
            )

            steps.append("llm_node:direct_response")

            logger.info(
                "LLM node completed (no context)",
                answer_length=len(answer),
            )

            return {
                **state,
                "answer": answer,
                "context_text": "",
                "confidence_level": "none",
                "quality_score": 0.0,
                "model_used": _llm_client._model,
                "steps_executed": steps,
            }

    except Exception as e:
        logger.error("LLM node failed", error=str(e))
        steps.append(f"llm_node:error")
        return {
            **state,
            "answer": "Une erreur est survenue lors de la génération de la réponse.",
            "error": str(e),
            "steps_executed": steps,
        }