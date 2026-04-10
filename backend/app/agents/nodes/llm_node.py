"""
backend/app/agents/nodes/llm_node.py

Node LLM — Phase 10.
Trois cas de génération :
1. Avec tool_output (données dynamiques)
2. Avec contexte RAG (documents rerankés)
3. Sans contexte (question générale)
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

# Prompt pour intégration tool_output
TOOL_SYSTEM_PROMPT = """Tu es un assistant expert qui répond en français.

Tu as accès à des données fraîches récupérées dynamiquement.
Utilise ces données comme source principale pour ta réponse.
Reformule intelligemment en français — ne traduis pas mot-à-mot.
Si les données sont insuffisantes, dis-le clairement."""

TOOL_USER_TEMPLATE = """Données récupérées dynamiquement :
{tool_data}

Question : {question}

Instructions :
- Réponds uniquement en français
- Utilise les données ci-dessus comme source principale
- Indique la source des données ([Source: ...])
- Sois précis et structuré

Réponse en français :"""


def llm_node(state: AgentState) -> AgentState:
    """
    Génère la réponse finale.

    Trois cas selon l'état :
    1. tool_output présent et success → réponse basée sur tool
    2. reranked_documents présents → réponse RAG
    3. Ni l'un ni l'autre → réponse LLM directe

    Args:
        state: État courant du graphe

    Returns:
        État mis à jour avec la réponse finale
    """
    question = state["question"]
    reranked_docs = state.get("reranked_documents", [])
    tool_output = state.get("tool_output", {})
    steps = state.get("steps_executed", [])

    logger.info(
        "LLM node started",
        question=question[:80],
        has_tool_output=bool(tool_output and tool_output.get("success")),
        has_documents=len(reranked_docs) > 0,
    )

    try:
        # ------------------------------------------------------------------
        # Cas 1 : Tool output disponible
        # ------------------------------------------------------------------
        if tool_output and tool_output.get("success") and tool_output.get("results"):
            answer = _generate_with_tool(question, tool_output)
            steps.append("llm_node:tool_response")

            logger.info("LLM node completed (tool output)", answer_length=len(answer))

            return {
                **state,
                "answer": answer,
                "context_text": str(tool_output),
                "confidence_level": "medium",
                "quality_score": 0.6,
                "model_used": _llm_client._model,
                "steps_executed": steps,
            }

        # ------------------------------------------------------------------
        # Cas 2 : Contexte RAG disponible
        # ------------------------------------------------------------------
        elif reranked_docs:
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

            steps.append(f"llm_node:rag_response(conf={context.confidence_level})")

            logger.info(
                "LLM node completed (RAG context)",
                answer_length=len(answer),
                confidence_level=context.confidence_level,
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
        # Cas 3 : LLM direct
        # ------------------------------------------------------------------
        else:
            prompt = _prompt_builder.build_comparison_prompt(question=question)

            answer = _llm_client.generate_completion(
                messages=prompt.to_messages(),
                temperature=0.3,
                max_tokens=1024,
            )

            steps.append("llm_node:direct_response")

            logger.info("LLM node completed (direct)", answer_length=len(answer))

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
        steps.append("llm_node:error")
        return {
            **state,
            "answer": "Une erreur est survenue lors de la génération.",
            "error": str(e),
            "steps_executed": steps,
        }


def _generate_with_tool(question: str, tool_output: dict) -> str:
    """
    Génère une réponse en se basant sur le tool_output.

    Args:
        question: Question de l'utilisateur
        tool_output: Résultat du tool

    Returns:
        Réponse générée en français
    """
    # Formatage des résultats du tool
    results = tool_output.get("results", [])
    source = tool_output.get("source", "source externe")

    tool_data_lines = []
    for i, result in enumerate(results, 1):
        title = result.get("title", f"Résultat {i}")
        content = result.get("content", "")
        src = result.get("source", source)
        date = result.get("date", "")
        tool_data_lines.append(
            f"[{i}] {title}\n"
            f"    Données : {content}\n"
            f"    Source : {src} ({date})"
        )

    tool_data = "\n\n".join(tool_data_lines) if tool_data_lines else str(tool_output)

    messages = [
        {"role": "system", "content": TOOL_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": TOOL_USER_TEMPLATE.format(
                tool_data=tool_data,
                question=question,
            ),
        },
    ]

    return _llm_client.generate_completion(
        messages=messages,
        temperature=0.2,
        max_tokens=1024,
    )