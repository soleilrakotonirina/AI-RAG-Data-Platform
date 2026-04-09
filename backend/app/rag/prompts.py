"""
backend/app/rag/prompts.py

Templates de prompts RAG — bilingues EN→FR.
Les documents sont en anglais, les réponses doivent être en français.

Règle critique :
- Contexte : anglais (documents source)
- Question : français (utilisateur)
- Réponse  : français (obligatoire)
- Reformulation intelligente — jamais de traduction mot-à-mot
"""

from dataclasses import dataclass
from backend.app.core.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# System prompts — bilingues
# ---------------------------------------------------------------------------

RAG_SYSTEM_PROMPT = """Tu es un assistant expert en analyse de documents \
économiques et de développement.

Les documents de contexte sont en anglais. Tu dois :
- Comprendre et analyser les documents en anglais
- Répondre TOUJOURS en français
- Reformuler intelligemment (jamais de traduction mot-à-mot)
- Te baser UNIQUEMENT sur le contexte fourni
- Citer les sources avec [Document N] quand tu utilises une information
- Si l'information est absente du contexte, le dire clairement en français"""


RAG_SYSTEM_HIGH_CONFIDENCE = """Tu es un assistant expert en analyse \
documentaire économique et de développement.

Le contexte fourni est pertinent et fiable.

Règles absolues :
- Réponds TOUJOURS en français
- Synthétise les informations des documents anglais
- Reformule intelligemment — jamais de traduction brute
- Cite les documents sources avec [Document N]
- Sois précis, structuré et complet"""


RAG_SYSTEM_LOW_CONFIDENCE = """Tu es un assistant expert honnête et rigoureux.

Le contexte disponible est de pertinence limitée pour cette question.

Règles absolues :
- Réponds TOUJOURS en français
- Signale clairement que le contexte est partiellement pertinent
- Utilise uniquement les informations directement applicables
- Ne fais aucune extrapolation
- Indique explicitement les lacunes d'information"""


NO_CONTEXT_SYSTEM_PROMPT = """Tu es un assistant honnête et transparent.

Aucun document pertinent n'a été trouvé dans la base de connaissances.

Règles :
- Réponds TOUJOURS en français
- Informe clairement l'utilisateur de l'absence d'information
- Ne fabrique jamais de réponse
- Suggère comment trouver l'information"""


COMPARISON_SYSTEM_PROMPT = """Tu es un assistant général.
Réponds en français avec tes connaissances générales, de façon concise."""


# ---------------------------------------------------------------------------
# User prompt templates — bilingues
# ---------------------------------------------------------------------------

RAG_USER_TEMPLATE_HIGH = """{context_block}

Question : {question}

Instructions :
- Réponds uniquement en français
- Analyse les documents ci-dessus (rédigés en anglais)
- Reformule les informations intelligemment en français
- Ne traduis pas mot-à-mot
- Cite les sources avec [Document N] pour chaque affirmation clé
- Sois complet et structuré

Réponse en français :"""


RAG_USER_TEMPLATE_MEDIUM = """{context_block}

Question : {question}

Instructions :
- Réponds uniquement en français
- Utilise uniquement les informations présentes dans le contexte ci-dessus
- Les documents sont en anglais — reformule en français (pas de traduction brute)
- Indique si certains aspects ne sont pas couverts par le contexte
- Cite les sources avec [Document N]

Réponse en français :"""


RAG_USER_TEMPLATE_LOW = """{context_block}

Question : {question}

Instructions :
- Réponds uniquement en français
- Le contexte ci-dessus est de pertinence limitée pour cette question
- Utilise uniquement les informations directement applicables
- Signale clairement les informations manquantes
- Ne fais aucune extrapolation non fondée
- Cite les sources avec [Document N] pour tout ce que tu utilises

Réponse en français :"""


NO_CONTEXT_USER_TEMPLATE = """Question : {question}

Aucun document pertinent n'a été trouvé dans la base de connaissances \
pour répondre à cette question.

Informe l'utilisateur en français :
1. Que la base ne contient pas cette information
2. Suggère des pistes pour trouver cette information"""


COMPARISON_USER_TEMPLATE = """Question : {question}

Réponds en français avec tes connaissances générales."""


# ---------------------------------------------------------------------------
# Prompt Builder
# ---------------------------------------------------------------------------

@dataclass
class BuiltPrompt:
    """Prompt construit prêt pour le LLM."""
    system: str
    user: str
    prompt_type: str = "rag"
    confidence_level: str = "medium"

    def to_messages(self) -> list[dict]:
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.user},
        ]

    def total_length(self) -> int:
        return len(self.system) + len(self.user)


class PromptBuilder:
    """
    Construit des prompts adaptatifs bilingues EN→FR.

    Sélection automatique selon confiance :
    - high   → template optimiste + synthèse
    - medium → template standard
    - low    → template prudent
    - none   → no-context
    """

    def build_rag_prompt(
        self,
        question: str,
        context_block: str,
        confidence_level: str = "medium",
    ) -> BuiltPrompt:
        """
        Construit un prompt RAG bilingue adaptatif.

        Args:
            question: Question en français
            context_block: Contexte en anglais formaté
            confidence_level: high / medium / low / none

        Returns:
            BuiltPrompt prêt pour OpenRouter
        """
        if confidence_level == "high":
            system = RAG_SYSTEM_HIGH_CONFIDENCE
            template = RAG_USER_TEMPLATE_HIGH
        elif confidence_level == "low":
            system = RAG_SYSTEM_LOW_CONFIDENCE
            template = RAG_USER_TEMPLATE_LOW
        else:
            system = RAG_SYSTEM_PROMPT
            template = RAG_USER_TEMPLATE_MEDIUM

        user_content = template.format(
            context_block=context_block,
            question=question,
        )

        logger.info(
            "RAG prompt built (EN→FR)",
            question_preview=question[:60],
            confidence_level=confidence_level,
            prompt_length=len(system) + len(user_content),
        )

        return BuiltPrompt(
            system=system,
            user=user_content,
            prompt_type="rag",
            confidence_level=confidence_level,
        )

    def build_no_context_prompt(self, question: str) -> BuiltPrompt:
        user_content = NO_CONTEXT_USER_TEMPLATE.format(question=question)
        logger.info("No-context prompt built", question_preview=question[:60])
        return BuiltPrompt(
            system=NO_CONTEXT_SYSTEM_PROMPT,
            user=user_content,
            prompt_type="no_context",
            confidence_level="none",
        )

    def build_comparison_prompt(self, question: str) -> BuiltPrompt:
        user_content = COMPARISON_USER_TEMPLATE.format(question=question)
        logger.info("Comparison prompt built", question_preview=question[:60])
        return BuiltPrompt(
            system=COMPARISON_SYSTEM_PROMPT,
            user=user_content,
            prompt_type="comparison",
            confidence_level="none",
        )