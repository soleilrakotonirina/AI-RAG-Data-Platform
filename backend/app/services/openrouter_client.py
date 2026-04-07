"""
backend/app/services/openrouter_client.py

Client HTTP pour l'API OpenRouter — couche transport pure.
Avec fallback automatique sur plusieurs modèles si rate-limit.
"""

import time
import httpx

from backend.app.core.settings import get_settings
from backend.app.core.logger import get_logger

logger = get_logger(__name__)

LLM_TIMEOUT = 60.0

# Modèle principal
DEFAULT_LLM_MODEL = "mistralai/mistral-small-3.1-24b-instruct"

# Fallback gratuits en cas de 429 ou 404
FALLBACK_FREE_MODELS = [
    "meta-llama/llama-3.3-70b-instruct:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "mistralai/mistral-7b-instruct:free",
    "qwen/qwen3.6-plus:free",
]


class OpenRouterClient:

    def __init__(self, model: str = None):
        settings = get_settings()
        api_key = settings.openrouter_api_key

        logger.info(
            "Checking API key",
            key_length=len(api_key),
            key_preview=api_key[:10] if api_key else "EMPTY",
        )

        if not api_key or not api_key.strip() or \
           api_key.strip() == "your_openrouter_api_key_here":
            raise EnvironmentError(
                "OPENROUTER_API_KEY manquante ou non configurée dans .env\n"
                "Obtenir une clé sur https://openrouter.ai"
            )

        self._api_key = api_key.strip()
        self._base_url = settings.openrouter_base_url.strip()
        self._model = model or DEFAULT_LLM_MODEL

        logger.info(
            "OpenRouterClient initialized",
            model=self._model,
            base_url=self._base_url,
        )

    def _call_api(
        self,
        model: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """
        Appel HTTP brut vers OpenRouter pour un modèle donné.

        Args:
            model: Identifiant du modèle
            messages: Liste de messages
            temperature: Créativité
            max_tokens: Tokens maximum

        Returns:
            Texte de la réponse

        Raises:
            httpx.HTTPStatusError: Erreur HTTP (401, 429, 404, etc.)
            httpx.RequestError: Erreur réseau
            ValueError: Format de réponse inattendu
        """
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        response = httpx.post(
            url=f"{self._base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/rag-agent-system",
                "X-Title": "RAG Agent System",
            },
            json=payload,
            timeout=LLM_TIMEOUT,
        )
        response.raise_for_status()

        data = response.json()

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise ValueError(
                f"Format de réponse inattendu : {e}\n"
                f"Réponse brute : {str(data)[:200]}"
            )

        usage = data.get("usage", {})
        logger.info(
            "OpenRouter response received",
            model=model,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            response_preview=content[:80],
        )

        return content

    def generate_completion(
        self,
        messages: list[dict],
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        """
        Génère une réponse via OpenRouter avec fallback automatique.

        Tente d'abord le modèle principal. Si 429 ou 404, essaie
        les modèles de fallback dans l'ordre.

        Args:
            messages: Liste de messages au format OpenAI
            temperature: Créativité du modèle
            max_tokens: Tokens maximum en sortie

        Returns:
            Texte de la réponse générée

        Raises:
            RuntimeError: Si tous les modèles échouent
        """
        models_to_try = [self._model] + [
            m for m in FALLBACK_FREE_MODELS if m != self._model
        ]

        last_error = None

        for model in models_to_try:
            logger.info(
                "Sending request to OpenRouter",
                model=model,
                message_count=len(messages),
                temperature=temperature,
                max_tokens=max_tokens,
            )

            try:
                return self._call_api(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

            except httpx.HTTPStatusError as e:
                status = e.response.status_code

                if status == 429:
                    logger.warning(
                        "Rate limit hit — trying next model",
                        model=model,
                        next_model=models_to_try[models_to_try.index(model) + 1]
                        if model != models_to_try[-1] else "none",
                    )
                    last_error = e
                    time.sleep(1)
                    continue

                elif status == 404:
                    logger.warning(
                        "Model not found — trying next model",
                        model=model,
                    )
                    last_error = e
                    continue

                else:
                    # 401, 402, 500 — erreur non récupérable
                    logger.error(
                        "OpenRouter HTTP error",
                        status_code=status,
                        response_body=e.response.text[:200],
                        model=model,
                    )
                    raise

            except httpx.TimeoutException:
                logger.error(
                    "OpenRouter request timed out",
                    timeout=LLM_TIMEOUT,
                    model=model,
                )
                last_error = Exception(f"Timeout sur {model}")
                continue

            except httpx.RequestError as e:
                logger.error(
                    "OpenRouter connection error",
                    error=str(e),
                    model=model,
                )
                raise

        raise RuntimeError(
            f"Tous les modèles ont échoué.\n"
            f"Modèles essayés : {models_to_try}\n"
            f"Dernière erreur : {last_error}\n"
            "Solutions :\n"
            "  1. Ajouter du crédit sur https://openrouter.ai/credits\n"
            "  2. Réessayer dans quelques minutes\n"
            "  3. Vérifier https://openrouter.ai/models pour les modèles disponibles"
        )

    def generate_simple(self, prompt: str, **kwargs) -> str:
        """
        Variante simplifiée : envoie un prompt texte unique.

        Args:
            prompt: Texte du prompt
            **kwargs: Paramètres pour generate_completion

        Returns:
            Texte de la réponse
        """
        messages = [{"role": "user", "content": prompt}]
        return self.generate_completion(messages=messages, **kwargs)