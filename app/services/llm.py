import logging

import httpx

from app.config import get_settings


logger = logging.getLogger(__name__)


class InvalidAPIKeyError(Exception):
    pass


class LLMRateLimitError(Exception):
    pass


class LLMTimeoutError(Exception):
    pass


class OpenRouterService:
    def __init__(self) -> None:
        self.settings = get_settings()

    @property
    def headers(self) -> dict[str, str]:
        if (
            not self.settings.openrouter_api_key
            or self.settings.openrouter_api_key == "replace_with_your_openrouter_key"
        ):
            raise InvalidAPIKeyError("OPENROUTER_API_KEY is not configured")
        return {
            "Authorization": f"Bearer {self.settings.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.settings.openrouter_site_url,
            "X-OpenRouter-Title": self.settings.openrouter_app_name,
        }

    async def generate(self, prompt: str) -> str:
        try:
            async with httpx.AsyncClient(timeout=self.settings.request_timeout_seconds) as client:
                response = await client.post(
                    self.settings.openrouter_api_url,
                    headers=self.headers,
                    json={
                        "model": self.settings.openrouter_model,
                        "messages": [
                            {
                                "role": "system",
                                "content": "You answer only from the provided RAG context.",
                            },
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": 0.2,
                        "top_p": 0.9,
                        "max_tokens": 800,
                    },
                )

            self._raise_for_status(response)
            payload = response.json()
            return (
                payload.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError("OpenRouter request timed out") from exc
        except httpx.HTTPError as exc:
            logger.exception("OpenRouter network error")
            raise RuntimeError("OpenRouter request failed") from exc
        except Exception as exc:
            self._raise_typed_error(str(exc))
            raise

    def _raise_for_status(self, response: httpx.Response) -> None:
        if response.status_code < 400:
            return

        message = self._extract_error_message(response)
        logger.error(
            "OpenRouter error status=%s message=%s",
            response.status_code,
            message,
        )

        if response.status_code in {401, 403}:
            raise InvalidAPIKeyError(message)
        if response.status_code == 429:
            raise LLMRateLimitError(message)
        if response.status_code in {408, 504}:
            raise LLMTimeoutError(message)
        raise RuntimeError(f"OpenRouter error {response.status_code}: {message}")

    @staticmethod
    def _extract_error_message(response: httpx.Response) -> str:
        try:
            payload = response.json()
        except ValueError:
            return response.text

        error = payload.get("error", payload)
        if isinstance(error, dict):
            return str(error.get("message") or error.get("code") or error)
        return str(error)

    @staticmethod
    def _raise_typed_error(raw_message: str) -> None:
        message = raw_message.lower()
        if "api key" in message or "permission" in message or "unauthenticated" in message:
            raise InvalidAPIKeyError(raw_message)
        if "quota" in message or "rate" in message or "resource exhausted" in message:
            raise LLMRateLimitError(raw_message)
        if "timeout" in message or "deadline" in message:
            raise LLMTimeoutError(raw_message)


llm_service = OpenRouterService()
