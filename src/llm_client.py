"""
Unified LLM client supporting multiple backends.
"""

import json
import os
from abc import ABC, abstractmethod
from enum import Enum

from pydantic import BaseModel


def _extract_and_parse_json(text: str, response_model: type[BaseModel]) -> BaseModel:
    """
    Robust JSON extraction for small models like TinyLlama that:
    - Echo the schema before writing the answer
    - Wrap output in markdown fences
    - Sometimes produce truncated JSON

    Strategy: find the last JSON object after any "JSON response:" marker,
    then attempt recovery if the object is truncated.
    """
    # If the model echoed the schema then wrote the real answer,
    # take only the portion after the last "JSON response:" marker.
    if "JSON response:" in text:
        text = text.split("JSON response:")[-1]

    # Strip markdown fences
    if "```json" in text:
        text = text.split("```json")[-1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]

    text = text.strip()

    # First attempt: parse as-is
    try:
        return response_model(**json.loads(text))
    except (json.JSONDecodeError, ValueError):
        pass

    # Second attempt: the JSON is truncated — try to close it so we can at
    # least recover the fields that were written before the cutoff.
    try:
        # Strip trailing incomplete token and close the object
        trimmed = text.rstrip().rstrip(",")
        # Walk back line-by-line until we're at a complete value boundary
        lines = trimmed.splitlines()
        while lines:
            last = lines[-1].rstrip().rstrip(",")
            if last.endswith('"') or last.endswith("}") or (last and last[-1].isdigit()):
                break
            lines.pop()
        recovered = "\n".join(lines).rstrip(",") + "\n}"
        return response_model(**json.loads(recovered))
    except (json.JSONDecodeError, ValueError) as e:
        raise ValueError(
            f"Failed to parse structured output: {e}\nResponse (after extraction):\n{text}"
        )


class LLMBackend(str, Enum):
    """Supported LLM backends."""
    OPENAI = "openai"
    OPENROUTER = "openrouter"
    BEDROCK = "bedrock"


class BaseLLMClient(ABC):
    """Base class for LLM clients."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> str:
        """Generate text completion."""
        pass

    @abstractmethod
    def generate_structured(
        self,
        prompt: str,
        response_model: type[BaseModel],
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> BaseModel:
        """Generate structured output matching Pydantic model."""
        pass

    def generate_batch(
        self,
        prompts: list[str],
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        batch_size: int = 32,
    ) -> list[str]:
        """Generate text completions for several prompts.

        The default implementation is sequential. Backends with a native batch
        API can override it; callers still need to handle individual failures.
        """
        del batch_size  # accepted for API compatibility with native batch clients
        return [
            self.generate(
                prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            for prompt in prompts
        ]


class OpenAICompatibleLLMClient(BaseLLMClient):
    """
    Unified client for OpenAI-compatible APIs.

    Supports:
    - OpenAI (GPT-4o, GPT-4o-mini, GPT-3.5-turbo)
    - OpenRouter (Claude, Gemini, Llama, Mistral, 100+ models)
    - Any other OpenAI-compatible API
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        supports_json_mode: bool = True,
    ):
        """
        Initialize OpenAI-compatible client.

        Args:
            model: Model identifier
            api_key: API key
            base_url: Optional base URL (default: OpenAI)
            supports_json_mode: Whether the API supports response_format JSON mode
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai required. Install: pip install openai")

        self.model = model
        self.supports_json_mode = supports_json_mode

        # Initialize client. The SDK retries transient 429/5xx responses; keep the
        # timeout finite so a stalled request cannot hang a data-generation run.
        client_kwargs = {"api_key": api_key, "timeout": 60.0, "max_retries": 3}
        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = OpenAI(**client_kwargs)

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> str:
        """Generate text completion."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        content = response.choices[0].message.content
        if content is None:
            raise RuntimeError(f"Empty completion from {self.model}")
        return content.strip()

    def generate_structured(
        self,
        prompt: str,
        response_model: type[BaseModel],
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> BaseModel:
        """Generate structured output."""
        schema = response_model.model_json_schema()
        full_prompt = f"{prompt}\n\nRespond with valid JSON matching this schema:\n{json.dumps(schema, indent=2)}"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": full_prompt})

        # Use JSON mode if supported (OpenAI), otherwise parse from text
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if self.supports_json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(**kwargs)
        response_text = response.choices[0].message.content
        if response_text is None:
            raise RuntimeError(f"Empty completion from {self.model}")
        response_text = response_text.strip()

        # Parse JSON
        if self.supports_json_mode:
            return response_model(**json.loads(response_text))
        else:
            return _extract_and_parse_json(response_text, response_model)


class BedrockLLMClient(BaseLLMClient):
    """AWS Bedrock client for Nova, Claude, Llama, Mistral."""

    def __init__(
        self,
        model_id: str = "meta.llama3-70b-instruct-v1:0",
        region: str = "us-east-1",
    ):
        """
        Initialize Bedrock client.

        Args:
            model_id: Bedrock model ID (e.g., meta.llama3-70b-instruct-v1:0)
            region: AWS region
        """
        try:
            import boto3
            from botocore.config import Config as BotocoreConfig
        except ImportError:
            raise ImportError("boto3 required for Bedrock. Install: pip install boto3")

        self.model_id = model_id
        self.client = boto3.client(
            "bedrock-runtime",
            region_name=region,
            config=BotocoreConfig(
                retries={"max_attempts": 5, "mode": "standard"},
                connect_timeout=10,
                read_timeout=120,
            ),
        )

    def _is_mistral(self) -> bool:
        return "mistral" in self.model_id.lower()

    def _is_nova(self) -> bool:
        return "nova" in self.model_id.lower()

    def _is_claude(self) -> bool:
        return "claude" in self.model_id.lower()

    def _format_prompt(self, prompt: str, system_prompt: str | None) -> str:
        """Format prompt for the specific model family."""
        if self._is_mistral():
            # Mistral instruct format
            if system_prompt:
                return f"<s>[INST] {system_prompt}\n{prompt} [/INST]"
            return f"<s>[INST] {prompt} [/INST]"
        else:
            # Llama 3 chat format
            if system_prompt:
                return (
                    f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
                    f"{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
                    f"{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
                )
            return (
                f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
                f"{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
            )

    @staticmethod
    def _nova_request_body(
        prompt: str,
        system_prompt: str | None,
        temperature: float,
        max_tokens: int,
    ) -> dict:
        """Request body for Amazon Nova on Bedrock."""
        body = {
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {
                "temperature": temperature,
                "maxTokens": max_tokens,
            },
        }
        if system_prompt:
            body["system"] = [{"text": system_prompt}]
        return body

    @staticmethod
    def _claude_request_body(
        prompt: str,
        system_prompt: str | None,
        temperature: float,
        max_tokens: int,
    ) -> dict:
        """Request body for Anthropic Claude on Bedrock's InvokeModel API."""
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            body["system"] = system_prompt
        return body

    @staticmethod
    def _nova_response_text(response_body: dict) -> str:
        """Extract text from a Nova InvokeModel response."""
        content = (response_body.get("output") or {}).get("message", {}).get("content") or []
        for block in content:
            if isinstance(block, dict) and block.get("text"):
                return str(block["text"]).strip()
        return ""

    @staticmethod
    def _claude_response_text(response_body: dict) -> str:
        """Extract text from an Anthropic Claude InvokeModel response."""
        content = response_body.get("content") or []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type", "text") == "text" and block.get("text"):
                return str(block["text"]).strip()
        return ""

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> str:
        """Generate with Bedrock."""
        if self._is_claude():
            body = json.dumps(
                self._claude_request_body(
                    prompt, system_prompt, temperature, max_tokens
                )
            )
            response = self.client.invoke_model(modelId=self.model_id, body=body)
            response_body = json.loads(response["body"].read())
            return self._claude_response_text(response_body)

        if self._is_nova():
            body = json.dumps(
                self._nova_request_body(prompt, system_prompt, temperature, max_tokens)
            )
            response = self.client.invoke_model(modelId=self.model_id, body=body)
            response_body = json.loads(response["body"].read())
            return self._nova_response_text(response_body)

        # Llama and Mistral use text completion format
        formatted_prompt = self._format_prompt(prompt, system_prompt)

        body = json.dumps({
            "prompt": formatted_prompt,
            "temperature": temperature,
            "max_gen_len": max_tokens,
            "top_p": 0.9,
        })

        response = self.client.invoke_model(
            modelId=self.model_id,
            body=body,
        )

        response_body = json.loads(response["body"].read())
        # Llama 3 returns "generation", Mistral returns "outputs[0].text"
        if "generation" in response_body:
            return response_body["generation"].strip()
        if "outputs" in response_body:
            return response_body["outputs"][0]["text"].strip()
        return ""

    def generate_structured(
        self,
        prompt: str,
        response_model: type[BaseModel],
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> BaseModel:
        """Generate structured output."""
        # Add JSON schema instruction
        schema = response_model.model_json_schema()
        json_instruction = f"\n\nRespond with valid JSON matching this schema:\n{json.dumps(schema, indent=2)}\n\nJSON response:"

        full_prompt = prompt + json_instruction
        response_text = self.generate(full_prompt, system_prompt, temperature, max_tokens)

        return _extract_and_parse_json(response_text, response_model)


def create_llm_client(
    backend: str | None = None,
    **kwargs
) -> BaseLLMClient:
    """
    Factory function to create LLM client.

    Args:
        backend: Backend type (openai, openrouter, bedrock)
        **kwargs: Backend-specific arguments

    Returns:
        LLM client instance

    Environment variables:
        LLM_BACKEND: Default backend if not specified
        LLM_MODEL: Model name/ID
        OPENAI_API_KEY: OpenAI API key
        OPENROUTER_API_KEY: OpenRouter API key
        AWS_REGION: AWS region for Bedrock (default: us-east-1)
    """
    # Default from environment
    backend = backend or os.getenv("LLM_BACKEND", "openai")

    if backend == LLMBackend.OPENAI:
        model = kwargs.get("model") or os.getenv("LLM_MODEL", "gpt-4o-mini")
        api_key = kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
        return OpenAICompatibleLLMClient(
            model=model,
            api_key=api_key,
            supports_json_mode=True,
        )

    elif backend == LLMBackend.OPENROUTER:
        model = kwargs.get("model") or os.getenv("LLM_MODEL", "anthropic/claude-3.5-sonnet")
        api_key = kwargs.get("api_key") or os.getenv("OPENROUTER_API_KEY")
        return OpenAICompatibleLLMClient(
            model=model,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            supports_json_mode=False,  # OpenRouter doesn't support JSON mode
        )

    elif backend == LLMBackend.BEDROCK:
        model_id = kwargs.get("model_id") or os.getenv("LLM_MODEL", "eu.amazon.nova-pro-v1:0")
        region = kwargs.get("region") or os.getenv("AWS_REGION", "us-east-1")
        return BedrockLLMClient(model_id=model_id, region=region)

    else:
        raise ValueError(
            f"Unknown backend: {backend!r}. "
            f"Choose from: {[b.value for b in LLMBackend]}"
        )
