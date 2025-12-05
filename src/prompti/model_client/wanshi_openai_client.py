"""万世OpenAI-compatible API client implementation."""
from .openai_client import OpenAIClient, SyncOpenAIClient


class WanshiOpenAIClient(OpenAIClient):
    """万世OpenAI-compatible API client."""

    provider = "wanshi-openai"



class SyncWanshiOpenAIClient(SyncOpenAIClient):
    """Synchronous 万世OpenAI-compatible API client."""

    provider = "wanshi-openai"