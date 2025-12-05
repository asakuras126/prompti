"""万世OpenAI-compatible API client implementation."""
from .litellm import LiteLLMClient, SyncLiteLLMClient


class AWSLiteLLMClient(LiteLLMClient):
    """aws-litellm-compatible API client."""

    provider = "aws-litellm"



class SyncAWSLiteLLMClient(SyncLiteLLMClient):
    """Synchronous aws-litellm-compatible API client."""

    provider = "aws-litellm"