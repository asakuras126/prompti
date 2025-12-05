"""Cloudway OpenAI-compatible API client implementation."""

from .openai_client import OpenAIClient, SyncOpenAIClient


class CloudwayOpenAIClient(OpenAIClient):
    """Cloudway OpenAI-compatible API client."""

    provider = "cloudway-openai"


class SyncCloudwayOpenAIClient(SyncOpenAIClient):
    """Synchronous Cloudway OpenAI-compatible API client."""

    provider = "cloudway-openai"