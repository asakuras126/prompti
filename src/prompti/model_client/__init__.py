"""Model clients for various providers."""

from __future__ import annotations

from ..message import Message
from .base import (
    ModelClient,
    ModelConfig,
    RunParams,
    ToolChoice,
    ToolParams,
    ToolSpec,
)

from .factory import create_client

__all__ = [
    "ModelConfig",
    "ModelClient",
    "RunParams",
    "ToolSpec",
    "ToolParams",
    "ToolChoice",
    "create_client",
    "Message",
    "ModelConfigLoader",
    "FileModelConfigLoader", 
    "HTTPModelConfigLoader",
    "ModelConfigNotFoundError",
]

# Optional import for LiteLLMClient
try:
    from .litellm import LiteLLMClient  # noqa: F401

    __all__.append("LiteLLMClient")
except ImportError:
    pass

# OpenAI clients
try:
    from .openai_client import OpenAIClient  # noqa: F401

    __all__.extend(["OpenAIClient"])
except ImportError:
    pass

# WanshiOpenAI clients
try:
    from .wanshi_openai_client import WanshiOpenAIClient  # noqa: F401

    __all__.extend(["WanshiOpenAIClient"])
except ImportError:
    pass

# CloudwayOpenAI clients  
try:
    from .cloudway_openai_client import CloudwayOpenAIClient  # noqa: F401

    __all__.extend(["CloudwayOpenAIClient"])
except ImportError:
    pass

try:
    from .qianfan_client import QianFanClient  # noqa: F401
    __all__.extend(["QianFanClient"])
except ImportError:
    pass


# Mock provider - standard provider implementation
try:
    from .mock_provider import MockClient, SyncMockClient, load_mock_data, reset_mock_sequence, get_mock_stats  # noqa: F401
    __all__.extend([
        "MockClient", "SyncMockClient", "load_mock_data", "reset_mock_sequence", "get_mock_stats"
    ])
except ImportError:
    pass

# Multi-route mock provider
try:
    from .multi_route_mock_provider import MultiRouteMockClient, SyncMultiRouteMockClient, add_route,\
        reset_multi_route, get_multi_route_stats  # noqa: F401
    __all__.extend([
        "MultiRouteMockClient", "SyncMultiRouteMockClient", "add_route", "reset_multi_route", "get_multi_route_stats"
    ])
except ImportError:
    pass

# AWS Litellm clients  
try:
    from .aws_litellm import AWSLiteLLMClient, SyncAWSLiteLLMClient  # noqa: F401

    __all__.extend(["AWSLiteLLMClient", "SyncAWSLiteLLMClient"])
except ImportError:
    pass

# Azure Litellm clients  
try:
    from .azure_litellm import AzureLiteLLMClient, SyncAzureLiteLLMClient  # noqa: F401

    __all__.extend(["AzureLiteLLMClient", "SyncAzureLiteLLMClient"])
except ImportError:
    pass
