"""PromptI: provider-agnostic prompt engine."""

from __future__ import annotations

from .engine import PromptEngine, BeforeRunHook, AfterRunHook, HookResult
from .experiment import (
    ExperimentRegistry,
    ExperimentSplit,
    GrowthBookRegistry,
    UnleashRegistry,
    bucket,
)
from .loader import (
    FileSystemLoader,
    HTTPLoader,
    LocalGitRepoLoader,
    MemoryLoader,
    TemplateLoader,
    TemplateNotFoundError,
)
from .message import (
    Message,
    Usage,
    Choice,
    ModelResponse,
    StreamingChoice,
    StreamingModelResponse,
)
from .model_client import (
    ModelClient,
    ModelConfig,
    RunParams,
    ToolChoice,
    ToolParams,
    ToolSpec,
    create_client,
)
from .replay import ModelClientRecorder, ReplayEngine
from .template import PromptTemplate

__all__ = [
    "Message",
    "Usage",
    "Choice", 
    "ModelResponse",
    "StreamingChoice",
    "StreamingModelResponse",
    "PromptTemplate",
    "PromptEngine",
    "BeforeRunHook",
    "AfterRunHook", 
    "HookResult",
    "ModelClient",
    "ModelConfig",
    "RunParams",
    "ToolSpec",
    "ToolParams",
    "ToolChoice",
    "create_client",
    "ReplayEngine",
    "ModelClientRecorder",
    "ExperimentRegistry",
    "ExperimentSplit",
    "UnleashRegistry",
    "GrowthBookRegistry",
    "bucket",
    "TemplateLoader",
    "TemplateNotFoundError",
    "HTTPLoader",
    "FileSystemLoader",
    "MemoryLoader",
    "LocalGitRepoLoader",
]

# Optional imports - only available when litellm is installed
try:
    from .model_client import LiteLLMClient  # noqa: F401

    __all__.append("LiteLLMClient")
except ImportError:
    # LiteLLMClient not available - litellm dependency not installed
    pass
