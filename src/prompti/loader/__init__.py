"""Loaders for serving templates and model configurations from different sources."""

from __future__ import annotations

# Template loaders
from .template_loader import (
    TemplateLoader,
    TemplateNotFoundError, 
    VersionEntry,
    MemoryLoader,
    FileLoader,
    FileSystemLoader,  # Backward compatibility
    HTTPLoader,
    LocalGitRepoLoader,
)

# Model config loaders  
from .model_config_loader import (
    ModelConfigLoader,
    ModelConfigNotFoundError,
    FileModelConfigLoader,
    HTTPModelConfigLoader,
    MemoryModelConfigLoader,
)

__all__ = [
    # Template loaders
    "TemplateLoader",
    "TemplateNotFoundError",
    "VersionEntry", 
    "MemoryLoader",
    "FileLoader",
    "FileSystemLoader",  # Backward compatibility
    "HTTPLoader",
    "LocalGitRepoLoader",
    # Model config loaders
    "ModelConfigLoader",
    "ModelConfigNotFoundError",
    "FileModelConfigLoader",
    "HTTPModelConfigLoader", 
    "MemoryModelConfigLoader",
]
