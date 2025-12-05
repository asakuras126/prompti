"""Template loader package for loading prompt templates from various sources."""

from .base import TemplateLoader, TemplateNotFoundError, VersionEntry
from .memory import MemoryLoader
from .file import FileLoader, FileSystemLoader
from .http import HTTPLoader
from .local_git_repo import LocalGitRepoLoader

__all__ = [
    "TemplateLoader",
    "TemplateNotFoundError", 
    "VersionEntry",
    "MemoryLoader",
    "FileLoader",
    "FileSystemLoader",  # Backward compatibility
    "HTTPLoader",
    "LocalGitRepoLoader",
]