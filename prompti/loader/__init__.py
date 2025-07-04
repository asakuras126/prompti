"""Template loaders package."""

from .agenta import AgentaLoader
from .filesystem import FileSystemLoader
from .github_repo import GitHubRepoLoader
from .http import HTTPLoader
from .langfuse import LangfuseLoader
from .local_git_repo import LocalGitRepoLoader
from .memory import MemoryLoader
from .pezzo import PezzoLoader
from .promptlayer import PromptLayerLoader

__all__ = [
    "FileSystemLoader",
    "MemoryLoader",
    "HTTPLoader",
    "PromptLayerLoader",
    "LangfuseLoader",
    "PezzoLoader",
    "AgentaLoader",
    "GitHubRepoLoader",
    "LocalGitRepoLoader",
]
