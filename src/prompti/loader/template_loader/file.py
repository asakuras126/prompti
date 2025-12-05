"""Filesystem-based loader for prompt templates."""

from __future__ import annotations

from pathlib import Path

import yaml

from ...template import PromptTemplate, Variant
from ...model_client import ModelConfig
from .base import TemplateLoader, TemplateNotFoundError, VersionEntry


class FileLoader(TemplateLoader):
    """Loader that reads templates from the local filesystem."""

    def __init__(self, base: Path) -> None:
        """Create loader with a base directory."""
        self.base = base

    async def alist_versions(self, name: str) -> list[VersionEntry]:
        """List all available versions for a template from filesystem.

        For filesystem loader, we only have one version per template file.
        """
        path = self.base / f"{name}.yaml"
        if not path.exists():
            return []

        try:
            text = path.read_text()
            data = yaml.safe_load(text)
            version = str(data.get("version", "0"))
            aliases = list(data.get("aliases", []))

            return [VersionEntry(id=version, aliases=aliases)]
        except (FileNotFoundError, yaml.YAMLError, KeyError):
            return []

    async def aget_template(self, name: str, version: str) -> PromptTemplate:
        """Load and return the template identified by name and version."""
        path = self.base / f"{name}.yaml"
        if not path.exists():
            return None

        text = path.read_text()
        data = yaml.safe_load(text)
        template_version = str(data.get("version", "0"))

        # Check if the requested version matches
        if version and version != template_version:
            # raise TemplateNotFoundError(f"Version {version} not found for template {name}")
            return None

        # 处理variants数据
        variants = {}
        for k, v in data.get("variants", {}).items():
            variant_data = v.copy()  # 复制以避免修改原始数据
            
            # 处理model_strategy字段兼容性
            model_cfg_data = variant_data.get("model_cfg") or {}
            model_strategy_data = variant_data.get("model_strategy")
            
            # 兼容性处理：如果model_strategy为空，但model_cfg中包含model_strategy，则从那里获取
            if model_strategy_data is None and isinstance(model_cfg_data, dict):
                model_strategy_data = model_cfg_data.get("model_strategy")
                variant_data["model_strategy"] = model_strategy_data
            
            # 处理model_config字段
            if "model_cfg" in variant_data and variant_data["model_cfg"]:
                clean_model_cfg = {k: v for k, v in model_cfg_data.items() if k != "model_strategy"}
                if clean_model_cfg:
                    variant_data["model_cfg"] = ModelConfig(**clean_model_cfg)
                else:
                    variant_data["model_cfg"] = None
            
            variants[k] = Variant(**variant_data)
        
        tmpl = PromptTemplate(
            id=name,
            name=data.get("name", name),
            description=data.get("description", ""),
            version=template_version,
            aliases=list(data.get("aliases", [])),
            variants=variants,
        )
        return tmpl

    def list_versions_sync(self, name: str) -> list[VersionEntry]:
        """Synchronous version of alist_versions."""
        path = self.base / f"{name}.yaml"
        if not path.exists():
            return []

        try:
            text = path.read_text()
            data = yaml.safe_load(text)
            version = str(data.get("version", "0"))
            aliases = list(data.get("aliases", []))

            return [VersionEntry(id=version, aliases=aliases)]
        except (FileNotFoundError, yaml.YAMLError, KeyError):
            return []

    def get_template_sync(self, name: str, version: str) -> PromptTemplate:
        """Synchronous version of aget_template."""
        path = self.base / f"{name}.yaml"
        if not path.exists():
            return None

        text = path.read_text()
        data = yaml.safe_load(text)
        template_version = str(data.get("version", "0"))

        # Check if the requested version matches
        if version and version != template_version:
            # raise TemplateNotFoundError(f"Version {version} not found for template {name}")
            return None

        # 处理variants数据
        variants = {}
        for k, v in data.get("variants", {}).items():
            variant_data = v.copy()  # 复制以避免修改原始数据
            
            # 处理model_strategy字段兼容性
            model_cfg_data = variant_data.get("model_cfg") or {}
            model_strategy_data = variant_data.get("model_strategy")
            
            # 兼容性处理：如果model_strategy为空，但model_cfg中包含model_strategy，则从那里获取
            if model_strategy_data is None and isinstance(model_cfg_data, dict):
                model_strategy_data = model_cfg_data.get("model_strategy")
                variant_data["model_strategy"] = model_strategy_data
            
            # 处理model_config字段
            if "model_cfg" in variant_data and variant_data["model_cfg"]:
                clean_model_cfg = {k: v for k, v in model_cfg_data.items() if k != "model_strategy"}
                if clean_model_cfg:
                    variant_data["model_cfg"] = ModelConfig(**clean_model_cfg)
                else:
                    variant_data["model_cfg"] = None
            
            variants[k] = Variant(**variant_data)
        
        tmpl = PromptTemplate(
            id=name,
            name=data.get("name", name),
            description=data.get("description", ""),
            version=template_version,
            aliases=list(data.get("aliases", [])),
            variants=variants,
        )
        return tmpl


# Backward compatibility alias
FileSystemLoader = FileLoader