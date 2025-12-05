"""Enhanced memory loader that accepts dict or PromptTemplate directly."""

from __future__ import annotations

import yaml
from typing import Union, Dict, List

from ...template import PromptTemplate, Variant
from .base import TemplateLoader, TemplateNotFoundError, VersionEntry


class MemoryLoader(TemplateLoader):
    """Load templates from in-memory dict data or PromptTemplate objects directly.
    
    Supports both:
    - New format: Dict[str, Union[Dict, PromptTemplate]]
    - Legacy format: Dict[str, Dict[str, str]] with YAML content
    """

    def __init__(self, templates: Dict[str, Union[Dict, PromptTemplate]]):
        """Initialize with template mapping.
        
        Args:
            templates: Dictionary mapping template names to either:
                - Dict containing template data (compatible with PromptTemplate.from_dict)
                - PromptTemplate instances directly
                - Legacy format: Dict with 'yaml' key containing YAML string
        """
        self.templates: Dict[str, PromptTemplate] = {}
        
        for name, template_data in templates.items():
            if isinstance(template_data, PromptTemplate):
                self.templates[name] = template_data
            elif isinstance(template_data, dict):
                # Check if this is legacy MemoryLoader format with YAML
                if 'yaml' in template_data and isinstance(template_data['yaml'], str):
                    self.templates[name] = self._parse_yaml_template(name, template_data)
                else:
                    # Use from_dict to create PromptTemplate from dict
                    self.templates[name] = PromptTemplate.from_dict(template_data)
            else:
                raise ValueError(f"Invalid template data type for '{name}': {type(template_data)}")

    def _parse_yaml_template(self, name: str, data: Dict[str, str]) -> PromptTemplate:
        """Parse legacy YAML format template data."""
        text = data.get("yaml", "")
        ydata = yaml.safe_load(text) if text else {}
        version = str(ydata.get("version", data.get("version", "0")))
        
        # Convert variants to the expected format
        variants = {}
        for k, v in ydata.get("variants", {}).items():
            variant_data = v.copy()
            # Handle messages_template -> messages conversion
            if "messages_template" in variant_data:
                variant_data["messages"] = variant_data.pop("messages_template")
            variants[k] = Variant(**variant_data)
        
        return PromptTemplate(
            id=name,
            name=ydata.get("name", name),
            description=ydata.get("description", ""),
            version=version,
            aliases=list(ydata.get("aliases", [])),
            variants=variants,
        )

    async def alist_versions(self, name: str) -> List[VersionEntry]:
        """Return available versions for the template name."""
        template = self.templates.get(name)
        if not template:
            return []

        version = template.version or "0"
        aliases = list(template.aliases)

        return [VersionEntry(id=version, aliases=aliases)]

    async def aget_template(self, name: str, version: str) -> PromptTemplate:
        """Return the template for the specific version."""
        template = self.templates.get(name)
        if not template:
            raise TemplateNotFoundError(name)

        template_version = template.version or "0"

        # Check if the requested version matches
        if version and version != template_version:
            raise TemplateNotFoundError(f"Version {version} not found for template {name}")

        return template

    def list_versions_sync(self, name: str) -> List[VersionEntry]:
        """Synchronous version of alist_versions."""
        template = self.templates.get(name)
        if not template:
            return []

        version = template.version or "0"
        aliases = list(template.aliases)

        return [VersionEntry(id=version, aliases=aliases)]

    def get_template_sync(self, name: str, version: str=None) -> PromptTemplate:
        """Synchronous version of aget_template."""
        template = self.templates.get(name)
        if not template:
            raise TemplateNotFoundError(name)

        template_version = template.version or "0"

        # Check if the requested version matches
        if version and version != template_version:
            raise TemplateNotFoundError(f"Version {version} not found for template {name}")

        return template

    def add_template(self, name: str, template: Union[Dict, PromptTemplate]) -> None:
        """Add or update a template in the loader.
        
        Args:
            name: Template name
            template: Either a dict or PromptTemplate instance
        """
        if isinstance(template, PromptTemplate):
            self.templates[name] = template
        elif isinstance(template, dict):
            if 'yaml' in template and isinstance(template['yaml'], str):
                self.templates[name] = self._parse_yaml_template(name, template)
            else:
                self.templates[name] = PromptTemplate.from_dict(template)
        else:
            raise ValueError(f"Invalid template type: {type(template)}")

    def remove_template(self, name: str) -> bool:
        """Remove a template from the loader.
        
        Args:
            name: Template name to remove
            
        Returns:
            True if template was removed, False if it didn't exist
        """
        if name in self.templates:
            del self.templates[name]
            return True
        return False

    def list_template_names(self) -> List[str]:
        """Get list of all template names."""
        return list(self.templates.keys())

    def has_template(self, name: str) -> bool:
        """Check if template exists."""
        return name in self.templates
