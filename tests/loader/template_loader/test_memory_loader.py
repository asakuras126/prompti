"""Tests for MemoryLoader."""

import pytest
import yaml

from prompti.loader.template_loader.memory import MemoryLoader
from prompti.loader.template_loader.base import TemplateNotFoundError
from prompti.template import PromptTemplate, Variant
from prompti.model_client import ModelConfig


class TestMemoryLoader:
    """Test suite for MemoryLoader."""

    def test_init_with_yaml_format(self):
        """Test initialization with legacy YAML format."""
        yaml_text = """
name: test_template
version: '1.0'
description: Test template
variants:
  base:
    selector: []
    model_cfg:
      provider: openai
      model: gpt-4
    messages:
      - role: user
        content: "Hello {{ name }}"
"""
        loader = MemoryLoader({"test_template": {"yaml": yaml_text}})

        assert loader.has_template("test_template")
        assert "test_template" in loader.list_template_names()

    def test_init_with_dict_format(self):
        """Test initialization with dict format."""
        template_data = {
            "id": "test_template",
            "name": "test_template",
            "version": "1.0",
            "description": "Test template",
            "variants": {
                "base": {
                    "selector": [],
                    "model_cfg": {
                        "provider": "openai",
                        "model": "gpt-4"
                    },
                    "messages": [
                        {"role": "user", "content": "Hello {{ name }}"}
                    ]
                }
            }
        }
        loader = MemoryLoader({"test_template": template_data})

        assert loader.has_template("test_template")

    def test_init_with_prompt_template_instance(self):
        """Test initialization with PromptTemplate instance."""
        template = PromptTemplate(
            id="test_template",
            name="test_template",
            version="1.0",
            description="Test template",
            variants={
                "base": Variant(
                    selector=[],
                    model_cfg=ModelConfig(provider="openai", model="gpt-4"),
                    messages=[{"role": "user", "content": "Hello"}]
                )
            }
        )
        loader = MemoryLoader({"test_template": template})

        assert loader.has_template("test_template")

    @pytest.mark.asyncio
    async def test_alist_versions_success(self):
        """Test listing versions successfully."""
        yaml_text = """
name: test_template
version: '1.0'
aliases: ['latest', 'prod']
variants:
  base:
    selector: []
    messages:
      - role: user
        content: "Hello"
"""
        loader = MemoryLoader({"test_template": {"yaml": yaml_text}})
        versions = await loader.alist_versions("test_template")

        assert len(versions) == 1
        assert versions[0].id == "1.0"
        assert "latest" in versions[0].aliases
        assert "prod" in versions[0].aliases

    @pytest.mark.asyncio
    async def test_alist_versions_not_found(self):
        """Test listing versions for non-existent template."""
        loader = MemoryLoader({})
        versions = await loader.alist_versions("nonexistent")

        assert versions == []

    @pytest.mark.asyncio
    async def test_aget_template_success(self):
        """Test getting template successfully."""
        yaml_text = """
name: test_template
version: '1.0'
variants:
  base:
    selector: []
    messages:
      - role: user
        content: "Hello {{ name }}"
"""
        loader = MemoryLoader({"test_template": {"yaml": yaml_text}})
        template = await loader.aget_template("test_template", "1.0")

        assert isinstance(template, PromptTemplate)
        assert template.name == "test_template"
        assert template.version == "1.0"
        assert "base" in template.variants

    @pytest.mark.asyncio
    async def test_aget_template_not_found(self):
        """Test getting non-existent template raises error."""
        loader = MemoryLoader({})

        with pytest.raises(TemplateNotFoundError):
            await loader.aget_template("nonexistent", "1.0")

    @pytest.mark.asyncio
    async def test_aget_template_wrong_version(self):
        """Test getting template with wrong version raises error."""
        yaml_text = """
name: test_template
version: '1.0'
variants:
  base:
    selector: []
    messages:
      - role: user
        content: "Hello"
"""
        loader = MemoryLoader({"test_template": {"yaml": yaml_text}})

        with pytest.raises(TemplateNotFoundError, match="Version 2.0 not found"):
            await loader.aget_template("test_template", "2.0")

    def test_add_template_with_dict(self):
        """Test adding template with dict."""
        loader = MemoryLoader({})

        template_data = {
            "id": "new_template",
            "name": "new_template",
            "version": "1.0",
            "variants": {
                "base": {
                    "selector": [],
                    "messages": [{"role": "user", "content": "Test"}]
                }
            }
        }

        loader.add_template("new_template", template_data)
        assert loader.has_template("new_template")

    def test_add_template_with_prompt_template(self):
        """Test adding template with PromptTemplate instance."""
        loader = MemoryLoader({})

        template = PromptTemplate(
            id="new_template",
            name="new_template",
            version="1.0",
            variants={
                "base": Variant(selector=[], messages=[])
            }
        )

        loader.add_template("new_template", template)
        assert loader.has_template("new_template")

    def test_remove_template_success(self):
        """Test removing existing template."""
        yaml_text = """
name: test_template
version: '1.0'
variants:
  base:
    selector: []
    messages: []
"""
        loader = MemoryLoader({"test_template": {"yaml": yaml_text}})

        result = loader.remove_template("test_template")
        assert result is True
        assert not loader.has_template("test_template")

    def test_remove_template_not_found(self):
        """Test removing non-existent template."""
        loader = MemoryLoader({})

        result = loader.remove_template("nonexistent")
        assert result is False

    def test_list_template_names(self):
        """Test listing all template names."""
        templates = {
            "template1": {"yaml": "name: t1\nversion: '1'\nvariants:\n  base:\n    selector: []\n    messages: []"},
            "template2": {"yaml": "name: t2\nversion: '1'\nvariants:\n  base:\n    selector: []\n    messages: []"},
        }
        loader = MemoryLoader(templates)

        names = loader.list_template_names()
        assert len(names) == 2
        assert "template1" in names
        assert "template2" in names

    def test_sync_methods(self):
        """Test synchronous versions of async methods."""
        yaml_text = """
name: test_template
version: '1.0'
variants:
  base:
    selector: []
    messages: []
"""
        loader = MemoryLoader({"test_template": {"yaml": yaml_text}})

        # Test list_versions_sync
        versions = loader.list_versions_sync("test_template")
        assert len(versions) == 1
        assert versions[0].id == "1.0"

        # Test get_template_sync
        template = loader.get_template_sync("test_template", "1.0")
        assert isinstance(template, PromptTemplate)
        assert template.version == "1.0"

    def test_invalid_template_type(self):
        """Test initialization with invalid template type."""
        with pytest.raises(ValueError, match="Invalid template data type"):
            MemoryLoader({"test": "invalid_string"})

    def test_yaml_with_messages_template_field(self):
        """Test YAML format with messages_template field (legacy support)."""
        yaml_text = """
name: test_template
version: '1.0'
variants:
  base:
    selector: []
    messages_template:
      - role: user
        content: "Hello"
"""
        loader = MemoryLoader({"test_template": {"yaml": yaml_text}})
        template_sync = loader.get_template_sync("test_template", "1.0")

        # messages_template should be converted to messages
        assert "base" in template_sync.variants
        assert len(template_sync.variants["base"].messages) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
