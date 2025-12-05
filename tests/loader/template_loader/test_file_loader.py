"""Tests for FileLoader."""

import pytest
import yaml
from pathlib import Path
from tempfile import TemporaryDirectory

from prompti.loader.template_loader.file import FileLoader, FileSystemLoader
from prompti.loader.template_loader.base import TemplateNotFoundError
from prompti.template import PromptTemplate


class TestFileLoader:
    """Test suite for FileLoader."""

    def test_file_system_loader_alias(self):
        """Test that FileSystemLoader is an alias for FileLoader."""
        assert FileSystemLoader is FileLoader

    @pytest.mark.asyncio
    async def test_alist_versions_success(self):
        """Test listing versions successfully."""
        with TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)
            template_file = base_path / "test_template.yaml"

            yaml_content = """
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
            template_file.write_text(yaml_content)

            loader = FileLoader(base_path)
            versions = await loader.alist_versions("test_template")

            assert len(versions) == 1
            assert versions[0].id == "1.0"
            assert "latest" in versions[0].aliases
            assert "prod" in versions[0].aliases

    @pytest.mark.asyncio
    async def test_alist_versions_file_not_found(self):
        """Test listing versions for non-existent file."""
        with TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)
            loader = FileLoader(base_path)
            versions = await loader.alist_versions("nonexistent")

            assert versions == []

    @pytest.mark.asyncio
    async def test_alist_versions_invalid_yaml(self):
        """Test listing versions with invalid YAML."""
        with TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)
            template_file = base_path / "bad_template.yaml"
            template_file.write_text("invalid: yaml: content:")

            loader = FileLoader(base_path)
            versions = await loader.alist_versions("bad_template")

            assert versions == []

    @pytest.mark.asyncio
    async def test_aget_template_success(self):
        """Test getting template successfully."""
        with TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)
            template_file = base_path / "test_template.yaml"

            yaml_content = """
name: test_template
version: '1.0'
description: Test description
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
            template_file.write_text(yaml_content)

            loader = FileLoader(base_path)
            template = await loader.aget_template("test_template", "1.0")

            assert isinstance(template, PromptTemplate)
            assert template.name == "test_template"
            assert template.version == "1.0"
            assert template.description == "Test description"
            assert "base" in template.variants

    @pytest.mark.asyncio
    async def test_aget_template_file_not_found(self):
        """Test getting non-existent template returns None."""
        with TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)
            loader = FileLoader(base_path)
            template = await loader.aget_template("nonexistent", "1.0")

            assert template is None

    @pytest.mark.asyncio
    async def test_aget_template_wrong_version(self):
        """Test getting template with wrong version returns None."""
        with TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)
            template_file = base_path / "test_template.yaml"

            yaml_content = """
name: test_template
version: '1.0'
variants:
  base:
    selector: []
    messages: []
"""
            template_file.write_text(yaml_content)

            loader = FileLoader(base_path)
            template = await loader.aget_template("test_template", "2.0")

            assert template is None

    @pytest.mark.asyncio
    async def test_aget_template_no_version_specified(self):
        """Test getting template without specifying version."""
        with TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)
            template_file = base_path / "test_template.yaml"

            yaml_content = """
name: test_template
version: '1.0'
variants:
  base:
    selector: []
    messages: []
"""
            template_file.write_text(yaml_content)

            loader = FileLoader(base_path)
            template = await loader.aget_template("test_template", "")

            assert isinstance(template, PromptTemplate)
            assert template.version == "1.0"

    @pytest.mark.asyncio
    async def test_model_strategy_compatibility(self):
        """Test model_strategy field compatibility handling."""
        with TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)
            template_file = base_path / "test_template.yaml"

            yaml_content = """
name: test_template
version: '1.0'
variants:
  base:
    selector: []
    model_cfg:
      provider: openai
      model: gpt-4
      model_strategy:
        primary: gpt-4
        fallback: gpt-3.5-turbo
    messages: []
"""
            template_file.write_text(yaml_content)

            loader = FileLoader(base_path)
            template = await loader.aget_template("test_template", "1.0")

            assert template is not None
            variant = template.variants["base"]
            # model_strategy should be extracted from model_cfg
            assert variant.model_strategy is not None

    @pytest.mark.asyncio
    async def test_model_cfg_without_strategy(self):
        """Test model_cfg without model_strategy field."""
        with TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)
            template_file = base_path / "test_template.yaml"

            yaml_content = """
name: test_template
version: '1.0'
variants:
  base:
    selector: []
    model_cfg:
      provider: openai
      model: gpt-4
    messages: []
"""
            template_file.write_text(yaml_content)

            loader = FileLoader(base_path)
            template = await loader.aget_template("test_template", "1.0")

            assert template is not None
            variant = template.variants["base"]
            assert variant.model_cfg is not None
            assert variant.model_cfg.provider == "openai"
            assert variant.model_cfg.model == "gpt-4"

    @pytest.mark.asyncio
    async def test_empty_model_cfg(self):
        """Test variant with empty model_cfg."""
        with TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)
            template_file = base_path / "test_template.yaml"

            yaml_content = """
name: test_template
version: '1.0'
variants:
  base:
    selector: []
    model_cfg: null
    messages: []
"""
            template_file.write_text(yaml_content)

            loader = FileLoader(base_path)
            template = await loader.aget_template("test_template", "1.0")

            assert template is not None
            variant = template.variants["base"]
            assert variant.model_cfg is None

    def test_sync_methods(self):
        """Test synchronous versions of async methods."""
        with TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)
            template_file = base_path / "test_template.yaml"

            yaml_content = """
name: test_template
version: '1.0'
variants:
  base:
    selector: []
    messages: []
"""
            template_file.write_text(yaml_content)

            loader = FileLoader(base_path)

            # Test list_versions_sync
            versions = loader.list_versions_sync("test_template")
            assert len(versions) == 1
            assert versions[0].id == "1.0"

            # Test get_template_sync
            template = loader.get_template_sync("test_template", "1.0")
            assert isinstance(template, PromptTemplate)
            assert template.version == "1.0"

    def test_sync_list_versions_file_not_found(self):
        """Test sync list_versions for non-existent file."""
        with TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)
            loader = FileLoader(base_path)
            versions = loader.list_versions_sync("nonexistent")

            assert versions == []

    def test_sync_get_template_file_not_found(self):
        """Test sync get_template for non-existent file."""
        with TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)
            loader = FileLoader(base_path)
            template = loader.get_template_sync("nonexistent", "1.0")

            assert template is None

    @pytest.mark.asyncio
    async def test_multiple_variants(self):
        """Test template with multiple variants."""
        with TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)
            template_file = base_path / "test_template.yaml"

            yaml_content = """
name: test_template
version: '1.0'
variants:
  base:
    selector: []
    messages:
      - role: user
        content: "Base variant"
  advanced:
    selector: []
    messages:
      - role: user
        content: "Advanced variant"
"""
            template_file.write_text(yaml_content)

            loader = FileLoader(base_path)
            template = await loader.aget_template("test_template", "1.0")

            assert len(template.variants) == 2
            assert "base" in template.variants
            assert "advanced" in template.variants


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
