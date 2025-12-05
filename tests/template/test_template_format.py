from pathlib import Path

import pytest

from prompti.loader import FileSystemLoader
from prompti.model_client import ModelConfig
from prompti.template import PromptTemplate, Variant


@pytest.mark.asyncio
async def test_load_from_file_has_expected_fields():
    # 使用相对于当前文件的路径
    test_dir = Path(__file__).parent.parent / "configs" / "prompts"
    loader = FileSystemLoader(test_dir)
    versions = await loader.alist_versions("summary")
    assert len(versions) > 0
    version = versions[0].id
    tmpl = await loader.aget_template("summary", version)
    assert version == "1.0"
    assert tmpl.name == "summary"
    assert tmpl.version == "1.0"
    assert tmpl.aliases == ["prod", "latest"]
    assert "default" in tmpl.variants


def test_single_message_format():
    template = PromptTemplate(
        name="hello",
        description="",
        version="1.0",
        aliases=["test"],
        variants={
            "base": Variant(
                selector=[],
                model_config=ModelConfig(provider="dummy", model="x"),
                messages=[
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "Hello {{ name }}!"}],
                    }
                ],
            )
        },
    )
    msgs, _ = template.format({"name": "World"}, variant="base")
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == [{"type": "text", "text": "Hello World!"}]


def test_multi_message_format():
    template = PromptTemplate(
        name="test",
        description="",
        version="1.0",
        aliases=["multi_test"],
        variants={
            "base": Variant(
                selector=[],
                model_config=ModelConfig(provider="dummy", model="x"),
                messages=[
                    {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant"}]},
                    {"role": "user", "content": [{"type": "text", "text": "Hello {{ name }}!"}]},
                ],
            )
        },
    )
    msgs, _ = template.format({"name": "World"}, variant="base")
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == [{"type": "text", "text": "You are a helpful assistant"}]
    assert msgs[1]["role"] == "user"
    assert msgs[1]["content"] == [{"type": "text", "text": "Hello World!"}]


def test_complex_jinja_multi_message():
    template = PromptTemplate(
        name="task",
        description="",
        version="1.0",
        aliases=["task_management"],
        variants={
            "base": Variant(
                selector=[],
                model_config=ModelConfig(provider="dummy", model="x"),
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Task Report:\n"
                                    "{% for t in tasks %}"
                                    "{{ '- ' + t.name }} ({{ t.priority }})\n"
                                    "{% endfor %}"
                                ),
                            }
                        ],
                    }
                ],
            )
        },
    )
    tasks = [{"name": "Fix", "priority": 1}, {"name": "Doc", "priority": 2}]
    msgs, _ = template.format({"tasks": tasks}, variant="base")
    content = msgs[0]["content"][0]["text"]
    assert "Fix" in content and "Doc" in content


def test_image_url_support():
    """Test support for image_url type with template variables."""
    template = PromptTemplate(
        name="image_test",
        description="",
        version="1.0",
        aliases=["image_analysis"],
        variants={
            "base": Variant(
                selector=[],
                model_config=ModelConfig(provider="dummy", model="x"),
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Analyze this image:"},
                            {"type": "image_url", "image_url": "{{ image_url }}"}
                        ]
                    }
                ],
            )
        },
    )
    
    msgs, _ = template.format({"image_url": "https://example.com/image.jpg"}, variant="base")
    assert len(msgs) == 1
    assert msgs[0]["content"][0]["text"] == "Analyze this image:"
    # image_url is wrapped in a dict with 'url' key after formatting
    assert msgs[0]["content"][1]["image_url"]["url"] == "https://example.com/image.jpg"


def test_string_content_support():
    """Test support for string content (non-list format)."""
    template = PromptTemplate(
        name="string_test",
        description="",
        version="1.0",
        aliases=["string_content"],
        variants={
            "base": Variant(
                selector=[],
                model_config=ModelConfig(provider="dummy", model="x"),
                messages=[
                    {
                        "role": "user",
                        "content": "Hello {{ name }}!"
                    }
                ],
            )
        },
    )
    
    msgs, _ = template.format({"name": "World"}, variant="base")
    assert len(msgs) == 1
    assert msgs[0]["content"] == "Hello World!"


def test_aliases_field():
    """Test that aliases field is properly handled."""
    template = PromptTemplate(
        name="aliases_test",
        description="Test template",
        version="1.0",
        aliases=["test_alias", "another_alias"],
        variants={
            "default": Variant(
                selector=[],
                model_config=ModelConfig(provider="dummy", model="x"),
                messages=[
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "Hello {{ name }}!"}],
                    }
                ],
            )
        },
    )
    
    assert template.aliases == ["test_alias", "another_alias"]
    assert "test_alias" in template.aliases
    assert "another_alias" in template.aliases


def test_choose_variant_with_aliases():
    """Test variant selection with aliases in selector."""
    template = PromptTemplate(
        name="variant_test",
        description="",
        version="1.0",
        aliases=["variant_alias"],
        variants={
            "vip": Variant(
                selector=["vip"],
                model_config=ModelConfig(provider="dummy", model="x"),
                messages=[
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "VIP: Hello {{ name }}!"}],
                    }
                ],
            ),
            "guest": Variant(
                selector=["guest"],
                model_config=ModelConfig(provider="dummy", model="x"),
                messages=[
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "Guest: Hello {{ name }}!"}],
                    }
                ],
            )
        },
    )
    
    # Test VIP variant selection
    variant = template.choose_variant({"role": "vip-user"})
    assert variant == "vip"
    
    # Test guest variant selection
    variant = template.choose_variant({"role": "guest-user"})
    assert variant == "guest"
    
    # Test no match
    variant = template.choose_variant({"role": "unknown"})
    assert variant is None
