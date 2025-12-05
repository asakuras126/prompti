from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from prompti.engine import PromptEngine, Setting
from prompti.loader import (
    FileSystemLoader,
    MemoryLoader,
    TemplateLoader,
    TemplateNotFoundError,
)
from prompti.message import Message, ModelResponse
from prompti.model_client import ModelClient, ModelConfig, RunParams
from prompti.loader.model_config_loader import ModelConfigLoader
from prompti.template import PromptTemplate, Variant


class DummyClient(ModelClient):
    provider = "dummy"

    async def _run(self, params: RunParams):
        yield ModelResponse(
            id="test_id",
            model=self.cfg.model,
            choices=[{
                "index": 0,
                "message": Message(role="assistant", content="ok"),
                "finish_reason": "stop"
            }]
        )


class DummyConfigLoader(ModelConfigLoader):
    def __init__(self, cfg: ModelConfig) -> None:
        self.cfg = cfg
        self.models = {cfg.model: [cfg]}

    def load(self):
        return self.cfg

    def get_model_config(self, model: str, provider: str=None, model_value: str=None, llm_token: str=None, model_control=None):
        if model == self.cfg.model:
            return [self.cfg]
        raise ValueError(f"Model {model} not found")

    def list_models(self):
        return [self.cfg.model]


@pytest.mark.asyncio
async def test_engine_run_uses_model_cfg():
    yaml_text = """
name: x
version: '1'
variants:
  base:
    selector: []
    model_cfg:
      provider: dummy
      model: x
    messages:
      - role: user
        content: "Hello"
"""
    # 使用补丁替换 create_client 函数，以便我们可以检查配置
    with patch("prompti.engine.create_client") as mock_create_client:
        # 创建一个返回 DummyClient 的模拟
        dummy_client = DummyClient(ModelConfig(provider="dummy", model="x"))
        mock_create_client.return_value = dummy_client

        # 使用内存模板加载器创建引擎
        engine = PromptEngine([MemoryLoader({"x": {"yaml": yaml_text}})])

        # 运行模板 - 使用 acompletion 替代 run
        out = [m async for m in engine.acompletion("x", {}, variant="base", stream=False)]

        # 验证结果
        assert isinstance(out[0], ModelResponse)
        assert out[0].choices[0].message.content == "ok"

        # 验证创建客户端时使用了正确的配置
        mock_create_client.assert_called_once()
        cfg_arg = mock_create_client.call_args[0][0]
        assert cfg_arg.provider == "dummy"
        assert cfg_arg.model == "x"


@pytest.mark.asyncio
async def test_load_returns_template():
    # 使用相对于当前文件的路径
    test_dir = Path(__file__).parent / "configs" / "prompts"
    engine = PromptEngine([FileSystemLoader(test_dir)])
    tmpl = await engine.aload("summary")
    assert isinstance(tmpl, PromptTemplate)
    assert tmpl.name == "summary"
    assert tmpl.version == "1.0"


@pytest.mark.asyncio
async def test_load_caches_result():
    from prompti.loader.template_loader.base import VersionEntry

    class CountingLoader(TemplateLoader):
        def __init__(self):
            self.calls = 0

        async def alist_versions(self, name: str) -> list[VersionEntry]:
            return [VersionEntry(id="1", aliases=[])]

        async def aget_template(self, name: str, version: str) -> PromptTemplate:
            self.calls += 1
            return PromptTemplate(
                id=name,
                name=name,
                description="",
                version="1",
                variants={
                    "base": Variant(
                        selector=[],
                        model_config=ModelConfig(provider="dummy", model="x"),
                        messages=[],
                    )
                },
            )

    loader = CountingLoader()
    engine = PromptEngine([loader])
    tmpl1 = await engine.aload("demo")
    tmpl2 = await engine.aload("demo")
    assert tmpl1 is tmpl2
    assert loader.calls == 1


@pytest.mark.asyncio
async def test_load_missing_raises():
    engine = PromptEngine([FileSystemLoader(Path("./prompts"))])
    with pytest.raises(TemplateNotFoundError):
        await engine.aload("nonexistent")


@pytest.mark.asyncio
async def test_global_model_config_used_when_variant_missing():
    yaml_text = """
name: x
version: '1'
variants:
  base:
    selector: []
    messages:
      - role: user
        content: "Hello"
"""
    # 创建配置和引擎
    global_cfg = ModelConfig(provider="dummy", model="z")
    setting = Setting(
        memory_templates={"x": {"yaml": yaml_text}},
        default_model_config=global_cfg,
    )

    # 使用patch监控create_client
    with patch("prompti.engine.create_client") as mock_create_client:
        # 设置模拟客户端
        dummy_client = DummyClient(global_cfg)
        mock_create_client.return_value = dummy_client

        # 创建引擎和运行测试 - 使用 acompletion 替代 run
        engine = PromptEngine.from_setting(setting)
        out = [m async for m in engine.acompletion("x", {}, variant="base", stream=False)]

        # 验证结果
        assert isinstance(out[0], ModelResponse)
        assert out[0].choices[0].message.content == "ok"

        # 验证使用了全局配置
        mock_create_client.assert_called_once()
        cfg_arg = mock_create_client.call_args[0][0]
        assert cfg_arg.provider == "dummy"
        assert cfg_arg.model == "z"


@pytest.mark.asyncio
async def test_variant_overrides_global_model_config():
    yaml_text = """
name: x
version: '1'
variants:
  base:
    selector: []
    model_cfg:
      provider: dummy
      model: x
    messages:
      - role: user
        content: "Hello"
"""
    # 创建配置和引擎
    global_cfg = ModelConfig(provider="dummy", model="z")
    setting = Setting(
        memory_templates={"x": {"yaml": yaml_text}},
        default_model_config=global_cfg,
    )

    # 使用patch监控create_client
    with patch("prompti.engine.create_client") as mock_create_client:
        # 设置模拟客户端
        dummy_client = DummyClient(ModelConfig(provider="dummy", model="x"))
        mock_create_client.return_value = dummy_client

        # 创建引擎和运行测试 - 使用 acompletion 替代 run
        engine = PromptEngine.from_setting(setting)
        [m async for m in engine.acompletion("x", {}, variant="base", stream=False)]

        # 验证使用了变体配置而不是全局配置
        mock_create_client.assert_called_once()
        cfg_arg = mock_create_client.call_args[0][0]
        assert cfg_arg.provider == "dummy"
        assert cfg_arg.model == "x"  # 变体配置的模型


@pytest.mark.asyncio
async def test_error_when_no_model_config_available():
    yaml_text = """
name: x
version: '1'
variants:
  base:
    selector: []
    messages:
      - role: user
        content: "Hello"
"""
    # 创建没有全局配置的引擎
    engine = PromptEngine([MemoryLoader({"x": {"yaml": yaml_text}})])

    # 没有模型配置应该引发错误 - 使用 acompletion 替代 run
    with pytest.raises(ValueError, match="ModelConfig required but not provided in template or globally"):
        [m async for m in engine.acompletion("x", {}, variant="base", stream=False)]


@pytest.mark.asyncio
async def test_engine_format():
    yaml_text = """
name: greet
version: '1'
variants:
  base:
    selector: []
    messages:
      - role: user
        content: "Hello {{ name }}"
"""

    engine = PromptEngine([MemoryLoader({"greet": {"yaml": yaml_text}})])
    msgs = await engine.aformat("greet", {"name": "Ada"}, variant="base")
    assert isinstance(msgs, list)
    assert len(msgs) == 1
    assert msgs[0].get("role") == "user"
    assert msgs[0].get("content") == "Hello Ada"


@pytest.mark.asyncio
async def test_direct_messages():
    """测试直接使用messages参数的情况"""
    # 创建直接使用的消息
    direct_messages = [
        Message(role="system", content="You are a helpful assistant"),
        Message(role="user", content="Hello!")
    ]

    # 创建模型配置
    model_cfg = ModelConfig(provider="dummy", model="direct_model")

    # 创建一个简单的模板对象用于variant配置
    dummy_template = PromptTemplate(
        id="direct",
        name="direct",
        description="Direct messages template",
        version="1",
        variants={
            "default": Variant(
                selector=[],
                model_config=model_cfg,
                messages=[],
            )
        },
    )

    # 使用patch监控create_client
    with patch("prompti.engine.create_client") as mock_create_client:
        # 设置模拟客户端
        dummy_client = DummyClient(model_cfg)
        mock_create_client.return_value = dummy_client

        # 创建引擎
        engine = PromptEngine([])  # 不需要加载器

        # 运行测试，使用直接传入的messages和template对象
        out = [m async for m in engine.acompletion(
            template_name="direct",
            variables={},
            template=dummy_template,  # 传递模板对象而不是解析
            model_cfg=model_cfg,
            messages=direct_messages,
            stream=False
        )]

        # 验证结果
        assert isinstance(out[0], ModelResponse)
        assert out[0].choices[0].message.content == "ok"

        # 验证传递给create_client的参数
        mock_create_client.assert_called_once()
        cfg_arg = mock_create_client.call_args[0][0]
        assert cfg_arg.provider == "dummy"
        assert cfg_arg.model == "direct_model"

        # 由于我们无法直接访问客户端的run方法调用参数
        # 我们只验证模型配置和结果
        assert out[0].model == "direct_model"
