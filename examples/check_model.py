"""Model compatibility testing for PromptI."""

from __future__ import annotations

import asyncio
import logging
from prompti.engine import PromptEngine, Setting
from prompti.model_client.base import ModelConfig, RunParams, ToolParams, ToolSpec

logging.basicConfig(level=logging.INFO)

model_call_dict = [
    # {
    #     "provider": "qianfan",
    #     "model": "ernie-4.0-8k"
    # },
    # {
    #
    #     "model": "ernie-3.5-8k",
    #     "provider": "qianfan",
    # },
    # {
    #     "model": "ernie-4.0-turbo-8k",
    #     "provider": "qianfan",
    # },
    # {
    #     "model": "ernie-4.5-turbo-32k",
    #     "provider": "qianfan",
    # },
    # {
    #     "model": "ernie-4.5-turbo-128k",
    #     "provider": "qianfan",
    # },
    # {
    #     "model": "ernie-4.0-turbo-128k",
    #     "provider": "qianfan",
    # },
    # {
    #     "model": "ernie-4.5-turbo-vl-32k",
    #     "provider": "qianfan",
    # },
    # {
    #     "model": "deepseek-v3",
    #     "provider": "qianfan",
    # },
    # {
    #     "model": "deepseek-r1",
    #     "provider": "qianfan",
    # },
    {
        "provider": "openai",
        "model": "gpt-4o",
    },
    {
        "provider": "openai",
        "model": "gpt-4.1"
    },
    {
        "provider": "openai",
        "model": "gpt-o4-mini"
    },
    {
        "provider": "openai",
        "model": "gemini-2.5-flash"
    },
    {
        "provider": "openai",
        "model": "claude-sonnet-4-20250514"
    },
    {
        "provider": "openai",
        "model": "claude-opus-4-20250514"
    },
    {
        "provider": "openai",
        "model": "claude-3-7-sonnet-20250219"
    },
    {
        "provider": "openai",
        "model": "claude-3-5-sonnet-20241022"
    }
]
setting = Setting(
    registry_url="http://localhost:8080/api/v1",
    registry_api_key="ccffa346-2e96-4596-b0a3-a63dee6be837",
)
engine = PromptEngine.from_setting(setting, )

async def test_stream_call(model_config: dict) -> None:
    """Test streaming call for a specific model."""
    print(f"\n=== Testing Streaming Call: {model_config['provider']}/{model_config['model']} ===")


    try:
        async for msg in engine.acompletion(
            "simple-demo",
            version="1.0.5",
            variables={"instruction": "请简单回答", "query": "你好"},
            model_cfg={
                "provider": model_config["provider"],
                "model": model_config["model"],
                "temperature": 0.7,
                "max_tokens": 100
            },
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant.",
                },
                {
                    "role": "user",
                    "content": "Hello, please reply briefly.",
                },
            ],
            stream=True,
        ):
            print(f"Stream response: {msg}")
        print(f"✅ Streaming call successful for {model_config['provider']}/{model_config['model']}")
    except Exception as e:
        print(f"❌ Streaming call failed for {model_config['provider']}/{model_config['model']}: {e}")
    finally:
        await engine.close()


async def test_no_stream_call(model_config: dict) -> None:
    """Test non-streaming call for a specific model."""
    print(f"\n=== Testing Non-Streaming Call: {model_config['provider']}/{model_config['model']} ===")
    try:
        async for msg in engine.acompletion(
            "simple-demo",
            version="1.0.5",
            variables={"instruction": "请简单回答", "query": "你好"},
            model_cfg={
                "provider": model_config["provider"],
                "model": model_config["model"],
                "temperature": 0.7,
                "max_tokens": 100
            },
            stream=False,
        ):
            print(f"Response: {msg}")
        print(f"✅ Non-streaming call successful for {model_config['provider']}/{model_config['model']}")
    except Exception as e:
        print(f"❌ Non-streaming call failed for {model_config['provider']}/{model_config['model']}: {e}")
    finally:
        await engine.close()


async def test_multi_modal_call(model_config: dict) -> None:
    """Test multi-modal call for a specific model."""
    print(f"\n=== Testing Multi-Modal Call: {model_config['provider']}/{model_config['model']} ===")

    try:
        async for msg in engine.acompletion(
            "simple-demo",
            variables={
                "instruction": "你是一个图片分析大师", "query": "这张图片描述了什么",
                "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
            },
            stream=False,
            model_cfg=ModelConfig(
                provider=model_config["provider"],
                model=model_config["model"],
                temperature=0.7,
                max_tokens=200
            ),
            variant="multimodal"
        ):
            print(f"Multi-modal response: {msg}")
        print(f"✅ Multi-modal call successful for {model_config['provider']}/{model_config['model']}")
    except Exception as e:
        print(f"❌ Multi-modal call failed for {model_config['provider']}/{model_config['model']}: {e}")
    finally:
        await engine.close()


async def test_multi_turn_conversation(model_config: dict) -> None:
    """Test multi-turn conversation for a specific model."""
    print(f"\n=== Testing Multi-Turn Conversation: {model_config['provider']}/{model_config['model']} ===")

    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "My name is Alice."},
            {"role": "assistant", "content": "Hello Alice! Nice to meet you."},
            {"role": "user", "content": "What's my name?"},
        ]

        async for msg in engine.acompletion(
            "simple-demo",
            version="^1",
            variables={"instruction": "请记住对话历史", "query": "回答问题"},
            model_cfg={
                "provider": model_config["provider"],
                "model": model_config["model"],
                "temperature": 0.7,
                "max_tokens": 100
            },
            messages=messages,
            stream=False,
        ):
            print(f"Multi-turn response: {msg}")
        print(f"✅ Multi-turn conversation successful for {model_config['provider']}/{model_config['model']}")
    except Exception as e:
        print(f"❌ Multi-turn conversation failed for {model_config['provider']}/{model_config['model']}: {e}")
    finally:
        await engine.close()


async def test_function_call(model_config: dict) -> None:
    """Test function calling for a specific model."""
    print(f"\n=== Testing Function Call: {model_config['provider']}/{model_config['model']} ===")

    try:
        weather_tool = ToolSpec(
            name="get_weather",
            description="获取指定城市的当前天气信息",
            parameters={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，例如：北京、上海、广州"
                    }
                },
                "required": ["city"]
            }
        )

        tool_params = ToolParams(
            tools=[weather_tool],
            choice="auto"
        )

        async for msg in engine.acompletion(
            "simple-demo",
            version="^1",
            variables={
                "instruction": "如果用户询问天气，请使用工具获取信息",
                "query": "北京今天天气怎么样？"
            },
            stream=False,
            model_cfg={
                "provider": model_config["provider"],
                "model": model_config["model"],
                "temperature": 0.7,
                "max_tokens": 200
            },
            tool_params=tool_params.model_dump(),
        ):
            print(f"Function call response: {msg}")
        print(f"✅ Function call successful for {model_config['provider']}/{model_config['model']}")
    except Exception as e:
        print(f"❌ Function call failed for {model_config['provider']}/{model_config['model']}: {e}")
    finally:
        await engine.close()


async def test_all_models():
    """Test all model configurations with all features."""
    print("🚀 Starting comprehensive model testing...")

    for i, model_config in enumerate(model_call_dict, 1):
        print(f"\n{'=' * 60}")
        print(f"Testing Model {i}/{len(model_call_dict)}: {model_config['provider']}/{model_config['model']}")
        print(f"{'=' * 60}")

        await test_stream_call(model_config)
        await asyncio.sleep(310)

        await test_no_stream_call(model_config)
        await asyncio.sleep(310)

        await test_multi_modal_call(model_config)
        await asyncio.sleep(310)

        await test_multi_turn_conversation(model_config)
        await asyncio.sleep(310)

        await test_function_call(model_config)
        await asyncio.sleep(310)

    print(f"\n{'=' * 60}")
    print("🎉 All model testing completed!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    asyncio.run(test_all_models())
