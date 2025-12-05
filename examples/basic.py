"""Minimal example demonstrating PromptI with LiteLLM."""

from __future__ import annotations

import asyncio

from prompti.engine import PromptEngine, Setting
import logging
from prompti.model_client.base import ModelConfig, RunParams, ToolParams, ToolSpec

logging.basicConfig(level=logging.INFO)

setting = Setting(
    registry_url="http://10.224.55.241/api/v1",
    registry_api_key="7e5d106c-e701-4587-a95a-b7c7c02ee619",
)
engine = PromptEngine.from_setting(setting)


async def stream_call() -> None:
    """Render ``support_reply`` and print the response."""
    try:
        async for msg in engine.acompletion(
            "simple-demo",
            variables={"user_name": "小明",
                       "tasks": [{"name": "task_a", "priority": 2}, {"name": "task_b", "priority": 2}], "urgent": 1},
            stream=True,
            variant="use-jinja2",

        ):
            print(msg)
    finally:
        await engine.aclose()


# 传递模版默认的ModelConfig

async def no_stream_call() -> None:
    """Render ``support_reply`` and print the response."""

    try:
        async for msg in engine.acompletion(
            "simple_demo",
            {"user_query": "hello", "system_prompt": "reply with hello at beginning"},
            stream=False,
        ):
            print(msg)
    finally:
        await engine.aclose()


async def multi_modal_call() -> None:
    """Render ``support_reply`` and print the response."""
    try:
        async for msg in engine.acompletion(
            "simple-demo",
            variables={
                "instruction": "你是图像分析大师",
                "query": "这张图片是什么？",
                # 多张图片使用 "image_url": ["image1_url", "image2_url"]
                "image_url": "https://agentos-promptstore.bj.bcebos.com/files/test/images/default/2989be85-9bfb-4e18-a339-48466320bf0f.jpg?authorization=bce-auth-v1%2Fagentos%2F2025-07-22T07%3A35%3A56Z%2F604800%2F%2F5761014c2df87427a1ab41a1d054820ecbb2f5cd1516e998dd2430a75549764a&response-content-disposition=inline&response-content-type=image%2Fjpeg",
            },
            variant="multimodal",
            stream=False,
            model_cfg=ModelConfig(
                provider="qianfan",
                model="ernie-4.5-turbo-vl-32k"
            )
        ):
            print(msg)
    finally:
        await engine.aclose()


async def tool_call():
    try:
        async for msg in engine.acompletion(
            "simple-demo",
            variables={
                "instruction": "你好",
                "query": "1+1=？"
            },
            stream=False,
            tool_params={"tools": [
                {
                    "name": "get_weather",
                    "description": "获取某地当前天气",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "城市名称，如北京"
                            }
                        },
                        "required": ["location"]
                    }

                },
                {
                    "name": "calculate",
                    "description": "执行基本的数学计算",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "要计算的数学表达式，例如：2+3*4"
                            }
                        },
                        "required": ["expression"]
                    }

                }
            ]}
        ):
            print(msg)
    finally:
        await engine.aclose()


def multi_chat() -> None:
    """Render ``support_reply`` and print the response."""
    try:
        from prompti.message import Message
        messages = Message.get_openai_messages([
                {
                    "role": "system",
                    "content": "你是一个聊天助手",
                },
                {
                    "role": "user",
                    "content": "你好啊",
                },
                {
                    "role": "assistant",
                    "content": "嗨~",
                },
                {
                    "role": "user",
                    "content": "你叫什么名字呀",
                },
            ])
        for msg in engine.completion(
            "simple-demo",
            variables={"instruction": "你是一个聊天助手", "query": "你好啊"},
            stream=True,
            messages=messages
        ):
            print(msg)
    finally:
        await engine.aclose()


def tool_call2() -> None:
    """Render ``support_reply`` and print the response."""
    setting = Setting(
        registry_url="http://10.224.55.241/api/v1",
        registry_api_key="7e5d106c-e701-4587-a95a-b7c7c02ee619",
    )
    engine = PromptEngine.from_setting(setting)

    try:
        async for msg in engine.acompletion(
            "simple-demo",
            variables={"instruction": "你是一个聊天助手", "query": "你好啊"},
            stream=True,
            messages=[
                {
                    "role": "user",
                    "content": "北京现在的天气怎么样？"
                },
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "tool_call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": "{\"location\": \"北京\"}"
                            }
                        }
                    ]
                },
                {
                    "role": "tool",
                    "tool_call_id": "tool_call_1",
                    "content": "{\"temperature\": \"30°C\", \"condition\": \"晴\"}"
                }
            ]
        ):
            print(msg)
    finally:
        await engine.aclose()


if __name__ == "__main__":
    # asyncio.run(stream_call())
    # asyncio.run(tool_call())
    # asyncio.run(tool_call2())

    # asyncio.run(multi_modal_call())
    asyncio.run(multi_chat())
