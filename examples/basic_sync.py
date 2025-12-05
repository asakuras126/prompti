"""Minimal example demonstrating PromptI with synchronous completion."""

from __future__ import annotations

import uuid
import time
from prompti.engine import PromptEngine, Setting, Message
from prompti.model_client.base import ModelConfig, RunParams, ToolParams, ToolSpec

setting = Setting(
    registry_url="http://10.224.55.241/api/v1",
    registry_api_key="7e5d106c-e701-4587-a95a-b7c7c02ee619",
)
engine = PromptEngine.from_setting(setting)

# gpt-5-codex
# model_cfg = ModelConfig(
#     provider="openai_response",
#     model="gpt-5-codex-eval"
# )
# gpt-5
# model_cfg = ModelConfig(
#     provider="openai_response",
#     model="gpt-5"
# )
# claude-sonnet-4-5-20250929 需要访问外网
model_cfg = ModelConfig(
    provider="aws-litellm",
    model="claude-sonnet-4-5-20250929"
)



def stream_call() -> None:
    """Render ``simple-demo`` and print the response using sync completion."""
    try:
        for msg in engine.completion(
            "simple-demo",
            version="^1#release",
            variables={
                "instruction": "你是图像分析大师",
                "query": "你好",
                "chat_history": "",
                "task_type": "aa",
                "current_time": "xx",
                "chat_log": "123",
                "image_url": "https://miaoda-conversation-file.cdn.bcebos.com/b1f91fbe6fe54d2eaf70ef0025f1c3c2/20250624/file-4bnn1e4a83y8.png",
            },
            stream=True,
            variant="default",
            request_id=str(uuid.uuid4()),
            conversation_id=str(uuid.uuid4()),
            user_id=str(uuid.uuid4()),
            span_id=str(uuid.uuid4()),
            parant_span_id=str(uuid.uuid4()),
            model_cfg=model_cfg,
        ):
            print(f"user_get_response: {msg}")
    finally:
        # Note: In sync version, we don't need await
        # engine.close() is not async in sync context
        pass


def no_stream_call() -> None:
    """Render template and print the response without streaming."""

    try:
        for msg in engine.completion(
            "aaa",
            variables={
                "instruction": "",
                "query": "你好啊",
                "chat_history": "",
                "user_name": "小明",
                "tasks": [
                    {"name": "task_a", "priority": 2},
                    {"name": "task_b", "priority": 2},
                ],
                "urgent": 1,
            },
            stream=False,
            variant="default",
            model_cfg=model_cfg,
            temperature=1,
            # timeout=5
        ):
            print(msg)
    finally:
        pass


def multi_modal_call() -> None:
    """Render multimodal template and print the response."""

    try:
        for msg in engine.completion(
            "simple-demo",
            variables={
                "instruction": "你是图像分析大师",
                "query": "这张图片是什么？",
                "chat_log": "123",
                "current_time": "123",
                # 多张图片使用 "image_url": ["image1_url", "image2_url"]
                "image_url": "https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg",
            },
            variant="multimodal",
            stream=False,
            model_cfg=model_cfg,
        ):
            print(msg)
    finally:
        pass


def tool_call():
    """Demonstrate tool calling with sync completion."""

    try:
        for msg in engine.completion(
            "simple-demo",
            variables={"instruction": "你好", "query": "1+1=？", "chat_history": "123"},
            variant="default",
            stream=False,
            model_cfg=model_cfg,
            tool_params={
                "tools": [
                    {
                        "name": "get_weather",
                        "description": "获取某地当前天气",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "城市名称，如北京",
                                }
                            },
                            "required": ["location"],
                        },
                    },
                    {
                        "name": "calculate",
                        "description": "执行基本的数学计算",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "expression": {
                                    "type": "string",
                                    "description": "要计算的数学表达式，例如：2+3*4",
                                }
                            },
                            "required": ["expression"],
                        },
                    },
                ]
            },
        ):
            print(msg)
    finally:
        pass


def multi_chat() -> None:
    """Demonstrate multi-turn chat with sync completion."""

    try:
        ans = ""
        for msg in engine.completion(
            "simple-demo",
            version="^1#release",
            variables={"instruction": "你好", "query": "1+1=？", "chat_history": 123},
            variant="default",
            stream=False,
            messages=[
                {
                    "role": "user",
                    "content": """*** 角色定xxx""",
                }
            ],
            model_cfg=model_cfg,
            timeout=50,
            conversation_id="aaa"
        ):
            print(f"\n\n\n user_get msg: {msg} \n\n\n")
            ans += msg.get_text_content() or ""
        print(repr(ans))
    finally:
        pass


if __name__ == "__main__":
    stream_call()
    no_stream_call()
    multi_modal_call()
    tool_call()
    multi_chat()
    time.sleep(3)


