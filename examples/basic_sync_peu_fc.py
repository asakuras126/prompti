#!/usr/bin/env python3
"""
LiteLLM调用脚本 - 基于basic_sync模式
使用prompti框架调用litellm模型客户端
"""

from __future__ import annotations

import logging
import uuid
from prompti.engine import PromptEngine, Setting
from prompti.model_client.base import ModelConfig
from loguru import logger


class InterceptHandler(logging.Handler):
    def emit(self, record):
        # 将 logging 的消息转给 loguru
        logger_opt = logger.opt(depth=6, exception=record.exc_info)
        logger_opt.log(record.levelname, record.getMessage())


logging.basicConfig(handlers=[InterceptHandler()], level=logging.INFO)

logging.getLogger("prompti.model_client").setLevel(logging.DEBUG)

setting = Setting(
    registry_url="http://10.224.55.241/api/v1",
    registry_api_key="7e5d106c-e701-4587-a95a-b7c7c02ee619",
)

# setting = Setting(
#     registry_url="http://localhost:8080/api/v1",
#     registry_api_key="ccffa346-2e96-4596-b0a3-a63dee6be837",
# )
engine = PromptEngine.from_setting(setting)


# response = completion(
#     # model="bedrock/converse/us.anthropic.claude-3-7-sonnet-20250219-v1:0",  # converse 接口
#     model="bedrock/converse/us.anthropic.claude-sonnet-4-20250514-v1:0",
#     messages=[{"role": "user", "content": "Hello"}],
#     api_base="https://nlb-test.miaoda.io",               # 指定 base_url
#     api_key="ABSKbWQtcmRtKzEtYXQtMTk0NjI1NzQwNjM1OnhsVTBML0VtSEZCbTJjVG1wSXpGQWlHdExvWk93WmRWeFFlZzVET3QzZVZpZURkai9NdXd6cm9vVno0PQ=="                        # 或者配置在环境变量里
# )

model_cfg = ModelConfig(
    provider="pseudo_function_calling",
    model="miaoda_1018",
)
def litellm_basic_call():
    """基础LiteLLM调用示例"""
    try:
        for msg in engine.completion(
            "simple-demo",  # 模板名称
            variables={
                "instruction": "你是一个AI助手",
                "query": "你好，请介绍一下自己",
                "chat_history": "",
            },
            stream=False,
            variant="default",
            # 使用LiteLLM配置
            model_cfg=model_cfg,
            request_id=str(uuid.uuid4()),
            conversation_id=str(uuid.uuid4()),
            user_id=str(uuid.uuid4()),
            timeout=30
        ):
            print(f"Response: {msg}")
    except Exception as e:
        print(f"Error: {e}")


def litellm_stream_call():
    """流式LiteLLM调用示例"""
    try:
        for msg in engine.completion(
            "simple-demo",
            variables={
                "instruction": "你是一个代码助手",
                "query": "请写一个Python函数来计算斐波那契数列",
                "chat_history": "",
            },
            stream=True,  # 启用流式响应
            variant="default",
            model_cfg=model_cfg,
        ):
            print(f"Response: {msg}")
    except Exception as e:
        print(f"Error: {e}")


def litellm_with_tools():
    """带工具调用的LiteLLM示例"""
    try:
        for msg in engine.completion(
            "simple-demo",
            variables={
                "instruction": "你是一个智能助手，可以调用工具",
                "query": "北京现在的天气怎么样？",
                "chat_history": "",
            },
            stream=False,
            variant="default",
            model_cfg=model_cfg,
            tool_params={
                "tools": [
                    {
                        "name": "get_weather",
                        "description": "获取指定城市的天气信息",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "city": {
                                    "type": "string",
                                    "description": "城市名称"
                                }
                            },
                            "required": ["city"]
                        }
                    }
                ]
            }
        ):
            print(f"Tool response: {msg}")
            if msg.error:
                raise RuntimeError()
    except Exception as e:
        print(f"Error: {e}")


def litellm_multimodal_call():
    """LiteLLM多模态调用示例 - 单张图片"""
    try:
        for msg in engine.completion(
            "simple-demo",
            variables={
                "instruction": "你是图像分析大师",
                "query": "请描述这张图片的内容",
                "chat_log": "123",
                "current_time": "2024-01-01 12:00:00",
                # 单张图片URL
                "image_url": [
                    "https://bj.bcebos.com/v1/miaoda-market/%E5%BA%A6%E6%96%AF%E5%8D%A1%E6%8F%92%E5%9B%BE.png?authorization=bce-auth-v1%2FALTAKfC4m4Wgd5CkntOMBgUKjS%2F2025-10-11T08%3A41%3A10Z%2F-1%2Fhost%2F7cc5b6191ccc5db5030dca8aa0b811aba14fccb3a18ebbabd745eb3aec1d35d3",
                    "https://bj.bcebos.com/v1/miaoda-market/%E5%BA%A6%E6%96%AF%E5%8D%A1%E6%8F%92%E5%9B%BE.png?authorization=bce-auth-v1%2FALTAKfC4m4Wgd5CkntOMBgUKjS%2F2025-10-11T08%3A41%3A10Z%2F-1%2Fhost%2F7cc5b6191ccc5db5030dca8aa0b811aba14fccb3a18ebbabd745eb3aec1d35d3",
                    "https://bj.bcebos.com/v1/miaoda-market/%E5%BA%A6%E6%96%AF%E5%8D%A1%E6%8F%92%E5%9B%BE.png?authorization=bce-auth-v1%2FALTAKfC4m4Wgd5CkntOMBgUKjS%2F2025-10-11T08%3A41%3A10Z%2F-1%2Fhost%2F7cc5b6191ccc5db5030dca8aa0b811aba14fccb3a18ebbabd745eb3aec1d35d3",
                    "https://bj.bcebos.com/v1/miaoda-market/%E5%BA%A6%E6%96%AF%E5%8D%A1%E6%8F%92%E5%9B%BE.png?authorization=bce-auth-v1%2FALTAKfC4m4Wgd5CkntOMBgUKjS%2F2025-10-11T08%3A41%3A10Z%2F-1%2Fhost%2F7cc5b6191ccc5db5030dca8aa0b811aba14fccb3a18ebbabd745eb3aec1d35d3",
                    "https://bj.bcebos.com/v1/miaoda-market/%E5%BA%A6%E6%96%AF%E5%8D%A1%E6%8F%92%E5%9B%BE.png?authorization=bce-auth-v1%2FALTAKfC4m4Wgd5CkntOMBgUKjS%2F2025-10-11T08%3A41%3A10Z%2F-1%2Fhost%2F7cc5b6191ccc5db5030dca8aa0b811aba14fccb3a18ebbabd745eb3aec1d35d3",
                    "https://bj.bcebos.com/v1/miaoda-market/%E5%BA%A6%E6%96%AF%E5%8D%A1%E6%8F%92%E5%9B%BE.png?authorization=bce-auth-v1%2FALTAKfC4m4Wgd5CkntOMBgUKjS%2F2025-10-11T08%3A41%3A10Z%2F-1%2Fhost%2F7cc5b6191ccc5db5030dca8aa0b811aba14fccb3a18ebbabd745eb3aec1d35d3",
                    "https://bj.bcebos.com/v1/miaoda-market/%E5%BA%A6%E6%96%AF%E5%8D%A1%E6%8F%92%E5%9B%BE.png?authorization=bce-auth-v1%2FALTAKfC4m4Wgd5CkntOMBgUKjS%2F2025-10-11T08%3A41%3A10Z%2F-1%2Fhost%2F7cc5b6191ccc5db5030dca8aa0b811aba14fccb3a18ebbabd745eb3aec1d35d3"
                    ]
            },
            variant="multi-modal",
            stream=False,
            model_cfg=model_cfg,
        ):
            print(f"Multimodal response: {msg}")
    except Exception as e:
        print(f"Error: {e}")


def litellm_multimodal_multiple_images():
    """LiteLLM多模态调用示例 - 多张图片"""
    try:
        for msg in engine.completion(
            "simple-demo",
            variables={
                "instruction": "你是图像分析专家",
                "query": "请比较这几张图片的差异",
                "chat_log": "123",
                "current_time": "2024-01-01 12:00:00",
                # 多张图片使用数组
                "image_url": [
                    "http://gips0.baidu.com/it/u=1690853528,2506870245&fm=3028&app=3028&f=JPEG&fmt=auto?w=1024&h=1024",
                    "https://miaoda-conversation-file.cdn.bcebos.com/b1f91fbe6fe54d2eaf70ef0025f1c3c2/20250624/file-4bnn1e4a83y8.png"
                ],
            },
            variant="multimodal",
            stream=False,
            model_cfg=model_cfg,
        ):
            print(f"Multiple images response: {msg}")
    except Exception as e:
        print(f"Error: {e}")


def litellm_multimodal_stream():
    """LiteLLM流式多模态调用示例"""
    try:
        for msg in engine.completion(
            "simple-demo",
            variables={
                "instruction": "你是专业的图像识别助手",
                "query": "请详细分析这张图片，包括颜色、构图、主要元素等",
                "chat_log": "123", 
                "current_time": "2024-01-01 12:00:00",
                "image_url": "https://miaoda-conversation-file.cdn.bcebos.com/b1f91fbe6fe54d2eaf70ef0025f1c3c2/20250624/file-4bnn1e4a83y8.png",
            },
            variant="multimodal",
            stream=True,  # 启用流式响应
            model_cfg=model_cfg,
        ):
            print(f"Streaming multimodal: {msg.get_text_content()}", end="", flush=True)
    except Exception as e:
        print(f"Error: {e}")


def litellm_custom_provider():
    """自定义LiteLLM provider示例（如Claude、Azure等）"""
    try:
        for msg in engine.completion(
            "simple-demo",
            variables={
                "instruction": "你是Claude助手",
                "query": "请解释什么是机器学习",
                "chat_history": "",
            },
            stream=False,
            variant="default",
            model_cfg=model_cfg,
        ):
            print(f"Claude response: {msg}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # print("=== LiteLLM基础调用 ===")
    # litellm_basic_call()

    # # print("\n=== LiteLLM流式调用 ===")
    # litellm_stream_call()
    #
    # # print("\n=== LiteLLM工具调用 ===")
    litellm_with_tools()
    import time
    time.sleep(10)
    #
    # # print("\n=== LiteLLM多模态调用（单图片） ===")
    # litellm_multimodal_call()
    #
    # # print("\n=== LiteLLM多模态调用（多图片） ===")
    # litellm_multimodal_multiple_images()
    #
    # # print("\n=== LiteLLM流式多模态调用 ===")
    # litellm_multimodal_stream()

