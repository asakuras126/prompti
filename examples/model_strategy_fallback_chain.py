#!/usr/bin/env python3
"""
Fallback Chain 真实环境演示 - 基于promptstore
使用真实的promptstore配置，演示fallback_chain在实际场景中的使用
"""

from __future__ import annotations

import logging
import uuid
import asyncio
from prompti.engine import PromptEngine, Setting
from prompti.model_client.base import ModelConfig
from loguru import logger


class InterceptHandler(logging.Handler):
    def emit(self, record):
        logger_opt = logger.opt(depth=6, exception=record.exc_info)
        logger_opt.log(record.levelname, record.getMessage())


logging.basicConfig(handlers=[InterceptHandler()], level=logging.INFO)


setting = Setting(
    registry_url="http://localhost:8080/api/v1",
    registry_api_key="ccffa346-2e96-4596-b0a3-a63dee6be837",
)

engine = PromptEngine.from_setting(setting)


def pass_params_in_runtime():
    """演示1：运行时传入fallback策略"""
    print("=== 演示1：运行时传入fallback策略 ===")
    

    ha_strategy = {
        "type": "fallback_chain",
        "models": [
            {
                "model": "gpt-4o",
                "provider": "openai",
                "priority": 1  # 最高优先级（故意让它失败来测试fallback）

            },
            {
                "model": "claude-sonnet-4-20250514",
                "provider": "aws-litellm",
                "priority": 2  # 第二优先级（fallback目标）
            },
            {
                "model": "claude-sonnet-4-5-20250929", 
                "provider": "aws-litellm",
                "priority": 3  # 第三优先级（最后的fallback）
            }
        ]
    }

    
    try:
        for msg in engine.completion(
            "simple-demo",  # 使用promptstore中的模板
            variables={
                "instruction": "你是一个AI助手",
                "query": "请简单介绍一下fallback机制的优势",
                "chat_history": "",
                "user_question": ""
            },
            model_strategy=ha_strategy,  # 直接传入fallback策略
            stream=False,
            request_id=str(uuid.uuid4()),
            timeout=0.1
        ):
            print(msg)
            
    except Exception as e:
        import traceback
        traceback.print_exc()


def pass_params_by_template():
    try:
        for msg in engine.completion(
                "simple-demo",  # 使用promptstore中的模板
                variables={
                    "instruction": "你是一个AI助手",
                    "query": "请简单介绍一下fallback机制的优势",
                    "chat_history": "",
                    "user_question": "这是啥问题？"
                },
                stream=False,
                request_id=str(uuid.uuid4()),
                timeout=1
        ):
            print(msg)
    except Exception as e:
        print(f"❌ 所有模型都调用失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    
    # 同步演示
    pass_params_in_runtime()
    # pass_params_by_template()
    



if __name__ == "__main__":
    main()