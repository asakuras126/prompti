"""Minimal example demonstrating PromptI with LiteLLM."""

from __future__ import annotations

import asyncio

from prompti.engine import PromptEngine, Setting
from prompti.model_client import ModelConfig, create_client
import logging

logging.basicConfig(level=logging.INFO)


async def get_template():
    """
    Get the template from the registry.
    """
    setting = Setting(
        registry_url="http://localhost:8080/api/v1",
        registry_api_key="ccffa346-2e96-4596-b0a3-a63dee6be837",
    )
    engine = PromptEngine.from_setting(setting)
    template = await engine.aload(
        template_name="simple-demo",
    )
    format_template = template.format(variables={
        "instruction": "你是图像分析大师",
        "query": "这张图片是什么？",
        "image_url": ["https://agentos-promptstore.bj.bcebos.com/files/test/images/default/2989be85-9bfb-4e18-a339-48466320bf0f.jpg?authorization=bce-auth-v1%2Fagentos%2F2025-07-22T07%3A35%3A56Z%2F604800%2F%2F5761014c2df87427a1ab41a1d054820ecbb2f5cd1516e998dd2430a75549764a&response-content-disposition=inline&response-content-type=image%2Fjpeg","https://agentos-promptstore.bj.bcebos.com/files/test/images/default/2989be85-9bfb-4e18-a339-48466320bf0f.jpg?authorization=bce-auth-v1%2Fagentos%2F2025-07-22T07%3A35%3A56Z%2F604800%2F%2F5761014c2df87427a1ab41a1d054820ecbb2f5cd1516e998dd2430a75549764a&response-content-disposition=inline&response-content-type=image%2Fjpeg"]
    },
        variant="multimodal")
    print(template)
    print("format_template: ", format_template)


if __name__ == "__main__":
    asyncio.run(get_template())
