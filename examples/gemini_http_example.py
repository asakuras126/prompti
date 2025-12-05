"""Minimal example demonstrating PromptI with synchronous completion."""

from __future__ import annotations
import time
import uuid
from prompti.engine import PromptEngine, Setting, Message

from prompti.model_client.base import ModelConfig, RunParams, ToolParams, ToolSpec


# setting = Setting(
#     registry_url="http://promptstore-prod.miaoda-bj-online.baidu-int.com/api/v1",
#     registry_api_key="cb31207c-b422-49b2-8420-5e35701b9bc1",
# )

# setting = Setting(
#     registry_url="http://localhost:8080/api/v1",
#     registry_api_key="ccffa346-2e96-4596-b0a3-a63dee6be837",
# )
# setting = Setting(
#     registry_api_key="5b8416a4-f21d-4423-830d-3f3af0effe3f",
#     registry_url="https://agentos.miaoda.io/api/v1"
# )
setting = Setting(
    registry_url="http://promptstore-qa.miaoda-bj-offline.baidu-int.com/api/v1",
    registry_api_key="2493e6db-e51f-48bc-89a1-b33c20543076",
)
engine = PromptEngine.from_setting(setting)

model_cfg = ModelConfig(
    provider="gemini_http",
    model="gemini-3-pro-preview",
    temperature=1,
    top_p=0.0001,
    max_tokens=64000,
    ext={
        "thinking_level": "LOW"
    }
)

def multi_chat() -> None:
    """Demonstrate multi-turn chat with sync completion."""

    try:
        ans = ""
        for msg in engine.completion(
            "coding_agent",
            version="^1#release",
            variables={},
            variant="coding2",
            stream=False,
            messages=[
                {
                    "role": "user",
                    "content": "你好",
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
    multi_chat()
    time.sleep(3)


