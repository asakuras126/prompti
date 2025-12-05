"""Minimal example demonstrating PromptI with synchronous completion."""

from __future__ import annotations

import logging
import uuid
import time
from prompti.engine import PromptEngine, Setting, Message

from prompti.model_client.base import ModelConfig, RunParams, ToolParams, ToolSpec

# setting = Setting(
#     registry_url="http://10.224.55.241/api/v1",
#     registry_api_key="7e5d106c-e701-4587-a95a-b7c7c02ee619",
#     otel_config={
#         "enabled": True,
#         "service_name": "prompti_local",
#         "endpoint": "http://10.224.55.33:8317",
#         "export_interval_ms": 5000,
#         "queue_size": 1000
#     }
# )

setting = Setting(
    registry_url="http://localhost:8080/api/v1",
    registry_api_key="ccffa346-2e96-4596-b0a3-a63dee6be837",
    otel_config={
        "enabled": True,
        "service_name": "prompti_local",
        "endpoint": "http://10.224.55.33:8317",
        "export_interval_ms": 5000,
        "queue_size": 1000
    }
)

engine = PromptEngine.from_setting(setting)

model_cfg = ModelConfig(
    provider="gemini_http1",
    model="gemini-3-pro-preview",
    temperature=1,
    top_p=0.0001,
    max_tokens=64000,
    ext={
        "thinking_level": "HIGH"
    }
)


def runtime(session_id: str) -> None:
    """Demonstrate multi-turn chat with sync completion."""
    context = {
        "app_id": "app-80367g8pkem9",
        "session_id": "conv-80367g8pkem8",
        "user_id": "user-78evyz7ulw5c",
        "region": "cn",
        "trace_id": "31d49d15-e03c-4e24-ac33-bea5d031f653",
        "subscription_plan": "free",
        "app_type": "Web",
        "template_level": "NONE",
        "turns": "head",
        "query_type": "coding",
        "mmu": False
      }
    print(context)
    rules =  [
    {
      "name": "white_user",
      "selector": {
        "type": "weighted_round_robin"
      },
      "candidates": [
        {
          "name": "coding",
          "weight": 1
        }
      ],
      "conditions": {
        "condition_list": [
          {
            "type": "value",
            "field": "user_id",
            "values": [
              "user_001"
            ]
          },
          {
            "type": "value",
            "field": "query_type",
            "values": [
              "coding"
            ]
          },
          {
            "type": "bool",
            "field": "mmu",
            "expected": False
          },
          {
            "type": "value",
            "field": "turns",
            "values": [
              "head"
            ]
          }
        ]
      }
    },
    {
      "name": "haiku_for_fronted_game",
      "selector": {
        "type": "weighted_round_robin"
      },
      "candidates": [
        {
          "name": "claude-haiku",
          "weight": 1
        }
      ],
      "conditions": {
        "condition_list": [
          {
            "type": "value",
            "field": "query_type",
            "values": [
              "coding"
            ]
          },
          {
            "type": "value",
            "field": "app_type",
            "values": [
              "Frontend Game"
            ]
          },
          {
            "type": "hash",
            "hash_key": "session_id",
            "percentage": 80
          }
        ]
      }
    },
    {
      "name": "default_coding",
      "selector": {
        "type": "weighted_round_robin"
      },
      "candidates": [
        {
          "name": "coding",
          "weight": 1
        }
      ],
      "conditions": {
        "condition_list": [
          {
            "type": "value",
            "field": "query_type",
            "values": [
              "coding"
            ]
          },
          {
            "type": "bool",
            "field": "mmu",
            "expected": False
          },
          {
            "type": "value",
            "field": "turns",
            "values": [
              "head"
            ]
          },
          {
            "type": "hash",
            "hash_key": "session_id",
            "percentage": 100
          }
        ]
      }
    },
    {
      "name": "default",
      "selector": {
        "type": "weighted_round_robin"
      },
      "candidates": [
        {
          "name": "default",
          "weight": 1
        }
      ],
      "conditions": []
    }
  ]

    try:
        ans = ""
        for msg in engine.completion(
            "coding_agent",
            version="^1#release",
            variables={"instruction": "你好", "query": "1+1=？", "chat_history": 123},
            variant_router={
                "context": context,
                "rules": rules
            },
            stream=False,
            messages=[
                {
                    "role": "user",
                    "content": "你好",
                }
            ],
            timeout=1,
            conversation_id=session_id,
            model_cfg=model_cfg
        ):
            print(f"\n\n\n user_get msg: {msg} \n\n\n")
            ans += msg.get_text_content() or ""
        print(repr(ans))
    finally:
        pass

if __name__ == "__main__":
    for i in range(5000):
        print("=" * 120)
        session_id = str(uuid.uuid4())
        for i in range(5):
            runtime(session_id=session_id)
            time.sleep(5)
        print("=" * 120)



