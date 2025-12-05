"""Dummy model client that outputs 50 '你好啊!' messages."""

import time
import asyncio
from typing import AsyncGenerator, Generator, Union
from collections.abc import AsyncGenerator as AsyncGeneratorType

from .base import ModelClient, SyncModelClient, ModelConfig, RunParams
from ..message import ModelResponse, StreamingModelResponse, Choice, StreamingChoice, Message, Usage


class DummyModelClient(ModelClient):
    """Dummy async model client that outputs 50 '你好啊!' messages."""
    
    provider: str = "dummy"
    
    def __init__(self, cfg: ModelConfig, **kwargs):
        super().__init__(cfg, **kwargs)
        self.message_count = 50
        self.greeting = "你好啊!"
    
    async def _run(self, params: RunParams) -> AsyncGenerator[Union[ModelResponse, StreamingModelResponse], None]:
        """Generate 50 '你好啊!' messages in streaming or non-streaming mode."""
        
        if params.stream:
            # 流式模式：逐个输出每个"你好啊!"
            for i in range(self.message_count):
                await asyncio.sleep(0.1)  # 模拟网络延迟
                
                # 构造流式响应
                delta_message = Message(role="assistant", content=f"{self.greeting} ")
                choice = StreamingChoice(
                    index=0,
                    delta=delta_message,
                    finish_reason=None if i < self.message_count - 1 else "stop"
                )
                
                response = StreamingModelResponse(
                    id=f"dummy-stream-{int(time.time())}-{i}",
                    object="chat.completion.chunk",
                    created=int(time.time()),
                    model=self.cfg.model or "dummy-model",
                    choices=[choice],
                    usage=Usage(
                        prompt_tokens=10,
                        completion_tokens=i + 1,
                        total_tokens=10 + i + 1
                    ) if i == self.message_count - 1 else None  # 只在最后一个chunk包含usage
                )
                
                # 记录响应到trace context
                params.trace_context["responses"].append(response.model_dump())
                
                yield response
        else:
            # 非流式模式：一次性输出所有50个"你好啊!"
            await asyncio.sleep(0.5)  # 模拟处理时间
            
            full_content = " ".join([self.greeting] * self.message_count)
            message = Message(role="assistant", content=full_content)
            choice = Choice(
                index=0,
                message=message,
                finish_reason="stop"
            )
            
            response = ModelResponse(
                id=f"dummy-non-stream-{int(time.time())}",
                object="chat.completion",
                created=int(time.time()),
                model=self.cfg.model or "dummy-model",
                choices=[choice],
                usage=Usage(
                    prompt_tokens=10,
                    completion_tokens=self.message_count * 2,  # 估算token数
                    total_tokens=10 + self.message_count * 2
                )
            )
            
            # 记录响应到trace context
            params.trace_context["responses"].append(response.model_dump())
            params.trace_context["llm_response_body"] = response.model_dump()
            
            yield response


class SyncDummyModelClient(SyncModelClient):
    """Dummy sync model client that outputs 50 '你好啊!' messages."""
    
    provider: str = "dummy"
    
    def __init__(self, cfg: ModelConfig, **kwargs):
        super().__init__(cfg, **kwargs)
        self.message_count = 50
        self.greeting = "你好啊!"
    
    def _run(self, params: RunParams) -> Generator[Union[ModelResponse, StreamingModelResponse], None, None]:
        """Generate 50 '你好啊!' messages in streaming or non-streaming mode."""
        
        if params.stream:
            # 流式模式：逐个输出每个"你好啊!"
            for i in range(self.message_count):
                time.sleep(0.1)  # 模拟网络延迟
                
                # 构造流式响应
                delta_message = Message(role="assistant", content=f"{self.greeting} ")
                choice = StreamingChoice(
                    index=0,
                    delta=delta_message,
                    finish_reason=None if i < self.message_count - 1 else "stop"
                )
                
                response = StreamingModelResponse(
                    id=f"dummy-stream-{int(time.time())}-{i}",
                    object="chat.completion.chunk",
                    created=int(time.time()),
                    model=self.cfg.model or "dummy-model",
                    choices=[choice],
                    usage=Usage(
                        prompt_tokens=10,
                        completion_tokens=i + 1,
                        total_tokens=10 + i + 1
                    ) if i == self.message_count - 1 else None  # 只在最后一个chunk包含usage
                )
                
                # 记录响应到trace context
                params.trace_context["responses"].append(response.model_dump())
                
                yield response
        else:
            # 非流式模式：一次性输出所有50个"你好啊!"
            time.sleep(0.5)  # 模拟处理时间
            
            full_content = " ".join([self.greeting] * self.message_count)
            message = Message(role="assistant", content=full_content)
            choice = Choice(
                index=0,
                message=message,
                finish_reason="stop"
            )
            
            response = ModelResponse(
                id=f"dummy-non-stream-{int(time.time())}",
                object="chat.completion",
                created=int(time.time()),
                model=self.cfg.model or "dummy-model",
                choices=[choice],
                usage=Usage(
                    prompt_tokens=10,
                    completion_tokens=self.message_count * 2,  # 估算token数
                    total_tokens=10 + self.message_count * 2
                )
            )
            
            # 记录响应到trace context
            params.trace_context["responses"].append(response.model_dump())
            params.trace_context["llm_response_body"] = response.model_dump()
            
            yield response