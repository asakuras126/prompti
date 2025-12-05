"""LiteLLM client implementation using the `litellm` package."""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncGenerator, Generator
from typing import Any, Dict, List, Union

import httpx

from ..message import Message, ModelResponse, StreamingModelResponse, Usage, Choice, StreamingChoice
from .base import ModelClient, SyncModelClient, ModelConfig, RunParams, ToolChoice, ToolParams, ToolSpec, should_retry_error, calculate_retry_delay, handle_model_client_error, is_context_length_error
from .image_utils import convert_image_urls_to_base64
from ..logger import get_logger

import json

logger = get_logger(__name__)

try:
    from json_repair import repair_json
except ImportError:
    repair_json = None


import litellm  # noqa: F401
litellm.drop_params = True
    
class LiteLLMClient(ModelClient):
    """Client for the LiteLLM API."""

    provider = "litellm"

    def __init__(self, cfg: ModelConfig, client: httpx.AsyncClient | None = None, is_debug: bool = False) -> None:
        """Instantiate the client with configuration and optional HTTP client."""

        super().__init__(cfg, client, is_debug=is_debug)
        self.api_key = cfg.api_key
        self.api_url = cfg.api_url
        self.base_url = cfg.api_url  # 为了兼容性，将api_url赋值给base_url

    def _fix_messages_json_errors(self, messages: List[Dict]) -> List[Dict]:
        """使用json_repair修复消息中的JSON错误，特别是tool calls中的转义字符问题"""
        try:
            for message in messages:
                # 清理 cache_control 标记
                if isinstance(message.get("content"), list):
                    for content_item in message["content"]:
                        if isinstance(content_item, dict) and "cache_control" in content_item:
                            del content_item["cache_control"]
                
                # 修复 tool calls 中的转义字符问题
                if message.get("tool_calls"):
                    for tool_call in message["tool_calls"]:
                        if tool_call.get("function", {}).get("arguments"):
                            args = tool_call["function"]["arguments"]
                            if isinstance(args, str):
                                try:
                                    # 验证是否为有效JSON
                                    json.loads(args)
                                except json.JSONDecodeError as e:
                                    self._logger.warning(f"Invalid JSON in tool call arguments: {e}")
                                    try:
                                        # 使用json-repair自动修复
                                        if repair_json is not None:
                                            fixed_args = repair_json(args)
                                            json.loads(fixed_args)  # 验证修复结果
                                            tool_call["function"]["arguments"] = fixed_args
                                            self._logger.info("Fixed tool call arguments using json-repair")
                                        else:
                                            self._logger.warning("json_repair not available, keeping original arguments")
                                    except Exception:
                                        self._logger.warning("Could not fix tool call arguments, keeping original")
                                        
        except Exception as e:
            self._logger.warning(f"Failed to process messages: {e}")
        
        return messages

    def _build_request_data(self, params: RunParams) -> Dict[str, Any]:
        """构建LiteLLM API请求数据。"""
        # 转换消息格式并处理图片URL
        messages = []
        for m in params.messages:
            openai_msg = m.to_openai()
            # 如果消息包含图片URL，转换为base64
            if 'content' in openai_msg and isinstance(openai_msg['content'], list):
                openai_msg['content'] = convert_image_urls_to_base64(openai_msg['content'])
            messages.append(openai_msg)

        # 针对DeepSeek模型进行特殊处理：将数组格式的content转换为字符串
        if "deepseek" in self.cfg.get_actual_model_name().lower():
            for msg in messages:
                if isinstance(msg.get('content'), list):
                    # 将数组格式转换为字符串格式
                    text_parts = []
                    for part in msg['content']:
                        if part.get('type') == 'text':
                            text_parts.append(part.get('text', ''))
                    msg['content'] = ' '.join(text_parts) if text_parts else ""

        # 基础请求数据
        request_data = {
            "model": self.cfg.get_actual_model_name(),  # 使用真实模型名称进行API调用
            "messages": messages,
            "stream": params.stream,
        }
        if params.stream:
            request_data.update({"stream_options": {"include_usage": True}})

        # 添加API配置参数
        if self.api_key:
            request_data["api_key"] = self.api_key
        if self.api_url:
            request_data["api_base"] = self.api_url  # litellm 使用 api_base 参数
        
        # 添加bedrock model_id标签，用于特定提供商（如AWS Bedrock应用推理配置文件）
        bedrock_model_tag = self.cfg.ext.get("bedrock_model_tag")
        if bedrock_model_tag:
            request_data["model_id"] = bedrock_model_tag

        # 添加可选参数
        if params.temperature is not None:
            request_data["temperature"] = params.temperature
        elif self.cfg.temperature is not None:
            request_data["temperature"] = self.cfg.temperature

        # 对于名称中包含claude-sonnet-4-5-20250929、gpt-5、gpt-5-pro的模型，不设置top_p参数
        model_name = self.cfg.get_actual_model_name()
        if not any(model in model_name for model in ["claude-sonnet-4-5-20250929", "claude-haiku-4-5-20251001",
                                                      "gpt-5", "gpt-5-pro"]):
            if params.top_p is not None:
                request_data["top_p"] = params.top_p
            elif self.cfg.top_p is not None:
                request_data["top_p"] = self.cfg.top_p

        if params.max_tokens is not None:
            request_data["max_tokens"] = params.max_tokens
        elif self.cfg.max_tokens is not None:
            request_data["max_tokens"] = self.cfg.max_tokens

        if params.stop:
            request_data["stop"] = params.stop

        if params.n is not None:
            request_data["n"] = params.n

        if params.seed is not None:
            request_data["seed"] = params.seed

        if params.logit_bias:
            request_data["logit_bias"] = params.logit_bias

        if params.response_format:
            request_data["response_format"] = {"type": params.response_format}

        if params.user_id:
            request_data["user"] = params.user_id

        # 添加超时参数 - LiteLLM使用request_timeout参数
        if params.timeout is not None:
            request_data["timeout"] = params.timeout
            request_data["request_timeout"] = params.timeout  # LiteLLM的标准超时参数
            
        # 添加重试参数 - 使用LiteLLM的内置重试
        request_data["num_retries"] = 0

        # 处理工具参数
        if params.tool_params:
            if isinstance(params.tool_params, ToolParams):
                tools = [
                    {"type": "function", "function": t.model_dump()} if isinstance(t, ToolSpec) else t
                    for t in params.tool_params.tools
                ]
                # 只有当tools不为空时才添加到请求数据中
                if tools:
                    request_data["tools"] = tools
                    choice = params.tool_params.choice
                    if isinstance(choice, ToolChoice):
                        if choice is not ToolChoice.AUTO:
                            request_data["tool_choice"] = choice.value
                    elif choice is not None:
                        request_data["tool_choice"] = choice
            else:
                tools = [
                    {"type": "function", "function": t.model_dump()} if isinstance(t, ToolSpec) else t
                    for t in params.tool_params
                ]
                # 只有当tools不为空时才添加到请求数据中
                if tools:
                    request_data["tools"] = tools

        # Support Claude thinking parameter via ModelConfig.ext
        if self.cfg.ext and 'thinking' in self.cfg.ext:
            thinking_config = self.cfg.ext['thinking']
            if isinstance(thinking_config, dict):
                # Validate thinking configuration
                thinking_type = thinking_config.get('type')
                if thinking_type in ('enabled', 'disabled'):
                    request_data['thinking'] = thinking_config.copy()
                    logger.debug(f"Added thinking parameter: {request_data['thinking']}")
                else:
                    logger.warning(f"Invalid thinking type: {thinking_type}. Must be 'enabled' or 'disabled'")

        # 添加额外参数
        request_data.update(params.extra_params)
        params.trace_context["llm_request_body"] = request_data
        return request_data

    async def _aprocess_streaming_response(self, response) -> AsyncGenerator[StreamingModelResponse, None]:
        """处理流式响应。"""
        async for chunk in response:
            if hasattr(chunk, "choices") and chunk.choices:
                choice = chunk.choices[0]
                delta = choice.delta if hasattr(choice, "delta") else {}

                # 处理流式工具调用 - 转换为字典格式
                tool_calls = None
                if hasattr(delta, "tool_calls") and delta.tool_calls:
                    tool_calls = []
                    for tool_call in delta.tool_calls:
                        # 构建工具调用字典
                        tool_call_dict = {"type": "function"}

                        # 处理ID
                        if hasattr(tool_call, "id"):
                            tool_call_dict["id"] = tool_call.id

                        # 处理函数信息
                        if hasattr(tool_call, "function"):
                            function_dict = {}
                            if hasattr(tool_call.function, "name"):
                                function_dict["name"] = tool_call.function.name
                            if hasattr(tool_call.function, "arguments"):
                                function_dict["arguments"] = tool_call.function.arguments
                            tool_call_dict["function"] = function_dict

                        tool_calls.append(tool_call_dict)

                # 创建StreamingChoice对象
                streaming_choice = StreamingChoice(
                    index=choice.index if hasattr(choice, "index") else 0,
                    delta=Message(
                        role="assistant",
                        content=delta.content if hasattr(delta, "content") else None,
                        tool_calls=tool_calls
                    ),
                    finish_reason=choice.finish_reason if hasattr(choice, "finish_reason") else None
                )

                # 创建StreamingResponse对象
                streaming_response = StreamingModelResponse(
                    id=chunk.id if hasattr(chunk, "id") else str(uuid.uuid4()),
                    created=chunk.created if hasattr(chunk, "created") else int(time.time()),
                    model=chunk.model if hasattr(chunk, "model") else self.cfg.get_aggregated_model_name(),
                    choices=[streaming_choice],
                    usage=Usage(
                        prompt_tokens=chunk.usage.prompt_tokens if hasattr(chunk, "usage") and chunk.usage else 0,
                        completion_tokens=chunk.usage.completion_tokens if hasattr(chunk,
                                                                                   "usage") and chunk.usage else 0,
                        total_tokens=chunk.usage.total_tokens if hasattr(chunk, "usage") and chunk.usage else 0,
                        cache_creation_input_tokens=getattr(chunk.usage, "cache_creation_input_tokens", None) if hasattr(chunk, "usage") and chunk.usage else None,
                        cache_read_input_tokens=getattr(chunk.usage, "cache_read_input_tokens", None) if hasattr(chunk, "usage") and chunk.usage else None,
                        cached_tokens=getattr(chunk.usage, "cached_tokens", None) if hasattr(chunk, "usage") and chunk.usage else None
                    ) if hasattr(chunk, "usage") and chunk.usage else None
                )

                yield streaming_response

    def _process_non_streaming_response(self, response) -> ModelResponse:
        """处理非流式响应。"""
        if hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            message = choice.message if hasattr(choice, "message") else {}

            # 处理工具调用 - LiteLLM返回的是LiteLLM特有的对象，需要转换为字典
            tool_calls = None
            if hasattr(message, "tool_calls") and message.tool_calls:
                tool_calls = []
                for tool_call in message.tool_calls:
                    # 转换工具调用为字典格式
                    # 获取arguments
                    arguments = "{}"
                    if hasattr(tool_call, "function") and hasattr(tool_call.function, "arguments"):
                        arguments = tool_call.function.arguments

                    tool_call_dict = {
                        "id": tool_call.id if hasattr(tool_call, "id") else "",
                        "type": tool_call.type if hasattr(tool_call, "type") else "function",
                        "function": {
                            "name": tool_call.function.name if hasattr(tool_call, "function") and hasattr(
                                tool_call.function, "name") else "",
                            "arguments": arguments
                        }
                    }
                    tool_calls.append(tool_call_dict)

            # Extract reasoning/thinking text from Claude responses
            # Note: litellm already extracts thinking_blocks into reasoning_content,
            # so we only need to use reasoning_content to avoid duplication
            reasoning_content = None
            if hasattr(message, "reasoning_content") and message.reasoning_content:
                reasoning_content = message.reasoning_content

            # Build message with reasoning_content
            msg = Message(
                role="assistant",
                content=message.content if hasattr(message, "content") else None,
                tool_calls=tool_calls,
                reasoning_content=reasoning_content
            )

            # 创建Choice对象
            model_choice = Choice(
                index=choice.index if hasattr(choice, "index") else 0,
                message=msg,
                finish_reason=choice.finish_reason if hasattr(choice, "finish_reason") else None
            )

            # 创建Usage对象
            usage = None
            if hasattr(response, "usage") and response.usage:
                usage = Usage(
                    prompt_tokens=response.usage.prompt_tokens if hasattr(response.usage, "prompt_tokens") else 0,
                    completion_tokens=response.usage.completion_tokens if hasattr(response.usage,
                                                                                  "completion_tokens") else 0,
                    total_tokens=response.usage.total_tokens if hasattr(response.usage, "total_tokens") else 0,
                    cache_creation_input_tokens=getattr(response.usage, "cache_creation_input_tokens", None),
                    cache_read_input_tokens=getattr(response.usage, "cache_read_input_tokens", None),
                    cached_tokens=getattr(response.usage, "cached_tokens", None)
                )

            # 创建ModelResponse对象
            return ModelResponse(
                id=response.id if hasattr(response, "id") else str(uuid.uuid4()),
                created=response.created if hasattr(response, "created") else int(time.time()),
                model=response.model if hasattr(response, "model") else self.cfg.get_aggregated_model_name(),
                choices=[model_choice],
                usage=usage
            )
        else:
            # 返回空响应
            return ModelResponse(
                id=str(uuid.uuid4()),
                model=self.cfg.get_aggregated_model_name(),
                choices=[Choice(
                    index=0,
                    message=Message(role="assistant", content=""),
                    finish_reason="stop"
                )]
            )

    def _create_error_response(self, error_message: str, is_streaming: bool = False,
                              error_type: str = None, error_code: str = None) -> Union[
        ModelResponse, StreamingModelResponse]:
        """创建错误响应。"""
        # 尝试解析错误格式
        error_object = None
        try:
            # 如果error_message是JSON格式，尝试解析
            if error_message.strip().startswith('{'):
                import json
                parsed_error = json.loads(error_message)
                if "error" in parsed_error:
                    error_object = parsed_error["error"]
                    # 检查是否为上下文长度错误
                    if is_context_length_error(error_object.get("message", "")):
                        error_object["type"] = "context_length_exceed_error"
                        error_object["code"] = "context_length_exceed"
                else:
                    # 检查是否为上下文长度错误
                    if is_context_length_error(error_message):
                        error_object = {
                            "message": error_message,
                            "type": "context_length_exceed_error",
                            "code": "context_length_exceed"
                        }
                    else:
                        # 包装成标准格式
                        error_object = {
                            "message": error_message,
                            "type": error_type or "litellm_error",
                            "code": error_code or "request_error"
                        }
            else:
                # 检查是否为上下文长度错误
                if is_context_length_error(error_message):
                    error_object = {
                        "message": error_message,
                        "type": "context_length_exceed_error",
                        "code": "context_length_exceed"
                    }
                else:
                    # 包装成标准格式
                    error_object = {
                        "message": error_message,
                        "type": error_type or "litellm_error",
                        "code": error_code or "request_error"
                    }
        except Exception:
            # 检查是否为上下文长度错误
            if is_context_length_error(error_message):
                error_object = {
                    "message": error_message,
                    "type": "context_length_exceed_error",
                    "code": "context_length_exceed"
                }
            else:
                # 如果不是JSON，包装成标准格式
                error_object = {
                    "message": error_message,
                    "type": error_type or "litellm_error",
                    "code": error_code or "request_error"
                }

        if is_streaming:
            return StreamingModelResponse(error=error_object)
        else:
            return ModelResponse(error=error_object)

    def _should_use_sonnet_optimization(self, params: RunParams, error_str: str) -> bool:
        """判断是否应该使用Sonnet优化策略。"""
        # 检查模型是否为sonnet-4或sonnet-4.5
        model_name = self.cfg.get_actual_model_name().lower()
        is_sonnet_4_or_45 = ("sonnet-4" in model_name or "sonnet4" in model_name) and not params.stream
        
        # 检查错误信息是否包含超时相关信息
        timeout_error_patterns = [
            "provider fallback required: unexpected error: litellm.timeout",
            "connection timed out after none seconds",
            "litellm.timeout",
            "timeout"
        ]
        
        has_timeout_error = any(pattern in error_str.lower() for pattern in timeout_error_patterns)
        
        return is_sonnet_4_or_45 and has_timeout_error

    async def _run(self, params: RunParams) -> AsyncGenerator[Union[ModelResponse, StreamingModelResponse], None]:
        """Execute the LiteLLM API call."""
        try:
            # 确保已安装litellm
            import litellm
            import asyncio
        except ImportError as e:
            raise ImportError(
                "litellm is required for LiteLLMClient. Install with: pip install 'prompti[litellm]'"
            ) from e

        # 构建请求数据
        request_data = self._build_request_data(params)
        timeout_info = f" with timeout={params.timeout}s" if params.timeout else " (no timeout set)"
        self._logger.debug(f"litellm request data{timeout_info}: {request_data}")
        
        # 手动重试逻辑 - 在LiteLLM内置重试的基础上增加额外重试
        max_retries = 2
        sonnet_optimization_attempted = False
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempt {attempt + 1}/{max_retries}")
                
                if params.stream:
                    # 处理流式响应
                    response = await litellm.acompletion(
                        **request_data
                    )
                    async for message in self._aprocess_streaming_response(response):
                        yield message
                    return
                else:
                    # 处理非流式响应
                    response = await litellm.acompletion(
                        **request_data
                    )
                    yield self._process_non_streaming_response(response)
                    return  # 成功则退出重试循环

            except Exception as e:
                is_last_attempt = attempt == max_retries - 1
                error_str = str(e)
                
                # 记录详细的错误信息，包括超时设置
                error_context = f"timeout={params.timeout}s" if params.timeout else "no timeout set"
                self._logger.error(f"LiteLLM request failed ({error_context}): {error_str}")

                # Sonnet 4/4.5 超时优化策略
                if (not sonnet_optimization_attempted and 
                    self._should_use_sonnet_optimization(params, error_str)):
                    
                    logger.info(f"Applying Sonnet timeout optimization: switching to stream=True with fine-grained-tool-streaming header")
                    sonnet_optimization_attempted = True
                    
                    # 创建新的请求参数，强制使用stream=True
                    optimized_params = RunParams(
                        messages=params.messages,
                        stream=True,  # 强制开启流式
                        temperature=params.temperature,
                        top_p=params.top_p,
                        max_tokens=params.max_tokens,
                        stop=params.stop,
                        n=params.n,
                        seed=params.seed,
                        logit_bias=params.logit_bias,
                        response_format=params.response_format,
                        user_id=params.user_id,
                        timeout=params.timeout,
                        tool_params=params.tool_params,
                        extra_params=params.extra_params.copy(),
                        trace_context=params.trace_context
                    )
                    
                    # 添加fine-grained-tool-streaming header
                    if "extra_headers" not in optimized_params.extra_params:
                        optimized_params.extra_params["extra_headers"] = {}
                    optimized_params.extra_params["extra_headers"]["anthropic-beta"] = "fine-grained-tool-streaming-2025-05-14"
                    
                    try:
                        # 重新构建请求数据
                        optimized_request_data = self._build_request_data(optimized_params)
                        
                        # 应用JSON错误修复到消息中
                        if "messages" in optimized_request_data:
                            optimized_request_data["messages"] = self._fix_messages_json_errors(optimized_request_data["messages"])
                        
                        logger.info("Executing optimized Sonnet request with streaming enabled")
                        
                        # 执行优化后的请求
                        response = await litellm.acompletion(**optimized_request_data)
                        
                        # 合并流式响应为单个响应
                        merged_response = None
                        async for streaming_chunk in self._aprocess_streaming_response(response):
                            if merged_response is None:
                                # 初始化合并响应
                                merged_response = ModelResponse(
                                    id=streaming_chunk.id,
                                    created=streaming_chunk.created,
                                    model=streaming_chunk.model,
                                    choices=[Choice(
                                        index=0,
                                        message=Message(role="assistant", content="", tool_calls=[]),
                                        finish_reason=None
                                    )],
                                    usage=None
                                )
                            
                            # 合并内容
                            if streaming_chunk.choices and streaming_chunk.choices[0].delta:
                                delta = streaming_chunk.choices[0].delta
                                if delta.content:
                                    merged_response.choices[0].message.content += delta.content
                                
                                # 合并tool_calls
                                if delta.tool_calls:
                                    if not merged_response.choices[0].message.tool_calls:
                                        merged_response.choices[0].message.tool_calls = []
                                    
                                    for tool_call in delta.tool_calls:
                                        # 查找已存在的tool_call或创建新的
                                        existing_call = None
                                        if tool_call.get("id"):
                                            for existing in merged_response.choices[0].message.tool_calls:
                                                if existing.get("id") == tool_call["id"]:
                                                    existing_call = existing
                                                    break
                                        
                                        if existing_call is None:
                                            # 创建新的tool_call
                                            new_call = {
                                                "id": tool_call.get("id", ""),
                                                "type": tool_call.get("type", "function"),
                                                "function": {
                                                    "name": tool_call.get("function", {}).get("name", ""),
                                                    "arguments": tool_call.get("function", {}).get("arguments", "")
                                                }
                                            }
                                            merged_response.choices[0].message.tool_calls.append(new_call)
                                        else:
                                            # 更新已存在的tool_call
                                            if tool_call.get("function", {}).get("name"):
                                                existing_call["function"]["name"] = tool_call["function"]["name"]
                                            if tool_call.get("function", {}).get("arguments"):
                                                existing_call["function"]["arguments"] += tool_call["function"]["arguments"]
                                
                                # 更新finish_reason
                                if streaming_chunk.choices[0].finish_reason:
                                    merged_response.choices[0].finish_reason = streaming_chunk.choices[0].finish_reason
                            
                            # 更新usage信息
                            if streaming_chunk.usage:
                                merged_response.usage = streaming_chunk.usage
                        
                        if merged_response:
                            yield merged_response
                            return
                        
                    except Exception as optimization_error:
                        logger.error(f"Sonnet optimization failed: {str(optimization_error)}")
                        # 如果优化失败，继续原有的错误处理流程

                
                should_retry = should_retry_error(e)
                
                if should_retry and not is_last_attempt:
                    # 计算等待时间（指数退避，基于错误类型调整）
                    wait_time = calculate_retry_delay(attempt, error=e)
                    logger.warning(f"Request failed (attempt {attempt + 1}), retrying in {wait_time}s: {str(e)}")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    # 最后一次尝试失败或不应该重试的错误
                    logger.error(f"Request failed permanently: {str(e)}")
                    # 处理错误并返回错误响应
                    yield handle_model_client_error(e, params.stream, self._create_error_response)
                    return

    async def aclose(self) -> None:
        """Close the underlying HTTP client and clean up LiteLLM resources."""
        # Close our httpx client
        await super().aclose()

        # Try to clean up LiteLLM's internal aiohttp sessions
        try:
            import aiohttp
            import gc

            # Force garbage collection of aiohttp connectors
            gc.collect()

            # Try to close any open aiohttp sessions
            for obj in gc.get_objects():
                if isinstance(obj, aiohttp.ClientSession):
                    if not obj.closed:
                        try:
                            await obj.close()
                        except Exception:
                            pass

        except Exception:
            # Ignore errors in cleanup
            pass


class SyncLiteLLMClient(SyncModelClient):
    """Synchronous client for the LiteLLM API."""

    provider = "litellm"

    def __init__(self, cfg: ModelConfig, client: httpx.Client | None = None, is_debug: bool = False) -> None:
        """Instantiate the client with configuration and optional HTTP client."""
        try:
            import litellm  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "litellm is required for SyncLiteLLMClient. Install with: pip install 'prompti[litellm]'"
            ) from e

        super().__init__(cfg, client, is_debug=is_debug)
        self.api_key = cfg.api_key
        self.api_url = cfg.api_url
        self.base_url = cfg.api_url

    def _fix_messages_json_errors(self, messages: List[Dict]) -> List[Dict]:
        """使用json_repair修复消息中的JSON错误，特别是tool calls中的转义字符问题"""
        try:
            for message in messages:
                # 修复 tool calls 中的转义字符问题
                if message.get("tool_calls"):
                    for tool_call in message["tool_calls"]:
                        if tool_call.get("function", {}).get("arguments"):
                            args = tool_call["function"]["arguments"]
                            if isinstance(args, str):
                                try:
                                    # 验证是否为有效JSON
                                    json.loads(args)
                                except json.JSONDecodeError as e:
                                    self._logger.warning(f"Invalid JSON in tool call arguments: {e}")
                                    try:
                                        # 使用json-repair自动修复
                                        if repair_json is not None:
                                            fixed_args = repair_json(args)
                                            json.loads(fixed_args)  # 验证修复结果
                                            tool_call["function"]["arguments"] = fixed_args
                                            self._logger.info("Fixed tool call arguments using json-repair")
                                        else:
                                            self._logger.warning("json_repair not available, keeping original arguments")
                                    except Exception:
                                        self._logger.warning("Could not fix tool call arguments, keeping original")
                                        
        except Exception as e:
            self._logger.warning(f"Failed to process messages: {e}")
        
        return messages

    def _build_request_data(self, params: RunParams) -> Dict[str, Any]:
        """构建LiteLLM API请求数据。"""
        # 转换消息格式并处理图片URL
        messages = []
        for m in params.messages:
            openai_msg = m.to_openai()
            # 如果消息包含图片URL，转换为base64
            if 'content' in openai_msg and isinstance(openai_msg['content'], list):
                openai_msg['content'] = convert_image_urls_to_base64(openai_msg['content'])
            messages.append(openai_msg)

        # 针对DeepSeek模型进行特殊处理：将数组格式的content转换为字符串
        if "deepseek" in self.cfg.get_actual_model_name().lower():
            for msg in messages:
                if isinstance(msg.get('content'), list):
                    # 将数组格式转换为字符串格式
                    text_parts = []
                    for part in msg['content']:
                        if part.get('type') == 'text':
                            text_parts.append(part.get('text', ''))
                    msg['content'] = ' '.join(text_parts) if text_parts else ""

        request_data = {
            "model": self.cfg.get_actual_model_name(),  # 使用真实模型名称进行API调用
            "messages": messages,
            "stream": params.stream,
        }
        if params.stream:
            request_data.update({"stream_options": {"include_usage": True}})

        # 添加API配置参数
        if self.api_key:
            request_data["api_key"] = self.api_key
        if self.api_url:
            request_data["api_base"] = self.api_url
        
        # 添加bedrock model_id标签，用于特定提供商（如AWS Bedrock应用推理配置文件）
        bedrock_model_tag = self.cfg.ext.get("bedrock_model_tag")
        if bedrock_model_tag:
            request_data["model_id"] = bedrock_model_tag

        if params.temperature is not None:
            request_data["temperature"] = params.temperature
        elif self.cfg.temperature is not None:
            request_data["temperature"] = self.cfg.temperature

        # 对于名称中包含claude-sonnet-4-5-20250929、gpt-5、gpt-5-pro的模型，不设置top_p参数
        model_name = self.cfg.get_actual_model_name()
        if not any(model in model_name for model in ["claude-sonnet-4-5-20250929", "claude-haiku-4-5-20251001", 
                                                     "gpt-5", "gpt-5-pro"]):
            if params.top_p is not None:
                request_data["top_p"] = params.top_p
            elif self.cfg.top_p is not None:
                request_data["top_p"] = self.cfg.top_p

        if params.max_tokens is not None:
            request_data["max_tokens"] = params.max_tokens
        elif self.cfg.max_tokens is not None:
            request_data["max_tokens"] = self.cfg.max_tokens

        if params.stop:
            request_data["stop"] = params.stop

        if params.n is not None:
            request_data["n"] = params.n

        if params.seed is not None:
            request_data["seed"] = params.seed

        if params.logit_bias:
            request_data["logit_bias"] = params.logit_bias

        if params.response_format:
            request_data["response_format"] = {"type": params.response_format}

        if params.user_id:
            request_data["user"] = params.user_id

        # 添加超时参数 - LiteLLM使用request_timeout参数
        if params.timeout is not None:
            request_data["timeout"] = params.timeout
            request_data["request_timeout"] = params.timeout  # LiteLLM的标准超时参数

        if params.tool_params:
            if isinstance(params.tool_params, ToolParams):
                tools = [
                    {"type": "function", "function": t.model_dump()} if isinstance(t, ToolSpec) else t
                    for t in params.tool_params.tools
                ]
                # 只有当tools不为空时才添加到请求数据中
                if tools:
                    request_data["tools"] = tools
                    choice = params.tool_params.choice
                    if isinstance(choice, ToolChoice):
                        if choice is not ToolChoice.AUTO:
                            request_data["tool_choice"] = choice.value
                    elif choice is not None:
                        request_data["tool_choice"] = choice
            else:
                tools = [
                    {"type": "function", "function": t.model_dump()} if isinstance(t, ToolSpec) else t
                    for t in params.tool_params
                ]
                # 只有当tools不为空时才添加到请求数据中
                if tools:
                    request_data["tools"] = tools

        # Support Claude thinking parameter via ModelConfig.ext
        if self.cfg.ext and 'thinking' in self.cfg.ext:
            thinking_config = self.cfg.ext['thinking']
            if isinstance(thinking_config, dict):
                # Validate thinking configuration
                thinking_type = thinking_config.get('type')
                if thinking_type in ('enabled', 'disabled'):
                    request_data['thinking'] = thinking_config.copy()
                    logger.debug(f"Added thinking parameter: {request_data['thinking']}")
                else:
                    logger.warning(f"Invalid thinking type: {thinking_type}. Must be 'enabled' or 'disabled'")

        request_data.update(params.extra_params)

        # 添加重试参数 - 使用LiteLLM的内置重试
        request_data["num_retries"] = 0

        params.trace_context["llm_request_body"] = request_data
        return request_data

    def _process_streaming_response(self, response) -> Generator[StreamingModelResponse, None, None]:
        """处理流式响应。"""
        for chunk in response:
            if hasattr(chunk, "choices") and chunk.choices:
                choice = chunk.choices[0]
                delta = choice.delta if hasattr(choice, "delta") else {}

                tool_calls = None
                if hasattr(delta, "tool_calls") and delta.tool_calls:
                    tool_calls = []
                    for tool_call in delta.tool_calls:
                        tool_call_dict = {"type": "function"}

                        if hasattr(tool_call, "id"):
                            tool_call_dict["id"] = tool_call.id

                        if hasattr(tool_call, "function"):
                            function_dict = {}
                            if hasattr(tool_call.function, "name"):
                                function_dict["name"] = tool_call.function.name
                            if hasattr(tool_call.function, "arguments"):
                                # 解码unicode内容
                                function_dict["arguments"] = tool_call.function.arguments
                            tool_call_dict["function"] = function_dict

                        tool_calls.append(tool_call_dict)

                streaming_choice = StreamingChoice(
                    index=choice.index if hasattr(choice, "index") else 0,
                    delta=Message(
                        role="assistant",
                        content=delta.content if hasattr(delta, "content") else None,
                        tool_calls=tool_calls
                    ),
                    finish_reason=choice.finish_reason if hasattr(choice, "finish_reason") else None
                )

                streaming_response = StreamingModelResponse(
                    id=chunk.id if hasattr(chunk, "id") else str(uuid.uuid4()),
                    created=chunk.created if hasattr(chunk, "created") else int(time.time()),
                    model=chunk.model if hasattr(chunk, "model") else self.cfg.get_aggregated_model_name(),
                    choices=[streaming_choice],
                    usage=Usage(
                        prompt_tokens=chunk.usage.prompt_tokens if hasattr(chunk, "usage") and chunk.usage else 0,
                        completion_tokens=chunk.usage.completion_tokens if hasattr(chunk,
                                                                                   "usage") and chunk.usage else 0,
                        total_tokens=chunk.usage.total_tokens if hasattr(chunk, "usage") and chunk.usage else 0,
                        cache_creation_input_tokens=getattr(chunk.usage, "cache_creation_input_tokens", None) if hasattr(chunk, "usage") and chunk.usage else None,
                        cache_read_input_tokens=getattr(chunk.usage, "cache_read_input_tokens", None) if hasattr(chunk, "usage") and chunk.usage else None,
                        cached_tokens=getattr(chunk.usage, "cached_tokens", None) if hasattr(chunk, "usage") and chunk.usage else None
                    ) if hasattr(chunk, "usage") and chunk.usage else None
                )

                yield streaming_response

    def _process_non_streaming_response(self, response) -> ModelResponse:
        """处理非流式响应。"""
        self._logger.debug(f"litellm no stream response: {response}")
        if hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            message = choice.message if hasattr(choice, "message") else {}

            tool_calls = None
            if hasattr(message, "tool_calls") and message.tool_calls:
                tool_calls = []
                for tool_call in message.tool_calls:
                    # 获取arguments
                    arguments = "{}"
                    if hasattr(tool_call, "function") and hasattr(tool_call.function, "arguments"):
                        arguments = tool_call.function.arguments

                    tool_call_dict = {
                        "id": tool_call.id if hasattr(tool_call, "id") else "",
                        "type": tool_call.type if hasattr(tool_call, "type") else "function",
                        "function": {
                            "name": tool_call.function.name if hasattr(tool_call, "function") and hasattr(
                                tool_call.function, "name") else "",
                            "arguments": arguments
                        }
                    }
                    tool_calls.append(tool_call_dict)

            # Extract reasoning/thinking text from Claude responses
            # Note: litellm already extracts thinking_blocks into reasoning_content,
            # so we only need to use reasoning_content to avoid duplication
            reasoning_content = None
            if hasattr(message, "reasoning_content") and message.reasoning_content:
                reasoning_content = message.reasoning_content

            # Build message with reasoning_content
            msg = Message(
                role="assistant",
                content=message.content if hasattr(message, "content") else None,
                tool_calls=tool_calls,
                reasoning_content=reasoning_content
            )

            model_choice = Choice(
                index=choice.index if hasattr(choice, "index") else 0,
                message=msg,
                finish_reason=choice.finish_reason if hasattr(choice, "finish_reason") else None
            )

            usage = None
            if hasattr(response, "usage") and response.usage:
                usage = Usage(
                    prompt_tokens=response.usage.prompt_tokens if hasattr(response.usage, "prompt_tokens") else 0,
                    completion_tokens=response.usage.completion_tokens if hasattr(response.usage,
                                                                                  "completion_tokens") else 0,
                    total_tokens=response.usage.total_tokens if hasattr(response.usage, "total_tokens") else 0,
                    cache_creation_input_tokens=getattr(response.usage, "cache_creation_input_tokens", None),
                    cache_read_input_tokens=getattr(response.usage, "cache_read_input_tokens", None),
                    cached_tokens=getattr(response.usage, "cached_tokens", None)
                )

            return ModelResponse(
                id=response.id if hasattr(response, "id") else str(uuid.uuid4()),
                created=response.created if hasattr(response, "created") else int(time.time()),
                model=response.model if hasattr(response, "model") else self.cfg.get_aggregated_model_name(),
                choices=[model_choice],
                usage=usage
            )
        else:
            return ModelResponse(
                id=str(uuid.uuid4()),
                model=self.cfg.get_aggregated_model_name(),
                choices=[Choice(
                    index=0,
                    message=Message(role="assistant", content=""),
                    finish_reason="stop"
                )]
            )

    def _create_error_response(self, error_message: str, is_streaming: bool = False,
                              error_type: str = None, error_code: str = None) -> Union[
        ModelResponse, StreamingModelResponse]:
        """创建错误响应。"""
        # 尝试解析错误格式
        error_object = None
        try:
            # 如果error_message是JSON格式，尝试解析
            if error_message.strip().startswith('{'):
                import json
                parsed_error = json.loads(error_message)
                if "error" in parsed_error:
                    error_object = parsed_error["error"]
                    # 检查是否为上下文长度错误
                    if is_context_length_error(error_object.get("message", "")):
                        error_object["type"] = "context_length_exceed_error"
                        error_object["code"] = "context_length_exceed"
                else:
                    # 检查是否为上下文长度错误
                    if is_context_length_error(error_message):
                        error_object = {
                            "message": error_message,
                            "type": "context_length_exceed_error",
                            "code": "context_length_exceed"
                        }
                    else:
                        # 包装成标准格式
                        error_object = {
                            "message": error_message,
                            "type": error_type or "litellm_error",
                            "code": error_code or "request_error"
                        }
            else:
                # 检查是否为上下文长度错误
                if is_context_length_error(error_message):
                    error_object = {
                        "message": error_message,
                        "type": "context_length_exceed_error",
                        "code": "context_length_exceed"
                    }
                else:
                    # 包装成标准格式
                    error_object = {
                        "message": error_message,
                        "type": error_type or "litellm_error",
                        "code": error_code or "request_error"
                    }
        except Exception:
            # 检查是否为上下文长度错误
            if is_context_length_error(error_message):
                error_object = {
                    "message": error_message,
                    "type": "context_length_exceed_error",
                    "code": "context_length_exceed"
                }
            else:
                # 如果不是JSON，包装成标准格式
                error_object = {
                    "message": error_message,
                    "type": error_type or "litellm_error",
                    "code": error_code or "request_error"
                }

        if is_streaming:
            return StreamingModelResponse(error=error_object)
        else:
            return ModelResponse(error=error_object)

    def _should_use_sonnet_optimization(self, params: RunParams, error_str: str) -> bool:
        """判断是否应该使用Sonnet优化策略。"""
        # 检查模型是否为sonnet-4或sonnet-4.5
        model_name = self.cfg.get_actual_model_name().lower()
        is_sonnet_4_or_45 = ("sonnet-4" in model_name or "sonnet4" in model_name) and not params.stream
        
        # 检查错误信息是否包含超时相关信息
        timeout_error_patterns = [
            "provider fallback required: unexpected error: litellm.timeout",
            "connection timed out after none seconds",
            "litellm.timeout",
            "timeout"
        ]
        
        has_timeout_error = any(pattern in error_str.lower() for pattern in timeout_error_patterns)
        
        return is_sonnet_4_or_45 and has_timeout_error

    def _run(self, params: RunParams) -> Generator[Union[ModelResponse, StreamingModelResponse], None, None]:
        """Execute the LiteLLM API call."""
        try:
            import litellm
        except ImportError as e:
            raise ImportError(
                "litellm is required for SyncLiteLLMClient. Install with: pip install 'prompti[litellm]'"
            ) from e

        request_data = self._build_request_data(params)
        timeout_info = f" with timeout={params.timeout}s" if params.timeout else " (no timeout set)"

        self._logger.debug(f"litellm request data{timeout_info}: {request_data}")
        
        # 手动重试逻辑 - 在LiteLLM内置重试的基础上增加额外重试
        max_retries = 2
        sonnet_optimization_attempted = False
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempt {attempt + 1}/{max_retries}")
                
                if params.stream:
                    # 处理流式响应
                    response = litellm.completion(
                        **request_data
                    )
                    for message in self._process_streaming_response(response):
                        yield message
                    return
                else:
                    # 处理非流式响应
                    response = litellm.completion(
                        **request_data
                    )
                    yield self._process_non_streaming_response(response)
                    return  # 成功则退出重试循环
                    
            except Exception as e:
                is_last_attempt = attempt == max_retries - 1
                error_str = str(e)
                
                # 记录详细的错误信息，包括超时设置
                error_context = f"timeout={params.timeout}s" if params.timeout else "no timeout set"
                self._logger.error(f"LiteLLM request failed ({error_context}): {error_str}")

                # Sonnet 4/4.5 超时优化策略
                if (not sonnet_optimization_attempted and 
                    self._should_use_sonnet_optimization(params, error_str)):
                    
                    logger.info(f"Applying Sonnet timeout optimization: switching to stream=True with fine-grained-tool-streaming header")
                    sonnet_optimization_attempted = True
                    
                    # 创建新的请求参数，强制使用stream=True
                    optimized_params = RunParams(
                        messages=params.messages,
                        stream=True,  # 强制开启流式
                        temperature=params.temperature,
                        top_p=params.top_p,
                        max_tokens=params.max_tokens,
                        stop=params.stop,
                        n=params.n,
                        seed=params.seed,
                        logit_bias=params.logit_bias,
                        response_format=params.response_format,
                        user_id=params.user_id,
                        timeout=params.timeout,
                        tool_params=params.tool_params,
                        extra_params=params.extra_params.copy(),
                        trace_context=params.trace_context
                    )
                    
                    # 添加fine-grained-tool-streaming header
                    if "extra_headers" not in optimized_params.extra_params:
                        optimized_params.extra_params["extra_headers"] = {}
                    optimized_params.extra_params["extra_headers"]["anthropic-beta"] = "fine-grained-tool-streaming-2025-05-14"
                    
                    try:
                        # 重新构建请求数据
                        optimized_request_data = self._build_request_data(optimized_params)
                        
                        # 应用JSON错误修复到消息中
                        if "messages" in optimized_request_data:
                            optimized_request_data["messages"] = self._fix_messages_json_errors(optimized_request_data["messages"])
                        
                        logger.info("Executing optimized Sonnet request with streaming enabled")
                        
                        # 执行优化后的请求
                        response = litellm.completion(**optimized_request_data)
                        
                        # 合并流式响应为单个响应
                        merged_response = None
                        for streaming_chunk in self._process_streaming_response(response):
                            self._logger.debug(streaming_chunk)
                            if merged_response is None:
                                # 初始化合并响应
                                merged_response = ModelResponse(
                                    id=streaming_chunk.id,
                                    created=streaming_chunk.created,
                                    model=streaming_chunk.model,
                                    choices=[Choice(
                                        index=0,
                                        message=Message(role="assistant", content="", tool_calls=[]),
                                        finish_reason=None
                                    )],
                                    usage=None
                                )
                            
                            # 合并内容
                            if streaming_chunk.choices and streaming_chunk.choices[0].delta:
                                delta = streaming_chunk.choices[0].delta
                                if delta.content:
                                    merged_response.choices[0].message.content += delta.content
                                
                                # 合并tool_calls
                                if delta.tool_calls:
                                    if not merged_response.choices[0].message.tool_calls:
                                        merged_response.choices[0].message.tool_calls = []
                                    
                                    for tool_call in delta.tool_calls:
                                        # 查找已存在的tool_call或创建新的
                                        existing_call = None
                                        if tool_call.get("id"):
                                            for existing in merged_response.choices[0].message.tool_calls:
                                                if existing.get("id") == tool_call["id"]:
                                                    existing_call = existing
                                                    break
                                        
                                        if existing_call is None:
                                            # 创建新的tool_call
                                            new_call = {
                                                "id": tool_call.get("id", ""),
                                                "type": tool_call.get("type", "function"),
                                                "function": {
                                                    "name": tool_call.get("function", {}).get("name", ""),
                                                    "arguments": tool_call.get("function", {}).get("arguments", "")
                                                }
                                            }
                                            merged_response.choices[0].message.tool_calls.append(new_call)
                                        else:
                                            # 更新已存在的tool_call
                                            if tool_call.get("function", {}).get("name"):
                                                existing_call["function"]["name"] = tool_call["function"]["name"]
                                            if tool_call.get("function", {}).get("arguments"):
                                                existing_call["function"]["arguments"] += tool_call["function"]["arguments"]
                                
                                # 更新finish_reason
                                if streaming_chunk.choices[0].finish_reason:
                                    merged_response.choices[0].finish_reason = streaming_chunk.choices[0].finish_reason
                            
                            # 更新usage信息
                            if streaming_chunk.usage:
                                merged_response.usage = streaming_chunk.usage
                        
                        if merged_response:
                            yield merged_response
                            return
                        
                    except Exception as optimization_error:
                        logger.error(f"Sonnet optimization failed: {str(optimization_error)}")
                        # 如果优化失败，继续原有的错误处理流程

                should_retry = should_retry_error(e)
                
                if should_retry and not is_last_attempt:
                    # 计算等待时间（指数退避，基于错误类型调整）
                    wait_time = calculate_retry_delay(attempt, error=e)
                    logger.warning(f"Request failed (attempt {attempt + 1}), retrying in {wait_time}s: {str(e)}")
                    time.sleep(wait_time)
                    continue
                else:
                    # 最后一次尝试失败或不应该重试的错误
                    logger.error(f"Request failed permanently: {str(e)}")
                    # 处理错误并返回错误响应
                    yield handle_model_client_error(e, params.stream, self._create_error_response)
                    return
