"""OpenAI-compatible API client implementation."""

from typing import AsyncGenerator, Generator, Union, Dict, Any
import json
import httpx
import time

from ..message import Message, ModelResponse, StreamingModelResponse, Choice, StreamingChoice, Usage
from .base import ModelClient, SyncModelClient, RunParams, should_retry_error, calculate_retry_delay, handle_model_client_error, is_context_length_error
from .image_utils import convert_image_urls_to_base64, preserve_original_image_urls
from ..logger import get_logger

logger = get_logger(__name__)

def filter_empty_text_content(content):
    """Filter out empty text items from message content."""
    if isinstance(content, str):
        return content
    
    if not isinstance(content, list):
        return content
    
    filtered_content = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            # 如果是text类型且text为空字符串，跳过这个item
            if item.get("text", "").strip():
                filtered_content.append(item)
        else:
            filtered_content.append(item)
    
    return filtered_content

def is_blank_user_message_text_item(msg: Message):
    """判断message是否为空的user消息。
    
    Args:
        msg: Message对象
        
    Returns:
        bool: 如果是user类型且内容为空则返回True，否则返回False
    """
    if msg.role != "user":
        return False
    
    if msg.content is None:
        return True
    
    # 如果content是字符串类型
    if isinstance(msg.content, str):
        return not msg.content
    
    # 如果content是列表类型
    if isinstance(msg.content, list):
        if len(msg.content) == 0:
            return True
        
        # 检查列表中所有item是否都为空
        for item in msg.content:
            if isinstance(item, dict) and item.get("type") == "text":
                # 如果有任何一个text不为空，则消息不为空
                if item.get("text", ""):
                    return False
            else:
                # 如果有非text类型的item（如image_url），则消息不为空
                return False
        
        # 所有text类型的item都为空，消息为空
        return True
    
    # 其他类型的content视为非空
    return False

        
class OpenAIClient(ModelClient):
    """OpenAI-compatible API client."""

    provider = "openai"

    async def _run(self, params: RunParams) -> AsyncGenerator[Union[ModelResponse, StreamingModelResponse], None]:
        """Execute the OpenAI API call."""
        # 构建请求数据
        request_data = self._build_request_data(params)
        url = self.cfg.api_url or "https://api.openai.com/v1/chat/completions"
        headers = self._build_headers()
        self._logger.debug(request_data)

        # 设置超时时间
        timeout = params.timeout if params.timeout is not None else 600
        
        # 手动重试逻辑
        max_retries = 2
        for attempt in range(max_retries):
            try:
                logger.debug(f"Attempt {attempt + 1}/{max_retries}")

                if params.stream:
                    # 处理流式响应 - 使用 client.stream()
                    async with self._client.stream(
                        "POST",
                        url,
                        headers=headers,
                        json=request_data,
                        timeout=timeout,
                    ) as response:
                        response.raise_for_status()
                        async for message in self._aprocess_streaming_response(response):
                            yield message
                        return
                else:
                    # 处理非流式响应 - 使用普通 post
                    response = await self._client.post(
                        url=url,
                        headers=headers,
                        json=request_data,
                        timeout=timeout,
                    )
                    response.raise_for_status()
                    yield self._process_non_streaming_response(response)
                    return  # 成功则退出重试循环

            except Exception as e:
                is_last_attempt = attempt == max_retries - 1
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


    def _create_error_response(self, error_message: str, is_streaming: bool = False,
                              error_type: str = None, error_code: str = None) -> Union[
        ModelResponse, StreamingModelResponse]:
        """创建错误响应"""
        
        # 尝试解析OpenAI标准错误格式
        error_object = None
        try:
            # 如果error_message是JSON格式，尝试解析
            if error_message.strip().startswith('{'):
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
                            "type": error_type or "api_error",
                            "code": error_code or "unknown_error"
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
                        "type": error_type or "api_error",
                        "code": error_code or "unknown_error"
                    }
        except json.JSONDecodeError:
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
                    "type": error_type or "api_error",
                    "code": error_code or "unknown_error"
                }

        # 根据请求类型创建相应的错误响应
        if is_streaming:
            return StreamingModelResponse(error=error_object)
        else:
            return ModelResponse(error=error_object)

    def _build_headers(self) -> Dict[str, str]:
        """构建请求头。"""
        headers = {
            "Content-Type": "application/json",
        }

        if self.cfg.api_key:
            headers["Authorization"] = f"Bearer {self.cfg.api_key}"

        return headers

    def _build_request_data(self, params: RunParams) -> Dict[str, Any]:
        """构建OpenAI API请求数据。"""
        # 转换消息格式
        messages = []
        
        for msg in params.messages:
            if is_blank_user_message_text_item(msg):
                continue
            converted_content = convert_image_urls_to_base64(msg.content) if msg.content else msg.content
            
            # 如果转换后的内容是空列表，跳过这个消息
            if isinstance(converted_content, list) and len(converted_content) == 0:
                continue
            
            openai_msg = {
                "role": msg.role,
                "content": converted_content
            }

            # 添加工具调用字段
            if msg.tool_calls:
                openai_msg["tool_calls"] = msg.tool_calls

            # 添加工具调用ID字段（用于工具结果消息）
            if msg.tool_call_id:
                openai_msg["tool_call_id"] = msg.tool_call_id

            messages.append(openai_msg)

        for item in messages:
            if item.get("role") == "assistant" and "tool_calls" in item and item.get("content") == "":
                item["content"] = None
            # 处理 tool 角色的消息，如果 content 是 list，需要提取其中的 text 字段
            elif item.get("role") == "tool" and isinstance(item.get("content"), list):
                content_list = item["content"]
                flatt_content = ""
                for content_item in content_list:
                    if isinstance(content_item, dict) and content_item.get("type") == "text" and "text" in content_item:
                        flatt_content += content_item["text"]
                item["content"] = flatt_content
        # 基础请求数据
        request_data = {
            "model": self.cfg.get_actual_model_name(),  # 使用真实模型名称进行API调用
            "messages": messages,
            "stream": params.stream,
        }

        if params.stream:
            request_data["stream_options"] = {
                "include_usage": True,
            }

        # 添加可选参数
        if params.temperature is not None:
            request_data["temperature"] = params.temperature
        elif self.cfg.temperature is not None:
            request_data["temperature"] = self.cfg.temperature

        # 对于名称中包含claude-sonnet-4-5、claude-haiku-4-5-20251001、gpt-5、gpt-5-pro的模型，不设置top_p参数
        model_name = self.cfg.get_actual_model_name()
        if not any(model in model_name for model in ["claude-sonnet-4-5", "claude-haiku-4-5-20251001", 
                                                     "gpt-5", "gpt-5-pro"]):
            if params.top_p is not None:
                request_data["top_p"] = params.top_p
            elif self.cfg.top_p is not None:
                request_data["top_p"] = self.cfg.top_p

        if params.max_tokens is not None:
            if self.cfg.get_actual_model_name() in ["o4-mini", "gpt-5", "gpt-5-mini", "gpt-5-nano"]:
                request_data["max_completion_tokens"] = params.max_tokens
            else:
                request_data["max_tokens"] = params.max_tokens
        elif self.cfg.max_tokens is not None:
            if self.cfg.get_actual_model_name() in ["o4-mini", "gpt-5", "gpt-5-mini", "gpt-5-nano"]:
                request_data["max_completion_tokens"] = self.cfg.max_tokens
            else:
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

        # 处理工具参数
        if params.tool_params:
            self._add_tool_params(request_data, params.tool_params)

        # 添加额外参数
        request_data.update(params.extra_params)
        if self.cfg.get_actual_model_name() in ["o4-mini", "gpt-5", "gpt-5-mini", "gpt-5-nano"]:
            request_data.pop("top_p", None)
        
        # 为数据上报保留原始URL格式
        original_request_data = request_data.copy()
        params.trace_context["llm_request_body"] = original_request_data
        
        return request_data

    def _add_tool_params(self, request_data: Dict[str, Any], tool_params) -> None:
        """添加工具参数到请求数据。"""
        if not tool_params or not tool_params.tools:
            return

        # 转换工具规范为OpenAI格式
        tools = []
        for tool in tool_params.tools:
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters
                }
            }
            tools.append(openai_tool)

        request_data["tools"] = tools

        # 设置工具选择策略
        if tool_params.choice == "auto":
            request_data["tool_choice"] = "auto"
        elif tool_params.choice == "none":
            request_data["tool_choice"] = "none"
        elif tool_params.choice == "required":
            request_data["tool_choice"] = "required"
        elif isinstance(tool_params.choice, str) and tool_params.choice.startswith("function:"):
            # 指定特定函数
            function_name = tool_params.choice[9:]  # 移除 "function:" 前缀
            request_data["tool_choice"] = {
                "type": "function",
                "function": {"name": function_name}
            }
        elif tool_params.choice:
            # 如果choice是具体的工具名称
            request_data["tool_choice"] = {
                "type": "function",
                "function": {"name": tool_params.choice}
            }

    async def _aprocess_streaming_response(self, response) -> AsyncGenerator[StreamingModelResponse, None]:
        """处理流式响应。"""
        buffer = ""

        async for chunk in response.aiter_text():
            buffer += chunk

            # 处理SSE数据格式
            lines = buffer.split('\n')
            buffer = lines[-1]  # 保留可能不完整的最后一行

            for line in lines[:-1]:
                line = line.strip()
                if not line:
                    continue

                if line == "data: [DONE]":
                    return

                if line.startswith("data: "):
                    data_str = line[6:]  # 移除 "data: " 前缀
                    try:
                        data = json.loads(data_str)
                        # 提取内容
                        if "choices" in data and len(data["choices"]) > 0:
                            choice_data = data["choices"][0]
                            delta_data = choice_data.get("delta", {})
                            content = delta_data.get("content", "")
                            reasoning_content = delta_data.get("reasoning_content")
                            tool_calls = delta_data.get("tool_calls")

                            # 创建Message对象作为delta
                            delta_message = Message(
                                role=delta_data.get("role", "assistant"),
                                content=content if content else None,
                                reasoning_content=reasoning_content,
                                tool_calls=tool_calls
                            )

                            # 创建StreamingChoice对象
                            streaming_choice = StreamingChoice(
                                index=choice_data.get("index", 0),
                                delta=delta_message,
                                finish_reason=choice_data.get("finish_reason")
                            )

                            usage = None
                            if "usage" in data:
                                usage_data = data["usage"] or {}
                                usage = Usage(
                                    prompt_tokens=usage_data.get("prompt_tokens", 0),
                                    completion_tokens=usage_data.get("completion_tokens", 0),
                                    total_tokens=usage_data.get("total_tokens", 0)
                                )

                            # 创建StreamingResponse对象
                            streaming_response = StreamingModelResponse(
                                id=data.get("id", ""),
                                object=data.get("object", "chat.completion.chunk"),
                                created=data.get("created", 0),
                                model=data.get("model", self.cfg.get_aggregated_model_name()),
                                choices=[streaming_choice],
                                system_fingerprint=data.get("system_fingerprint"),
                                usage=usage
                            )

                            yield streaming_response

                    except json.JSONDecodeError:
                        # 忽略无效的JSON行
                        continue

    def _process_non_streaming_response(self, response) -> ModelResponse:
        """处理非流式响应。"""
        data = response.json()

        if "choices" in data and len(data["choices"]) > 0:
            choice_data = data["choices"][0]
            message_data = choice_data["message"]

            # 创建Message对象
            message = Message(
                role=message_data["role"],
                content=message_data.get("content"),
                reasoning_content=message_data.get("reasoning_content"),
                tool_calls=message_data.get("tool_calls")
            )

            # 创建Choice对象
            choice = Choice(
                index=choice_data.get("index", 0),
                message=message,
                finish_reason=choice_data.get("finish_reason")
            )

            # 创建Usage对象（如果存在）
            usage = None
            if "usage" in data:
                usage_data = data["usage"]
                usage = Usage(
                    prompt_tokens=usage_data.get("prompt_tokens", 0),
                    completion_tokens=usage_data.get("completion_tokens", 0),
                    total_tokens=usage_data.get("total_tokens", 0)
                )

            return ModelResponse(
                id=data.get("id", ""),
                object=data.get("object", "chat.completion"),
                created=data.get("created", 0),
                model=data.get("model", self.cfg.get_aggregated_model_name()),
                choices=[choice],
                usage=usage,
                system_fingerprint=data.get("system_fingerprint")
            )
        else:
            raise ValueError(f"Unexpected response format: {data}")


class SyncOpenAIClient(SyncModelClient):
    """Synchronous OpenAI-compatible API client."""

    provider = "openai"

    def _run(self, params: RunParams) -> Generator[Union[ModelResponse, StreamingModelResponse], None, None]:
        """Execute the OpenAI API call."""
        # 构建请求数据
        request_data = self._build_request_data(params)
        url = self.cfg.api_url or "https://api.openai.com/v1/chat/completions"
        headers = self._build_headers()
        logger.debug(f"Request data: {request_data}, headers: {headers}, url: {url}")
        # 设置超时时间
        timeout = params.timeout if params.timeout is not None else 600
        
        # 手动重试逻辑
        max_retries = 2
        for attempt in range(max_retries):
            try:
                logger.debug(f"Attempt {attempt + 1}/{max_retries}")
                
                if params.stream:
                    # 处理流式响应
                    with self._client.stream(
                        "POST",
                        url,
                        headers=headers,
                        json=request_data,
                        timeout=timeout,
                    ) as response:
                        response.raise_for_status()
                        for message in self._process_streaming_response(response):
                            yield message
                        return
                else:
                    # 处理非流式响应
                    response = self._client.post(
                        url=url,
                        headers=headers,
                        json=request_data,
                        timeout=timeout,
                    )

                    response.raise_for_status()
                    yield self._process_non_streaming_response(response)
                    return  # 成功则退出重试循环
                    
            except Exception as e:
                is_last_attempt = attempt == max_retries - 1
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

    def _create_error_response(self, error_message: str, is_streaming: bool = False,
                              error_type: str = None, error_code: str = None) -> Union[
        ModelResponse, StreamingModelResponse]:
        """创建错误响应"""
        # 检测上下文长度超限错误（使用base.py中的共用函数）
        
        error_object = None
        try:
            if error_message.strip().startswith('{'):
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
                        error_object = {
                            "message": error_message,
                            "type": error_type or "api_error",
                            "code": error_code or "unknown_error"
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
                    error_object = {
                        "message": error_message,
                        "type": error_type or "api_error",
                        "code": error_code or "unknown_error"
                    }
        except json.JSONDecodeError:
            # 检查是否为上下文长度错误
            if is_context_length_error(error_message):
                error_object = {
                    "message": error_message,
                    "type": "context_length_exceed_error",
                    "code": "context_length_exceed"
                }
            else:
                error_object = {
                    "message": error_message,
                    "type": error_type or "api_error",
                    "code": error_code or "unknown_error"
                }

        if is_streaming:
            return StreamingModelResponse(error=error_object)
        else:
            return ModelResponse(error=error_object)

    def _build_headers(self) -> Dict[str, str]:
        """构建请求头。"""
        headers = {
            "Content-Type": "application/json",
        }

        if self.cfg.api_key:
            headers["Authorization"] = f"Bearer {self.cfg.api_key}"

        return headers

    def _build_request_data(self, params: RunParams) -> Dict[str, Any]:
        """构建OpenAI API请求数据。"""
        messages = []
        
        for msg in params.messages:
            # 先过滤空text，再转换图片URL为base64
            if is_blank_user_message_text_item(msg):
                continue
            converted_content = convert_image_urls_to_base64(msg.content) if msg.content else msg.content
            
            openai_msg = {
                "role": msg.role,
                "content": converted_content
            }

            if msg.tool_calls:
                openai_msg["tool_calls"] = msg.tool_calls

            if msg.tool_call_id:
                openai_msg["tool_call_id"] = msg.tool_call_id

            messages.append(openai_msg)

        for item in messages:
            if item.get("role") == "assistant" and "tool_calls" in item and item.get("content") == "":
                item["content"] = None
            # 处理 tool 角色的消息，如果 content 是 list，需要提取其中的 text 字段
            elif item.get("role") == "tool" and isinstance(item.get("content"), list):
                content_list = item["content"]
                flatt_content = ""
                for content_item in content_list:
                    if isinstance(content_item, dict) and content_item.get("type") == "text" and "text" in content_item:
                        flatt_content += content_item["text"]
                item["content"] = flatt_content

        request_data = {
            "model": self.cfg.get_actual_model_name(),  # 使用真实模型名称进行API调用
            "messages": messages,
            "stream": params.stream,
        }

        if params.stream:
            request_data["stream_options"] = {
                "include_usage": True,
            }

        if params.temperature is not None:
            request_data["temperature"] = params.temperature
        elif self.cfg.temperature is not None:
            request_data["temperature"] = self.cfg.temperature

        # 对于名称中包含claude-sonnet-4-5、claude-haiku-4-5-20251001、gpt-5、gpt-5-pro的模型，不设置top_p参数
        model_name = self.cfg.get_actual_model_name()
        if not any(model in model_name for model in ["claude-sonnet-4-5", "claude-haiku-4-5-20251001", 
                                                     "gpt-5", "gpt-5-pro"]):
            if params.top_p is not None:
                request_data["top_p"] = params.top_p
            elif self.cfg.top_p is not None:
                request_data["top_p"] = self.cfg.top_p

        if params.max_tokens is not None:
            if self.cfg.get_actual_model_name() in ["o4-mini", "gpt-5", "gpt-5-mini", "gpt-5-nano"]:
                request_data["max_completion_tokens"] = params.max_tokens
            else:
                request_data["max_tokens"] = params.max_tokens
        elif self.cfg.max_tokens is not None:
            if self.cfg.get_actual_model_name() in ["o4-mini", "gpt-5", "gpt-5-mini", "gpt-5-nano"]:
                request_data["max_completion_tokens"] = self.cfg.max_tokens
            else:
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

        if params.tool_params:
            self._add_tool_params(request_data, params.tool_params)

        request_data.update(params.extra_params)
        if self.cfg.get_actual_model_name() in ["o4-mini", "gpt-5", "gpt-5-mini", "gpt-5-nano"]:
            request_data.pop("top_p", None)
        
        # 为数据上报保留原始URL格式
        original_request_data = request_data.copy()
        params.trace_context["llm_request_body"] = original_request_data
        return request_data

    def _add_tool_params(self, request_data: Dict[str, Any], tool_params) -> None:
        """添加工具参数到请求数据。"""
        if not tool_params or not tool_params.tools:
            return

        tools = []
        for tool in tool_params.tools:
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters
                }
            }
            tools.append(openai_tool)

        request_data["tools"] = tools

        if tool_params.choice == "auto":
            request_data["tool_choice"] = "auto"
        elif tool_params.choice == "none":
            request_data["tool_choice"] = "none"
        elif tool_params.choice == "required":
            request_data["tool_choice"] = "required"
        elif isinstance(tool_params.choice, str) and tool_params.choice.startswith("function:"):
            function_name = tool_params.choice[9:]
            request_data["tool_choice"] = {
                "type": "function",
                "function": {"name": function_name}
            }
        elif tool_params.choice:
            request_data["tool_choice"] = {
                "type": "function",
                "function": {"name": tool_params.choice}
            }

    def _process_streaming_response(self, response) -> Generator[StreamingModelResponse, None, None]:
        """处理流式响应。"""
        buffer = ""

        for chunk in response.iter_text():
            buffer += chunk

            lines = buffer.split('\n')
            buffer = lines[-1]

            for line in lines[:-1]:
                line = line.strip()
                if not line:
                    continue

                if line == "data: [DONE]":
                    return

                if line.startswith("data: "):
                    data_str = line[6:]
                    try:
                        data = json.loads(data_str)
                        if "choices" in data and len(data["choices"]) > 0:
                            choice_data = data["choices"][0]
                            delta_data = choice_data.get("delta", {})
                            content = delta_data.get("content", "")
                            reasoning_content = delta_data.get("reasoning_content")
                            tool_calls = delta_data.get("tool_calls")

                            delta_message = Message(
                                role=delta_data.get("role", "assistant"),
                                content=content if content else None,
                                reasoning_content=reasoning_content,
                                tool_calls=tool_calls
                            )

                            streaming_choice = StreamingChoice(
                                index=choice_data.get("index", 0),
                                delta=delta_message,
                                finish_reason=choice_data.get("finish_reason")
                            )

                            usage = None
                            if "usage" in data:
                                usage_data = data["usage"] or {}
                                usage = Usage(
                                    prompt_tokens=usage_data.get("prompt_tokens", 0),
                                    completion_tokens=usage_data.get("completion_tokens", 0),
                                    total_tokens=usage_data.get("total_tokens", 0)
                                )

                            streaming_response = StreamingModelResponse(
                                id=data.get("id", ""),
                                object=data.get("object", "chat.completion.chunk"),
                                created=data.get("created", 0),
                                model=data.get("model", self.cfg.get_aggregated_model_name()),
                                choices=[streaming_choice],
                                system_fingerprint=data.get("system_fingerprint"),
                                usage=usage
                            )

                            yield streaming_response

                    except json.JSONDecodeError:
                        continue

    def _process_non_streaming_response(self, response) -> ModelResponse:
        """处理非流式响应。"""
        data = response.json()

        if "choices" in data and len(data["choices"]) > 0:
            choice_data = data["choices"][0]
            message_data = choice_data["message"]

            message = Message(
                role=message_data["role"],
                content=message_data.get("content"),
                reasoning_content=message_data.get("reasoning_content"),
                tool_calls=message_data.get("tool_calls")
            )

            choice = Choice(
                index=choice_data.get("index", 0),
                message=message,
                finish_reason=choice_data.get("finish_reason")
            )

            usage = None
            if "usage" in data:
                usage_data = data["usage"]
                usage = Usage(
                    prompt_tokens=usage_data.get("prompt_tokens", 0),
                    completion_tokens=usage_data.get("completion_tokens", 0),
                    total_tokens=usage_data.get("total_tokens", 0)
                )

            return ModelResponse(
                id=data.get("id", ""),
                object=data.get("object", "chat.completion"),
                created=data.get("created", 0),
                model=data.get("model", self.cfg.get_aggregated_model_name()),
                choices=[choice],
                usage=usage,
                system_fingerprint=data.get("system_fingerprint")
            )
        else:
            raise ValueError(f"Unexpected response format: {data}")
