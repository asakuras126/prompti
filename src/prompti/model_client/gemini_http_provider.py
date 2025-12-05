"""Gemini HTTP Provider using direct requests to Google AI Platform API.

This provider uses raw HTTP requests to call Gemini 2.5 Pro API directly.
Supports:
- Streaming and non-streaming responses
- Multimodal input (text + images)
- Tool calling (function calling)
"""

from __future__ import annotations

import base64
import json
import time
import uuid
import asyncio
from typing import Any, Dict, Union, List
from collections.abc import AsyncGenerator, Generator

from .base import (
    ModelClient,
    SyncModelClient,
    ModelConfig,
    RunParams,
    ToolParams,
    ToolSpec,
    ToolChoice,
    should_retry_error,
    calculate_retry_delay,
    handle_model_client_error,
)
from ..message import Message, ModelResponse, StreamingModelResponse, Usage, Choice, StreamingChoice
from ..logger import get_logger

logger = get_logger(__name__)


class GeminiHttpProvider(ModelClient):
    """Gemini HTTP provider for direct API calls."""

    provider = "gemini_http"

    def __init__(self, cfg: ModelConfig, client=None, is_debug: bool = False) -> None:
        super().__init__(cfg, client, is_debug=is_debug)
        self.api_key = cfg.api_key
        self.model_name = cfg.model
        self.base_url = cfg.api_url or "https://aiplatform.googleapis.com"
        self.project_id = cfg.ext.get("project_id", "cls-connectnow-gemini")
        self.location = cfg.ext.get("location", "global")
        self.single_tool_call: bool = cfg.ext.get("single_tool_call", False)

    async def _run(
        self, params: RunParams
    ) -> AsyncGenerator[Union[ModelResponse, StreamingModelResponse], None]:
        max_retries = 2

        for attempt in range(max_retries):
            try:
                logger.debug(f"Attempt {attempt + 1}/{max_retries}")

                request_data = self._build_request_data(params)
                url = self._build_url(params.stream)
                headers = self._build_headers()

                timeout = params.timeout if params.timeout is not None else 600
                logger.debug(
                    f"[gemini_http] request_data: {request_data}, url: {url},"
                    f" headers: {headers}, timeout: {timeout}"
                )
                if params.stream:
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
                    response = await self._client.post(
                        url=url,
                        headers=headers,
                        json=request_data,
                        timeout=timeout,
                    )
                    response.raise_for_status()
                    yield self._process_non_streaming_response(response)
                    return

            except Exception as e:
                is_last_attempt = attempt == max_retries - 1
                should_retry = should_retry_error(e)

                if should_retry and not is_last_attempt:
                    wait_time = calculate_retry_delay(attempt, error=e)
                    logger.warning(
                        f"Request failed (attempt {attempt + 1}), retrying in {wait_time}s: {str(e)}"
                    )
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Request failed permanently: {str(e)}")
                    yield handle_model_client_error(
                        e, params.stream, self._create_error_response
                    )
                    return

    def _build_url(self, is_stream: bool) -> str:
        if is_stream:
            endpoint = "streamGenerateContent"
        else:
            endpoint = "generateContent"

        return (
            f"{self.base_url}/v1/projects/{self.project_id}/"
            f"locations/{self.location}/publishers/google/models/{self.model_name}:{endpoint}"
            f"?key={self.api_key}"
        )

    def _build_headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
        }

    def _build_request_data(self, params: RunParams) -> Dict[str, Any]:
        request_data = {
            "contents": self._convert_messages_to_gemini_format(params.messages),
            "generationConfig": self._build_generation_config(params),
        }

        # Add systemInstruction if there's a system message
        system_instruction = self._extract_system_instruction(params.messages)
        if system_instruction:
            request_data["systemInstruction"] = system_instruction

        if params.tool_params:
            tools = self._convert_tools_to_gemini_format(params.tool_params)
            if tools:
                request_data["tools"] = tools

        params.trace_context["llm_request_body"] = request_data
        return request_data

    def _extract_system_instruction(self, messages: List[Message]) -> Dict[str, Any] | None:
        """Extract system instruction from messages."""
        for msg in messages:
            if msg.role == "system" and msg.content:
                content_text = msg.content
                if isinstance(msg.content, list):
                    content_text = ""
                    for item in msg.content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            content_text += item.get("text", "")

                if content_text:
                    return {
                        "role": "system",
                        "parts": [{"text": content_text}],
                    }
        return None

    def _convert_messages_to_gemini_format(
        self, messages: List[Message]
    ) -> List[Dict[str, Any]]:
        gemini_contents: List[Dict[str, Any]] = []

        i = 0
        while i < len(messages):
            msg = messages[i]

            if msg.role == "system":
                i += 1
                continue

            role = "user" if msg.role in ["user", "tool"] else "model"
            parts: List[Dict[str, Any]] = []

            # 文本 / 图片
            if msg.content and not msg.tool_call_id:
                if isinstance(msg.content, str):
                    parts.append({"text": msg.content})
                elif isinstance(msg.content, list):
                    for item in msg.content:
                        if isinstance(item, dict):
                            if item.get("type") == "text":
                                parts.append({"text": item.get("text", "")})
                            elif item.get("type") == "image_url":
                                image_data = self._extract_image_data(
                                    item.get("image_url", {}).get("url", "")
                                )
                                if image_data:
                                    parts.append(image_data)

            # functionCall：从 OpenAI-style tool_calls 映射到 Gemini parts
            if msg.tool_calls:
                for tool_call in msg.tool_calls:
                    if tool_call.get("type") == "function":
                        function_call = tool_call.get("function", {})

                        # 使用原始的 thought_signature，如果没有则使用默认值
                        thought_sig = tool_call.get("thought_signature", "skip_thought_signature_validator")

                        part: Dict[str, Any] = {
                            "functionCall": {
                                "name": function_call.get("name", ""),
                                "args": json.loads(
                                    function_call.get("arguments", "{}")
                                ),
                            },
                            "thought_signature": thought_sig,
                        }
                        parts.append(part)

                        # 如果开启 single_tool_call，只保留第一个 functionCall
                        if self.single_tool_call:
                            break

            # functionResponse：tool 调用结果
            # 合并连续的 tool 消息到同一个 content
            if msg.tool_call_id and msg.content:
                # 收集当前和后续连续的 tool responses
                j = i
                while j < len(messages) and messages[j].role == "tool" and messages[j].tool_call_id:
                    tool_msg = messages[j]
                    content_str = tool_msg.content
                    if isinstance(tool_msg.content, list):
                        content_str = ""
                        for item in tool_msg.content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                content_str += item.get("text", "")

                    parts.append(
                        {
                            "functionResponse": {
                                "name": tool_msg.tool_call_id,
                                "response": {"result": content_str},
                            }
                        }
                    )
                    j += 1

                # 跳过已处理的 tool messages
                i = j - 1

            if parts:
                gemini_contents.append({"role": role, "parts": parts})

            i += 1

        return gemini_contents

    def _extract_image_data(self, image_url: str) -> Dict[str, Any] | None:
        if not image_url:
            return None

        if image_url.startswith("data:image"):
            mime_type = "image/jpeg"
            if "data:image/png" in image_url:
                mime_type = "image/png"
            elif "data:image/jpeg" in image_url or "data:image/jpg" in image_url:
                mime_type = "image/jpeg"

            base64_data = image_url.split(",", 1)[1] if "," in image_url else ""
            return {"inlineData": {"mimeType": mime_type, "data": base64_data}}
        else:
            return None

    def _convert_tools_to_gemini_format(
        self, tool_params: ToolParams | List[ToolSpec] | List[dict]
    ) -> List[Dict[str, Any]]:
        tools_list: List[Any] = []

        if isinstance(tool_params, ToolParams):
            tools_list = tool_params.tools
        elif isinstance(tool_params, list):
            tools_list = tool_params
        else:
            return []

        if not tools_list:
            return []

        function_declarations = []
        for tool in tools_list:
            if isinstance(tool, ToolSpec):
                function_declarations.append(
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    }
                )
            elif isinstance(tool, dict):
                if "function" in tool:
                    func = tool["function"]
                    function_declarations.append(
                        {
                            "name": func.get("name", ""),
                            "description": func.get("description", ""),
                            "parameters": func.get("parameters", {}),
                        }
                    )
                else:
                    function_declarations.append(
                        {
                            "name": tool.get("name", ""),
                            "description": tool.get("description", ""),
                            "parameters": tool.get("parameters", {}),
                        }
                    )

        return [{"functionDeclarations": function_declarations}]

    def _build_generation_config(self, params: RunParams) -> Dict[str, Any]:
        config: Dict[str, Any] = {}

        temperature = params.temperature
        if temperature is None:
            temperature = self.cfg.temperature
        if temperature is not None:
            config["temperature"] = temperature

        # 关闭topP
        # top_p = params.top_p
        # if top_p is None:
        #     top_p = self.cfg.top_p
        # if top_p is not None:
    #     config["topP"] = top_p

        max_tokens = params.max_tokens
        if max_tokens is None:
            max_tokens = self.cfg.max_tokens
        if max_tokens is not None:
            config["maxOutputTokens"] = max_tokens

        # Add thinking_level support
        thinking_level = params.extra_params.get("thinking_level") if params.extra_params else None
        if thinking_level is None:
            thinking_level = self.cfg.ext.get("thinking_level", "LOW")
        if thinking_level is not None:
            config["thinkingConfig"] = {
                "thinkingLevel": thinking_level
            }

        return config

    async def _aprocess_streaming_response(
        self, response
    ) -> AsyncGenerator[StreamingModelResponse, None]:
        buffer = ""

        async for chunk in response.aiter_text():
            buffer += chunk

            lines = buffer.split("\n")
            buffer = lines[-1]

            for line in lines[:-1]:
                line = line.strip()
                if not line:
                    continue

                if line.startswith("["):
                    try:
                        data = json.loads(line)
                        for item in data:
                            streaming_resp = self._parse_gemini_chunk(item)
                            if streaming_resp:
                                yield streaming_resp
                    except json.JSONDecodeError:
                        continue

    def _parse_gemini_chunk(self, chunk_data: Dict[str, Any]) -> StreamingModelResponse | None:
        if "candidates" not in chunk_data:
            return None

        candidates = chunk_data.get("candidates", [])
        if not candidates:
            return None

        candidate = candidates[0]
        content = candidate.get("content", {})
        parts = content.get("parts", [])

        text_content = ""
        tool_calls: List[Dict[str, Any]] = []

        for part in parts:
            if "text" in part:
                text_content += part["text"]
            elif "functionCall" in part:
                # 若开启 single_tool_call，只取第一个 functionCall
                if self.single_tool_call and tool_calls:
                    continue

                func_call = part["functionCall"]
                thought_sig = part.get("thought_signature") or part.get("thoughtSignature")

                tool_call_obj: Dict[str, Any] = {
                    "id": f"call_{uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function": {
                        "name": func_call.get("name", ""),
                        "arguments": json.dumps(func_call.get("args", {})),
                    },
                }
                if thought_sig:
                    tool_call_obj["thought_signature"] = thought_sig

                tool_calls.append(tool_call_obj)

        delta_message = Message(
            role="assistant",
            content=text_content if text_content else None,
            tool_calls=tool_calls if tool_calls else None,
        )

        streaming_choice = StreamingChoice(
            index=0,
            delta=delta_message,
            finish_reason=candidate.get("finishReason"),
        )

        usage = None
        if "usageMetadata" in chunk_data:
            usage_data = chunk_data["usageMetadata"]
            usage = Usage(
                prompt_tokens=usage_data.get("promptTokenCount", 0),
                completion_tokens=usage_data.get("candidatesTokenCount", 0),
                total_tokens=usage_data.get("totalTokenCount", 0),
            )

        return StreamingModelResponse(
            id=str(uuid.uuid4()),
            created=int(time.time()),
            model=self.cfg.get_aggregated_model_name(),
            choices=[streaming_choice],
            usage=usage,
        )

    def _process_non_streaming_response(self, response) -> ModelResponse:
        data = response.json()

        if "candidates" not in data or not data["candidates"]:
            return ModelResponse(
                id=str(uuid.uuid4()),
                model=self.cfg.get_aggregated_model_name(),
                choices=[
                    Choice(
                        index=0,
                        message=Message(role="assistant", content=""),
                        finish_reason="stop",
                    )
                ],
            )

        candidate = data["candidates"][0]
        content = candidate.get("content", {})
        parts = content.get("parts", [])

        text_content = ""
        tool_calls: List[Dict[str, Any]] = []

        for part in parts:
            if "text" in part:
                text_content += part["text"]
            elif "functionCall" in part:
                if self.single_tool_call and tool_calls:
                    continue

                func_call = part["functionCall"]
                thought_sig = part.get("thought_signature") or part.get("thoughtSignature")

                tool_call_obj: Dict[str, Any] = {
                    "id": f"call_{uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function": {
                        "name": func_call.get("name", ""),
                        "arguments": json.dumps(func_call.get("args", {})),
                    },
                }
                if thought_sig:
                    tool_call_obj["thought_signature"] = thought_sig

                tool_calls.append(tool_call_obj)

        message = Message(
            role="assistant",
            content=text_content if text_content else None,
            tool_calls=tool_calls if tool_calls else None,
        )

        choice = Choice(
            index=0,
            message=message,
            finish_reason=candidate.get("finishReason"),
        )

        usage = None
        if "usageMetadata" in data:
            usage_data = data["usageMetadata"]
            usage = Usage(
                prompt_tokens=usage_data.get("promptTokenCount", 0),
                completion_tokens=usage_data.get("candidatesTokenCount", 0),
                total_tokens=usage_data.get("totalTokenCount", 0),
            )

        return ModelResponse(
            id=str(uuid.uuid4()),
            created=int(time.time()),
            model=self.cfg.get_aggregated_model_name(),
            choices=[choice],
            usage=usage,
        )

    def _create_error_response(
        self,
        error_message: str,
        is_streaming: bool = False,
        error_type: str = None,
        error_code: str = None,
    ) -> Union[ModelResponse, StreamingModelResponse]:
        error_object = {
            "message": error_message,
            "type": error_type or "api_error",
            "code": error_code or "gemini_error",
        }
        if is_streaming:
            return StreamingModelResponse(error=error_object)
        else:
            return ModelResponse(error=error_object)


class SyncGeminiHttpProvider(SyncModelClient):
    """Synchronous Gemini HTTP provider for direct API calls."""

    provider = "gemini_http"

    def __init__(self, cfg: ModelConfig, client=None, is_debug: bool = False) -> None:
        super().__init__(cfg, client, is_debug=is_debug)
        self.api_key = cfg.api_key
        self.model_name = cfg.model
        self.base_url = cfg.api_url or "https://aiplatform.googleapis.com"
        self.project_id = cfg.ext.get("project_id", "cls-connectnow-gemini")
        self.location = cfg.ext.get("location", "global")
        self.single_tool_call: bool = cfg.ext.get("single_tool_call", True)

    def _run(
        self, params: RunParams
    ) -> Generator[Union[ModelResponse, StreamingModelResponse], None, None]:
        max_retries = 2

        for attempt in range(max_retries):
            try:
                logger.debug(f"Attempt {attempt + 1}/{max_retries}")

                request_data = self._build_request_data(params)
                url = self._build_url(params.stream)
                headers = self._build_headers()

                timeout = params.timeout if params.timeout is not None else 600
                logger.debug(
                    f"[gemini_http] request_data: {request_data}, url: {url},"
                    f" headers: {headers}, timeout: {timeout}"
                )
                if params.stream:
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
                    response = self._client.post(
                        url=url,
                        headers=headers,
                        json=request_data,
                        timeout=timeout,
                    )
                    response.raise_for_status()
                    yield self._process_non_streaming_response(response)
                    return

            except Exception as e:
                is_last_attempt = attempt == max_retries - 1
                should_retry = should_retry_error(e)

                if should_retry and not is_last_attempt:
                    wait_time = calculate_retry_delay(attempt, error=e)
                    logger.warning(
                        f"Request failed (attempt {attempt + 1}), retrying in {wait_time}s: {str(e)}"
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Request failed permanently: {str(e)}")
                    yield handle_model_client_error(
                        e, params.stream, self._create_error_response
                    )
                    return

    def _build_url(self, is_stream: bool) -> str:
        if is_stream:
            endpoint = "streamGenerateContent"
        else:
            endpoint = "generateContent"

        return (
            f"{self.base_url}/v1/projects/{self.project_id}/"
            f"locations/{self.location}/publishers/google/models/{self.model_name}:{endpoint}"
            f"?key={self.api_key}"
        )

    def _build_headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
        }

    def _build_request_data(self, params: RunParams) -> Dict[str, Any]:
        request_data = {
            "contents": self._convert_messages_to_gemini_format(params.messages),
            "generationConfig": self._build_generation_config(params),
        }

        # Add systemInstruction if there's a system message
        system_instruction = self._extract_system_instruction(params.messages)
        if system_instruction:
            request_data["systemInstruction"] = system_instruction

        if params.tool_params:
            tools = self._convert_tools_to_gemini_format(params.tool_params)
            if tools:
                request_data["tools"] = tools

        params.trace_context["llm_request_body"] = request_data
        return request_data

    def _extract_system_instruction(self, messages: List[Message]) -> Dict[str, Any] | None:
        """Extract system instruction from messages."""
        for msg in messages:
            if msg.role == "system" and msg.content:
                content_text = msg.content
                if isinstance(msg.content, list):
                    content_text = ""
                    for item in msg.content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            content_text += item.get("text", "")

                if content_text:
                    return {
                        "role": "system",
                        "parts": [{"text": content_text}],
                    }
        return None

    def _convert_messages_to_gemini_format(
        self, messages: List[Message]
    ) -> List[Dict[str, Any]]:
        gemini_contents: List[Dict[str, Any]] = []

        i = 0
        while i < len(messages):
            msg = messages[i]

            if msg.role == "system":
                i += 1
                continue

            role = "user" if msg.role in ["user", "tool"] else "model"
            parts: List[Dict[str, Any]] = []

            # 文本 / 图片
            if msg.content and not msg.tool_call_id:
                if isinstance(msg.content, str):
                    parts.append({"text": msg.content})
                elif isinstance(msg.content, list):
                    for item in msg.content:
                        if isinstance(item, dict):
                            if item.get("type") == "text":
                                parts.append({"text": item.get("text", "")})
                            elif item.get("type") == "image_url":
                                image_data = self._extract_image_data(
                                    item.get("image_url", {}).get("url", "")
                                )
                                if image_data:
                                    parts.append(image_data)

            # functionCall：从 OpenAI-style tool_calls 映射到 Gemini parts
            if msg.tool_calls:
                for tool_call in msg.tool_calls:
                    if tool_call.get("type") == "function":
                        function_call = tool_call.get("function", {})

                        # 使用原始的 thought_signature，如果没有则使用默认值
                        thought_sig = tool_call.get("thought_signature", "skip_thought_signature_validator")

                        part: Dict[str, Any] = {
                            "functionCall": {
                                "name": function_call.get("name", ""),
                                "args": json.loads(
                                    function_call.get("arguments", "{}")
                                ),
                            },
                            "thought_signature": thought_sig,
                        }
                        parts.append(part)

                        # 如果开启 single_tool_call，只保留第一个 functionCall
                        if self.single_tool_call:
                            break

            # functionResponse：tool 调用结果
            # 合并连续的 tool 消息到同一个 content
            if msg.tool_call_id and msg.content:
                # 收集当前和后续连续的 tool responses
                j = i
                while j < len(messages) and messages[j].role == "tool" and messages[j].tool_call_id:
                    tool_msg = messages[j]
                    content_str = tool_msg.content
                    if isinstance(tool_msg.content, list):
                        content_str = ""
                        for item in tool_msg.content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                content_str += item.get("text", "")

                    parts.append(
                        {
                            "functionResponse": {
                                "name": tool_msg.tool_call_id,
                                "response": {"result": content_str},
                            }
                        }
                    )
                    j += 1

                # 跳过已处理的 tool messages
                i = j - 1

            if parts:
                gemini_contents.append({"role": role, "parts": parts})

            i += 1

        return gemini_contents

    def _extract_image_data(self, image_url: str) -> Dict[str, Any] | None:
        if not image_url:
            return None

        if image_url.startswith("data:image"):
            mime_type = "image/jpeg"
            if "data:image/png" in image_url:
                mime_type = "image/png"
            elif "data:image/jpeg" in image_url or "data:image/jpg" in image_url:
                mime_type = "image/jpeg"

            base64_data = image_url.split(",", 1)[1] if "," in image_url else ""
            return {"inlineData": {"mimeType": mime_type, "data": base64_data}}
        else:
            return None

    def _convert_tools_to_gemini_format(
        self, tool_params: ToolParams | List[ToolSpec] | List[dict]
    ) -> List[Dict[str, Any]]:
        tools_list: List[Any] = []

        if isinstance(tool_params, ToolParams):
            tools_list = tool_params.tools
        elif isinstance(tool_params, list):
            tools_list = tool_params
        else:
            return []

        if not tools_list:
            return []

        function_declarations = []
        for tool in tools_list:
            if isinstance(tool, ToolSpec):
                function_declarations.append(
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    }
                )
            elif isinstance(tool, dict):
                if "function" in tool:
                    func = tool["function"]
                    function_declarations.append(
                        {
                            "name": func.get("name", ""),
                            "description": func.get("description", ""),
                            "parameters": func.get("parameters", {}),
                        }
                    )
                else:
                    function_declarations.append(
                        {
                            "name": tool.get("name", ""),
                            "description": tool.get("description", ""),
                            "parameters": tool.get("parameters", {}),
                        }
                    )

        return [{"functionDeclarations": function_declarations}]

    def _build_generation_config(self, params: RunParams) -> Dict[str, Any]:
        config: Dict[str, Any] = {}

        temperature = params.temperature
        if temperature is None:
            temperature = self.cfg.temperature
        if temperature is not None:
            config["temperature"] = temperature

        # top_p = params.top_p
        # if top_p is None:
        #     top_p = self.cfg.top_p
        # if top_p is not None:
        #     config["topP"] = top_p

        max_tokens = params.max_tokens
        if max_tokens is None:
            max_tokens = self.cfg.max_tokens
        if max_tokens is not None:
            config["maxOutputTokens"] = max_tokens

        # Add thinking_level support
        thinking_level = params.extra_params.get("thinking_level") if params.extra_params else None
        if thinking_level is None:
            thinking_level = self.cfg.ext.get("thinking_level", "LOW")
        if thinking_level is not None:
            config["thinkingConfig"] = {
                "thinkingLevel": thinking_level
            }

        return config

    def _process_streaming_response(
        self, response
    ) -> Generator[StreamingModelResponse, None, None]:
        buffer = ""

        for chunk in response.iter_text():
            buffer += chunk

            lines = buffer.split("\n")
            buffer = lines[-1]

            for line in lines[:-1]:
                line = line.strip()
                if not line:
                    continue

                if line.startswith("["):
                    try:
                        data = json.loads(line)
                        for item in data:
                            streaming_resp = self._parse_gemini_chunk(item)
                            if streaming_resp:
                                yield streaming_resp
                    except json.JSONDecodeError:
                        continue

    def _parse_gemini_chunk(self, chunk_data: Dict[str, Any]) -> StreamingModelResponse | None:
        if "candidates" not in chunk_data:
            return None

        candidates = chunk_data.get("candidates", [])
        if not candidates:
            return None

        candidate = candidates[0]
        content = candidate.get("content", {})
        parts = content.get("parts", [])

        text_content = ""
        tool_calls: List[Dict[str, Any]] = []

        for part in parts:
            if "text" in part:
                text_content += part["text"]
            elif "functionCall" in part:
                if self.single_tool_call and tool_calls:
                    continue

                func_call = part["functionCall"]
                thought_sig = part.get("thought_signature") or part.get("thoughtSignature")

                tool_call_obj: Dict[str, Any] = {
                    "id": f"call_{uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function": {
                        "name": func_call.get("name", ""),
                        "arguments": json.dumps(func_call.get("args", {})),
                    },
                }
                if thought_sig:
                    tool_call_obj["thought_signature"] = thought_sig

                tool_calls.append(tool_call_obj)

        delta_message = Message(
            role="assistant",
            content=text_content if text_content else None,
            tool_calls=tool_calls if tool_calls else None,
        )

        streaming_choice = StreamingChoice(
            index=0,
            delta=delta_message,
            finish_reason=candidate.get("finishReason"),
        )

        usage = None
        if "usageMetadata" in chunk_data:
            usage_data = chunk_data["usageMetadata"]
            usage = Usage(
                prompt_tokens=usage_data.get("promptTokenCount", 0),
                completion_tokens=usage_data.get("candidatesTokenCount", 0),
                total_tokens=usage_data.get("totalTokenCount", 0),
            )

        return StreamingModelResponse(
            id=str(uuid.uuid4()),
            created=int(time.time()),
            model=self.cfg.get_aggregated_model_name(),
            choices=[streaming_choice],
            usage=usage,
        )

    def _process_non_streaming_response(self, response) -> ModelResponse:
        data = response.json()

        if "candidates" not in data or not data["candidates"]:
            return ModelResponse(
                id=str(uuid.uuid4()),
                model=self.cfg.get_aggregated_model_name(),
                choices=[
                    Choice(
                        index=0,
                        message=Message(role="assistant", content=""),
                        finish_reason="stop",
                    )
                ],
            )

        candidate = data["candidates"][0]
        content = candidate.get("content", {})
        parts = content.get("parts", [])

        text_content = ""
        tool_calls: List[Dict[str, Any]] = []

        for part in parts:
            if "text" in part:
                text_content += part["text"]
            elif "functionCall" in part:
                if self.single_tool_call and tool_calls:
                    continue

                func_call = part["functionCall"]
                thought_sig = part.get("thought_signature") or part.get("thoughtSignature")

                tool_call_obj: Dict[str, Any] = {
                    "id": f"call_{uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function": {
                        "name": func_call.get("name", ""),
                        "arguments": json.dumps(func_call.get("args", {})),
                    },
                }
                if thought_sig:
                    tool_call_obj["thought_signature"] = thought_sig

                tool_calls.append(tool_call_obj)

        message = Message(
            role="assistant",
            content=text_content if text_content else None,
            tool_calls=tool_calls if tool_calls else None,
        )

        choice = Choice(
            index=0,
            message=message,
            finish_reason=candidate.get("finishReason"),
        )

        usage = None
        if "usageMetadata" in data:
            usage_data = data["usageMetadata"]
            usage = Usage(
                prompt_tokens=usage_data.get("promptTokenCount", 0),
                completion_tokens=usage_data.get("candidatesTokenCount", 0),
                total_tokens=usage_data.get("totalTokenCount", 0),
            )

        return ModelResponse(
            id=str(uuid.uuid4()),
            created=int(time.time()),
            model=self.cfg.get_aggregated_model_name(),
            choices=[choice],
            usage=usage,
        )

    def _create_error_response(
        self,
        error_message: str,
        is_streaming: bool = False,
        error_type: str = None,
        error_code: str = None,
    ) -> Union[ModelResponse, StreamingModelResponse]:
        error_object = {
            "message": error_message,
            "type": error_type or "api_error",
            "code": error_code or "gemini_error",
        }
        if is_streaming:
            return StreamingModelResponse(error=error_object)
        else:
            return ModelResponse(error=error_object)