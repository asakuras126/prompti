"""Gemini LiteLLM client implementation for Google Vertex AI Gemini models.

Storage format:

1. LLM Token (store in llm_tokens.token_config):
Option A (Recommended - direct vertex_credentials key):
{
    "vertex_credentials": {
        "type": "service_account",
        "project_id": "your-project-id",
        "private_key_id": "...",
        "private_key": "-----BEGIN PRIVATE KEY-----\\n...\\n-----END PRIVATE KEY-----\\n",
        "client_email": "...",
        "client_id": "...",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "...",
        "client_x509_cert_url": "..."
    }
}

Option B (Auto-detect - any key with service_account type):
{
    "gemini": {
        "type": "service_account",
        "project_id": "your-project-id",
        ...
    }
}

2. Model ext (store in models.ext):
{
    "vertex_project": "cls-connectnow-gemini",
    "vertex_location": "us-central1"
}

Example Model configuration:
{
    "name": "gemini-2.5-pro",
    "value": "vertex_ai/gemini-2.5-pro",
    "provider": "gemini_litellm",
    "mode": "chat",
    "llm_tokens": ["gemini-token"],
    "ext": {
        "vertex_project": "cls-connectnow-gemini",
        "vertex_location": "us-central1"
    }
}
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Dict, Union
from collections.abc import AsyncGenerator, Generator

from .base import ModelClient, SyncModelClient, ModelConfig, RunParams, ToolParams, ToolSpec, ToolChoice, should_retry_error, calculate_retry_delay, handle_model_client_error
from ..message import Message, ModelResponse, StreamingModelResponse, Usage, Choice, StreamingChoice
from ..logger import get_logger
import litellm
import asyncio

litellm.drop_params = True

logger = get_logger(__name__)


class GeminiLiteLLMClient(ModelClient):
    """Gemini-specific LiteLLM client for Google Vertex AI Gemini models."""

    provider = "gemini_litellm"

    def __init__(self, cfg: ModelConfig, client=None, is_debug: bool = False) -> None:
        """Initialize Gemini LiteLLM client with Vertex AI-specific configurations."""
        super().__init__(cfg, client, is_debug=is_debug)
        self.api_key = cfg.api_key
        self.api_url = cfg.api_url
        self._setup_vertex_environment()

    def _setup_vertex_environment(self):
        """Setup Vertex AI-specific environment and configurations.
        
        - vertex_credentials: from token_config (supports both direct key and auto-detect)
        - vertex_project & vertex_location: from model ext
        """
        token_config = self.cfg.token_config or {}
        
        vertex_credentials = token_config.get("vertex_credentials")
        vertex_credentials_file = token_config.get("vertex_credentials_file")
        
        if vertex_credentials:
            if isinstance(vertex_credentials, dict):
                self.vertex_credentials = vertex_credentials
            else:
                self.vertex_credentials = None
        elif vertex_credentials_file and isinstance(vertex_credentials_file, str):
            try:
                with open(vertex_credentials_file, "r") as f:
                    self.vertex_credentials = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load vertex credentials from file: {e}")
                self.vertex_credentials = None
        else:
            self.vertex_credentials = None
            for key, value in token_config.items():
                if isinstance(value, dict) and value.get("type") == "service_account":
                    self.vertex_credentials = value
                    logger.debug(f"Auto-detected vertex credentials under key: {key}")
                    break
        
        self.vertex_project = self.cfg.ext.get("vertex_project", "")
        self.vertex_location = self.cfg.ext.get("vertex_location", "us-central1")
        
        logger.info(f"Configured Gemini model with project={self.vertex_project}, location={self.vertex_location}")

    async def _run(self, params: RunParams) -> AsyncGenerator[Union[ModelResponse, StreamingModelResponse], None]:
        """Execute call using litellm.acompletion() API for Gemini models."""
        max_retries = 2
        
        for attempt in range(max_retries):
            try:
                logger.debug(f"Attempt {attempt + 1}/{max_retries}")
                
                request_data = self._build_request_data(params)
                
                if params.stream:
                    response = await litellm.acompletion(**request_data)
                    async for chunk in self._aprocess_streaming_response(response):
                        yield chunk
                    return
                else:
                    response = await litellm.acompletion(**request_data)
                    yield self._process_non_streaming_response(response)
                    return
                    
            except Exception as e:
                is_last_attempt = attempt == max_retries - 1
                should_retry = should_retry_error(e)
                
                if should_retry and not is_last_attempt:
                    wait_time = calculate_retry_delay(attempt, error=e)
                    logger.warning(f"Request failed (attempt {attempt + 1}), retrying in {wait_time}s: {str(e)}")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Request failed permanently: {str(e)}")
                    yield handle_model_client_error(e, params.stream, self._create_error_response)
                    return

    def _build_request_data(self, params: RunParams) -> Dict[str, Any]:
        """Build request data for litellm.completion() API with Vertex AI credentials."""
        messages = [m.to_openai() for m in params.messages]
        
        model_name = self.cfg.get_actual_model_name()
        
        request_data = {
            "model": model_name,
            "messages": messages,
            "stream": params.stream,
        }
        
        if params.stream:
            request_data["stream_options"] = {"include_usage": True}
        
        if self.vertex_credentials:
            request_data["vertex_credentials"] = self.vertex_credentials
        if self.vertex_project:
            request_data["vertex_project"] = self.vertex_project
        if self.vertex_location:
            request_data["vertex_location"] = self.vertex_location
        
        if params.timeout is not None:
            request_data["timeout"] = params.timeout
            request_data["request_timeout"] = params.timeout
        
        request_data["num_retries"] = 0
        
        if params.temperature is not None:
            request_data["temperature"] = params.temperature
        elif self.cfg.temperature is not None:
            request_data["temperature"] = self.cfg.temperature
        
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
        
        if params.tool_params:
            if isinstance(params.tool_params, ToolParams):
                tools = [
                    {"type": "function", "function": t.model_dump()} if isinstance(t, ToolSpec) else t
                    for t in params.tool_params.tools
                ]
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
                if tools:
                    request_data["tools"] = tools
        
        request_data.update(params.extra_params)
        
        logger.debug(f"Gemini request data: {json.dumps(request_data, indent=2, ensure_ascii=False, default=str)}")
        
        params.trace_context["llm_request_body"] = request_data
        
        return request_data

    async def _aprocess_streaming_response(self, response) -> AsyncGenerator[StreamingModelResponse, None]:
        """Process streaming response."""
        async for chunk in response:
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
                        completion_tokens=chunk.usage.completion_tokens if hasattr(chunk, "usage") and chunk.usage else 0,
                        total_tokens=chunk.usage.total_tokens if hasattr(chunk, "usage") and chunk.usage else 0,
                        cache_creation_input_tokens=getattr(chunk.usage, "cache_creation_input_tokens", None) if hasattr(chunk, "usage") and chunk.usage else None,
                        cache_read_input_tokens=getattr(chunk.usage, "cache_read_input_tokens", None) if hasattr(chunk, "usage") and chunk.usage else None,
                        cached_tokens=getattr(chunk.usage, "cached_tokens", None) if hasattr(chunk, "usage") and chunk.usage else None
                    ) if hasattr(chunk, "usage") and chunk.usage else None
                )

                yield streaming_response

    def _process_non_streaming_response(self, response) -> ModelResponse:
        """Process non-streaming response."""
        if hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            message = choice.message if hasattr(choice, "message") else {}

            tool_calls = None
            if hasattr(message, "tool_calls") and message.tool_calls:
                tool_calls = []
                for tool_call in message.tool_calls:
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

            model_choice = Choice(
                index=choice.index if hasattr(choice, "index") else 0,
                message=Message(
                    role="assistant",
                    content=message.content if hasattr(message, "content") else None,
                    tool_calls=tool_calls
                ),
                finish_reason=choice.finish_reason if hasattr(choice, "finish_reason") else None
            )

            usage = None
            if hasattr(response, "usage") and response.usage:
                usage = Usage(
                    prompt_tokens=response.usage.prompt_tokens if hasattr(response.usage, "prompt_tokens") else 0,
                    completion_tokens=response.usage.completion_tokens if hasattr(response.usage, "completion_tokens") else 0,
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

    def _create_error_response(self, error_message: str, is_streaming: bool = False, error_type: str = None, error_code: str = None) -> Union[ModelResponse, StreamingModelResponse]:
        """Create error response."""
        error_object = {
            "message": error_message, 
            "type": error_type or "api_error", 
            "code": error_code or "gemini_error"
        }
        if is_streaming:
            return StreamingModelResponse(error=error_object)
        else:
            return ModelResponse(error=error_object)


class SyncGeminiLiteLLMClient(SyncModelClient):
    """Synchronous Gemini-specific LiteLLM client for Google Vertex AI Gemini models."""

    provider = "gemini_litellm"

    def __init__(self, cfg: ModelConfig, client=None, is_debug: bool = False) -> None:
        """Initialize synchronous Gemini LiteLLM client with Vertex AI-specific configurations."""
        super().__init__(cfg, client, is_debug=is_debug)
        self.api_key = cfg.api_key
        self.api_url = cfg.api_url
        self._setup_vertex_environment()

    def _setup_vertex_environment(self):
        """Setup Vertex AI-specific environment and configurations.
        
        - vertex_credentials: from token_config (supports both direct key and auto-detect)
        - vertex_project & vertex_location: from model ext
        """
        token_config = self.cfg.token_config or {}
        
        vertex_credentials = token_config.get("vertex_credentials")
        vertex_credentials_file = token_config.get("vertex_credentials_file")
        
        if vertex_credentials:
            if isinstance(vertex_credentials, dict):
                self.vertex_credentials = vertex_credentials
            else:
                self.vertex_credentials = None
        elif vertex_credentials_file and isinstance(vertex_credentials_file, str):
            try:
                with open(vertex_credentials_file, "r") as f:
                    self.vertex_credentials = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load vertex credentials from file: {e}")
                self.vertex_credentials = None
        else:
            self.vertex_credentials = None
            for key, value in token_config.items():
                if isinstance(value, dict) and value.get("type") == "service_account":
                    self.vertex_credentials = value
                    logger.debug(f"Auto-detected vertex credentials under key: {key}")
                    break
        
        self.vertex_project = self.cfg.ext.get("vertex_project", "")
        self.vertex_location = self.cfg.ext.get("vertex_location", "us-central1")
        
        logger.info(f"Configured Gemini model with project={self.vertex_project}, location={self.vertex_location}")

    def _run(self, params: RunParams) -> Generator[Union[ModelResponse, StreamingModelResponse], None, None]:
        """Execute call using litellm.completion() API for Gemini models."""
        max_retries = 2
        
        for attempt in range(max_retries):
            try:
                logger.debug(f"Attempt {attempt + 1}/{max_retries}")
                
                request_data = self._build_request_data(params)
                
                if params.stream:
                    response = litellm.completion(**request_data)
                    for chunk in self._process_streaming_response(response):
                        yield chunk
                    return
                else:
                    response = litellm.completion(**request_data)
                    yield self._process_non_streaming_response(response)
                    return
                    
            except Exception as e:
                is_last_attempt = attempt == max_retries - 1
                should_retry = should_retry_error(e)
                
                if should_retry and not is_last_attempt:
                    wait_time = calculate_retry_delay(attempt, error=e)
                    logger.warning(f"Request failed (attempt {attempt + 1}), retrying in {wait_time}s: {str(e)}")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Request failed permanently: {str(e)}")
                    yield handle_model_client_error(e, params.stream, self._create_error_response)
                    return

    def _build_request_data(self, params: RunParams) -> Dict[str, Any]:
        """Build request data for litellm.completion() API with Vertex AI credentials."""
        messages = [m.to_openai() for m in params.messages]
        
        model_name = self.cfg.get_actual_model_name()
        
        request_data = {
            "model": model_name,
            "messages": messages,
            "stream": params.stream,
        }
        
        if params.stream:
            request_data["stream_options"] = {"include_usage": True}
        
        if self.vertex_credentials:
            request_data["vertex_credentials"] = self.vertex_credentials
        if self.vertex_project:
            request_data["vertex_project"] = self.vertex_project
        if self.vertex_location:
            request_data["vertex_location"] = self.vertex_location
        
        if params.timeout is not None:
            request_data["timeout"] = params.timeout
            request_data["request_timeout"] = params.timeout
        
        request_data["num_retries"] = 0
        
        if params.temperature is not None:
            request_data["temperature"] = params.temperature
        elif self.cfg.temperature is not None:
            request_data["temperature"] = self.cfg.temperature
        
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
        
        if params.tool_params:
            if isinstance(params.tool_params, ToolParams):
                tools = [
                    {"type": "function", "function": t.model_dump()} if isinstance(t, ToolSpec) else t
                    for t in params.tool_params.tools
                ]
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
                if tools:
                    request_data["tools"] = tools
        
        request_data.update(params.extra_params)
        
        logger.debug(f"Gemini request data: {json.dumps(request_data, indent=2, ensure_ascii=False, default=str)}")
        
        params.trace_context["llm_request_body"] = request_data
        
        return request_data

    def _process_streaming_response(self, response) -> Generator[StreamingModelResponse, None, None]:
        """Process streaming response."""
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
                        completion_tokens=chunk.usage.completion_tokens if hasattr(chunk, "usage") and chunk.usage else 0,
                        total_tokens=chunk.usage.total_tokens if hasattr(chunk, "usage") and chunk.usage else 0,
                        cache_creation_input_tokens=getattr(chunk.usage, "cache_creation_input_tokens", None) if hasattr(chunk, "usage") and chunk.usage else None,
                        cache_read_input_tokens=getattr(chunk.usage, "cache_read_input_tokens", None) if hasattr(chunk, "usage") and chunk.usage else None,
                        cached_tokens=getattr(chunk.usage, "cached_tokens", None) if hasattr(chunk, "usage") and chunk.usage else None
                    ) if hasattr(chunk, "usage") and chunk.usage else None
                )

                yield streaming_response

    def _process_non_streaming_response(self, response) -> ModelResponse:
        """Process non-streaming response."""
        if hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            message = choice.message if hasattr(choice, "message") else {}

            tool_calls = None
            if hasattr(message, "tool_calls") and message.tool_calls:
                tool_calls = []
                for tool_call in message.tool_calls:
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

            model_choice = Choice(
                index=choice.index if hasattr(choice, "index") else 0,
                message=Message(
                    role="assistant",
                    content=message.content if hasattr(message, "content") else None,
                    tool_calls=tool_calls
                ),
                finish_reason=choice.finish_reason if hasattr(choice, "finish_reason") else None
            )

            usage = None
            if hasattr(response, "usage") and response.usage:
                usage = Usage(
                    prompt_tokens=response.usage.prompt_tokens if hasattr(response.usage, "prompt_tokens") else 0,
                    completion_tokens=response.usage.completion_tokens if hasattr(response.usage, "completion_tokens") else 0,
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

    def _create_error_response(self, error_message: str, is_streaming: bool = False, error_type: str = None, error_code: str = None) -> Union[ModelResponse, StreamingModelResponse]:
        """Create error response."""
        error_object = {
            "message": error_message, 
            "type": error_type or "api_error", 
            "code": error_code or "gemini_error"
        }
        if is_streaming:
            return StreamingModelResponse(error=error_object)
        else:
            return ModelResponse(error=error_object)
