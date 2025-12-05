"""Base classes and data models for model clients."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from enum import Enum
from time import perf_counter
from typing import Any, Union

import httpx
from opentelemetry import trace
from opentelemetry.baggage import set_baggage
from prometheus_client import Counter, Gauge, Histogram
from pydantic import BaseModel, Field, model_validator
from collections.abc import Generator

from ..message import Message, ModelResponse, StreamingModelResponse
from typing import Optional
from ..logger import get_logger




def should_retry_error(error: Exception) -> bool:
    """判断错误是否应该重试。

    对网络相关错误、服务器临时错误、特定的配额/速率限制错误和上下文超长错误进行重试。
    上下文超长错误也应该重试，因为会使用不同的模型兜底，后续的模型可能有更长的上下文。

    Args:
        error: 捕获到的异常

    Returns:
        bool: True表示应该重试，False表示不应该重试
    """
    import httpx

    # 检查是否为上下文长度超限错误，这类错误也应该重试（用于模型fallback）
    if is_context_length_error(str(error)):
        return True
    
    # 检查特定的异常类型
    # 对于LiteLLM的异常，先检查类型
    try:
        import litellm.exceptions
        if isinstance(error, (
            litellm.exceptions.RateLimitError,
            litellm.exceptions.APIConnectionError,
            litellm.exceptions.Timeout,
            litellm.exceptions.ServiceUnavailableError,
            litellm.exceptions.InternalServerError,
            litellm.exceptions.APIError,
        )):
            return True
    except ImportError:
        pass  # LiteLLM未安装，跳过检查
    
    # 对网络连接错误进行重试
    if isinstance(error, httpx.RequestError):
        # 进一步判断具体的网络错误类型
        if isinstance(error, (
            httpx.ConnectError,         # 连接失败
            httpx.TimeoutException,     # 超时
            httpx.ReadError,            # 读取错误
            httpx.WriteError,           # 写入错误
            httpx.PoolTimeout,          # 连接池超时
            httpx.ConnectTimeout,       # 连接超时
            httpx.ReadTimeout,          # 读取超时
            httpx.WriteTimeout,         # 写入超时
            httpx.RemoteProtocolError,  # 远程协议错误（包括服务器断开连接）
        )):
            return True
            
    # 对特定的HTTP状态码进行重试（服务器临时错误）
    if isinstance(error, httpx.HTTPStatusError):
        status_code = error.response.status_code
        
        # 对服务器临时错误和认证错误进行重试/fallback
        if status_code in (
            401,  # Unauthorized (API key issues, should fallback to other providers)
            429,  # Too Many Requests (Rate Limiting)
            500,  # Internal Server Error
            502,  # Bad Gateway
            503,  # Service Unavailable
            504,  # Gateway Timeout
            507,  # Insufficient Storage
            508,  # Loop Detected
            509,  # Bandwidth Limit Exceeded
            510,  # Not Extended
            511,  # Network Authentication Required
            520,  # Unknown Error (Cloudflare)
            521,  # Web Server Is Down (Cloudflare)
            522,  # Connection Timed Out (Cloudflare)
            523,  # Origin Is Unreachable (Cloudflare)
            524,  # A Timeout Occurred (Cloudflare)
            525,  # SSL Handshake Failed (Cloudflare)
            526,  # Invalid SSL Certificate (Cloudflare)
            527,  # Railgun Error (Cloudflare)
            530,  # Origin DNS Error (Cloudflare)
        ):
            return True
    
    # 检查异常消息中是否包含特定的错误关键词
    try:
        error_message = str(error).lower()
        
        # 服务器断开连接错误
        server_disconnect_keywords = [
            "server disconnected without sending a response",
            "server disconnected",
            "connection broken",
            "remote end closed connection",
            "connection reset by peer",
            "timed out",
            "connection"
        ]
        
        # 配额和速率限制错误
        quota_keywords = [
            "quota exceeded",
            "quota_exceeded", 
            "quota limit",
            "rate limit",
            "rate_limit",
            "request rate limit exceeded",
            "too many tokens",
            "throttled",
            "throttling",
            "model is getting throttled",
            "try your request again",
            "retry your request",
            "resource exhausted",
            "insufficient capacity",
        ]
        
        # Bedrock 特定错误
        bedrock_keywords = [
            "throttlingexception",
            "servicequotaexceedederror",
            "modelnotreadyexception",
            "internalserverexception",
            "modeltimeoutexception",
            "modelstreamingexception",
            "accessdeniedexception",  # 某些情况下可能是临时的
            "validationexception",  # 某些情况下可重试
            "output blocked by content filtering policy",  # 内容被内容过滤策略阻止
            "content filtering policy",  # 内容过滤策略相关错误
        ]
        
        # AWS 通用错误
        aws_keywords = [
            "requestlimitexceeded",
            "toomanyrequests", 
            "slowdown",
            "provisioned throughput exceeded",
            "capacity not available",
        ]
        
        # OpenAI/Azure 错误
        openai_keywords = [
            "rate_limit_exceeded",
            "insufficient_quota",
            "model_overloaded",
            "server_error",
            "deployment_not_found",  # Azure specific
            "content_filter",  # 某些情况下可重试
            "internal error",  # OpenAI 内部错误
            "unsupported parameter",  # 参数不支持，可能其他模型支持
            "not supported with this model",  # 当前模型不支持，可以尝试其他模型
            "invalid parameter",  # 无效参数，可能是模型特定的
            "parameter is not supported",  # 参数不支持的另一种表述
        ]
        
        # 网络和服务不可用错误
        service_unavailable_keywords = [
            "service unavailable",
            "service_unavailable",
            "temporarily unavailable",
            "maintenance",
            "overloaded",
            "busy",
            "capacity",
            "backoff",
            "please retry",
            "try again",
            "retry after",
            "wait",
            "cooling down",
        ]
        
        # 模型特定错误
        model_specific_keywords = [
            "model busy",
            "model unavailable", 
            "model not ready",
            "model loading",
            "model timeout",
            "inference timeout",
            "processing timeout",
            "request timeout",
            "execution timeout",
        ]
        
        # 合并所有需要重试的错误关键词
        retry_keywords = (server_disconnect_keywords + quota_keywords + 
                         bedrock_keywords + aws_keywords + openai_keywords + 
                         service_unavailable_keywords + model_specific_keywords)
        
        if any(keyword in error_message for keyword in retry_keywords):
            return True
            
    except Exception:
        # 如果获取错误消息失败，继续其他判断
        pass
    
    return False


def calculate_retry_delay(attempt: int, max_delay: int = 8, error: Exception = None) -> int:
    """计算重试延迟时间（指数退避）。
    
    Args:
        attempt: 当前尝试次数（从0开始）
        max_delay: 最大延迟时间（秒）
        error: 可选的异常对象，用于基于错误类型调整延迟
        
    Returns:
        int: 延迟时间（秒）
    """
    base_delay = min(2 ** attempt, max_delay)
    
    # 根据错误类型调整延迟时间
    if error is not None:
        error_message = str(error).lower()
        
        # 对于限流错误，使用更长的延迟
        if any(keyword in error_message for keyword in [
            "throttled", "throttling", "rate limit", "quota exceeded", 
            "too many requests", "too many tokens"
        ]):
            # 限流错误使用保守的退避策略
            return min(base_delay + 3, 15)  # 基础延迟+3秒，最多15秒
        
        # 对于服务不可用错误，使用中等延迟
        elif any(keyword in error_message for keyword in [
            "service unavailable", "overloaded", "busy", "capacity"
        ]):
            return min(base_delay + 1, 12)  # 基础延迟+1秒，最多12秒
        
        # 对于网络连接错误，使用基础延迟
        elif any(keyword in error_message for keyword in [
            "connection", "timeout", "network"
        ]):
            return min(base_delay, 8)  # 使用基础延迟，最多8秒
    
    return base_delay



def is_context_length_error(message: str) -> bool:
    """检测是否为上下文长度超限错误。
    
    Args:
        message: 错误消息字符串
        
    Returns:
        bool: True表示是上下文长度错误，False表示不是
    """
    message_lower = message.lower()
    return (
        "input length and `max_tokens` exceed context limit" in message_lower or
        "context limit" in message_lower or
        "maximum context length" in message_lower or
        "context_length_exceeded" in message_lower or
        "context window" in message_lower or
        "input characters limit" in message_lower or
        "too long" in message_lower
    )


def handle_model_client_error(error: Exception, is_streaming: bool, create_error_response_func) -> Union['ModelResponse', 'StreamingModelResponse']:
    """处理model client的最终错误（不再重试）。
    
    Args:
        error: 异常对象
        is_streaming: 是否为流式响应
        create_error_response_func: 创建错误响应的函数
        
    Returns:
        错误响应对象
    """
    import httpx

    logger = get_logger("model_client")
    
    if isinstance(error, httpx.HTTPStatusError):
        # HTTP错误（4xx, 5xx）
        error_detail = "Unknown error"
        error_type = "http_error"
        error_code = f"http_{error.response.status_code}"
        try:
            if error.response.content:
                error_data = error.response.json()
                if "error" in error_data:
                    error_obj = error_data["error"]
                    error_detail = error_obj.get("message", str(error_obj))
                    error_type = error_obj.get("type", "http_error")
                    error_code = error_obj.get("code", f"http_{error.response.status_code}")
                else:
                    error_detail = str(error_data)
            else:
                error_detail = f"HTTP {error.response.status_code}"
        except Exception:
            error_detail = f"HTTP {error.response.status_code}: {error.response.text}"

        logger.error(f"API HTTP error: {error_detail}")
        return create_error_response_func(error_detail, is_streaming=is_streaming, 
                                        error_type=error_type, error_code=error_code)
    
    elif isinstance(error, httpx.RequestError):
        # 网络连接错误
        error_msg = f"Network error: {str(error)}"
        logger.error(error_msg)
        return create_error_response_func(error_msg, is_streaming=is_streaming,
                                        error_type="network_error", error_code="request_failed")
    
    else:
        # 其他错误
        error_msg = f"Unexpected error: {str(error)}"
        logger.error(error_msg)
        return create_error_response_func(error_msg, is_streaming=is_streaming,
                                        error_type="internal_error", error_code="unexpected_error")


class ModelConfig(BaseModel):
    """Static connection and default generation parameters."""

    provider: Optional[str] | None = None
    model: Optional[str] | None = None  # 聚合模型名称（用于模型分组和查找）
    model_value: Optional[str] | None = None  # 真实调用的模型名称
    api_key: Optional[str] | None = None
    api_url: Optional[str] | None = None

    # generation defaults (may be overridden per call)
    temperature: Optional[float] = None
    top_p: Optional[float] | None = None
    max_tokens: Optional[int] | None = None
    
    # load balancing weight for multiple providers of the same model
    weight: Optional[int] = 50
    
    # extension fields from promptstore model configuration
    ext: dict[str, Any] = {}
    
    # token configuration from llm_tokens (credentials, etc.)
    token_config: dict[str, Any] = {}
    
    # extra parameters for client construction
    extra_params: dict[str, Any] = {}
    
    def get_aggregated_model_name(self) -> str:
        """Get the model name to use for aggregation purposes.
        
        Returns:
            str: The aggregated model name (model field).
        """
        return self.model or ""
    
    def get_actual_model_name(self) -> str:
        """Get the actual model name for API calls.
        
        Returns:
            str: The actual model name (model_value field), fallback to model if not set.
        """
        return self.model_value or self.model or ""


class ToolSpec(BaseModel):
    """Specification for a single tool."""

    name: str
    description: str
    parameters: dict[str, Any]


class ToolChoice(str, Enum):
    """Allowed tool invocation policies."""

    AUTO = "auto"
    BLOCK = "none"
    REQUIRED = "required"
    FORCE = "force"


class ModelControlParams(BaseModel):
    """Dynamic model control parameters for runtime adjustment."""
    
    # Weight overrides for specific providers or models
    # Format: {"provider_name": weight} or {"provider_name/model_name": weight}
    weight_overrides: dict[str, int] | None = None
    
    # Disabled providers or models
    # Format: ["provider_name"] or ["provider_name/model_name"]
    disabled_models: list[str] | None = None
    
    # Enabled providers or models (if specified, only these will be used)
    # Format: ["provider_name"] or ["provider_name/model_name"]
    enabled_models: list[str] | None = None


class ToolParams(BaseModel):
    """Tool catalogue and invocation configuration."""

    tools: list[ToolSpec]
    choice: ToolChoice | dict[str, Any] = ToolChoice.AUTO
    force_tool: str | None = None
    parallel_allowed: bool = True
    max_calls: int | None = None


class RunParams(BaseModel):
    """Per-call parameters for :class:`ModelClient.run`."""

    messages: list[Message]
    tool_params: ToolParams | list[ToolSpec] | list[dict] | None = None
    
    # dynamic model control
    model_control: ModelControlParams | None = None

    # sampling & length
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_tokens: int | None = None
    stop: str | list[str] | None = None

    # control & reproducibility
    stream: bool = True
    n: int | None = None
    seed: int | None = None
    logit_bias: dict[int, float] | None = None
    response_format: str | None = None

    # misc
    user_id: str | None = None
    request_id: str | None = None
    app_id: str | None = None
    session_id: str | None = None  # Deprecated: use conversation_id instead
    conversation_id: str | None = None
    span_id : str | None = None
    parent_span_id : str | None = None
    source: str | None = None
    timeout: float | None = None
    extra_params: dict[str, Any] = {}

    
    # trace data capture - used to pass data between engine and model client
    trace_context: dict[str, Any] = {}
    
    @model_validator(mode='before')
    @classmethod
    def handle_session_conversation_compatibility(cls, data):
        """Handle backward compatibility between session_id and conversation_id."""
        if isinstance(data, dict):
            # If only session_id is provided, copy to conversation_id
            if 'session_id' in data and 'conversation_id' not in data:
                data['conversation_id'] = data['session_id']
            # If only conversation_id is provided, copy to session_id for backward compatibility
            elif 'conversation_id' in data and 'session_id' not in data:
                data['session_id'] = data['conversation_id']
            # If both are provided, conversation_id takes precedence
            elif 'conversation_id' in data and 'session_id' in data:
                data['session_id'] = data['conversation_id']
        
        return data


class ModelClient:
    """Base class for model clients."""

    provider: str = "generic"

    _counter = Counter("llm_tokens_total", "Tokens in/out", labelnames=["direction"])
    _histogram = Histogram("llm_request_latency_seconds", "LLM latency", labelnames=["provider"])
    _inflight = Gauge(
        "llm_inflight_requests",
        "Inflight LLM requests",
        labelnames=["provider", "is_error"],
    )
    _request_counter = Counter(
        "llm_requests_total",
        "LLM request results",
        labelnames=["provider", "result", "is_error"],
    )
    _first_token = Histogram(
        "llm_first_token_latency_seconds",
        "Time to first token",
        labelnames=["provider", "model"],
        buckets=(0.1, 0.25, 0.5, 1, 2, 5, 10),
    )
    _token_gap = Histogram(
        "llm_stream_intertoken_gap_seconds",
        "Gap between streamed tokens",
        labelnames=["provider", "model"],
    )
    _prompt_tokens = Counter(
        "llm_prompt_tokens_total",
        "Prompt tokens sent to the provider",
        labelnames=["provider", "model"],
    )
    _completion_tokens = Counter(
        "llm_completion_tokens_total",
        "Completion tokens received from the provider",
        labelnames=["provider", "model"],
    )

    def __init__(
        self, cfg: ModelConfig, client: httpx.AsyncClient | None = None, is_debug: bool = False, **_: Any
    ) -> None:
        """Create the client with static :class:`ModelConfig` and optional HTTP client."""
        self.cfg = cfg
        self._client = client or httpx.AsyncClient(http2=True, timeout=httpx.Timeout(600))
        self._tracer = trace.get_tracer(__name__)
        self._logger = get_logger("model_client")
        self._is_debug = is_debug

        if self._is_debug:
            self._client.event_hooks.setdefault("request", []).append(self._log_request)
            self._client.event_hooks.setdefault("response", []).append(self._log_response)
        else:
            self._client.event_hooks.setdefault("request", []).append(self._log_request_jsonl)
            self._client.event_hooks.setdefault("response", []).append(self._log_response_jsonl)

    async def _log_request(self, request: httpx.Request) -> None:
        """Log outgoing HTTP request details as a cURL command."""
        import shlex

        command = f"curl -X {request.method} '{request.url}'"
        for k, v in request.headers.items():
            command += f" \\\n  -H '{k}: {v}'"

        body_bytes = request.content
        if body_bytes:
            body_str = ""
            try:
                body_str = body_bytes.decode()
            except UnicodeDecodeError:
                body_str = "<...binary data...>"

            command += f" \\\n  -d {shlex.quote(body_str)}"

        self._logger.info(f"http request as curl:\n{command}")

    async def _log_response(self, response: httpx.Response) -> None:
        """Log incoming HTTP response details."""
        # Log response in structured format similar to cURL output
        log_lines = [
            f"http response: {response.status_code} {response.reason_phrase}",
            f"  url: {response.url}"
        ]

        # Add headers
        for k, v in response.headers.items():
            log_lines.append(f"  header '{k}: {v}'")

        # Only read content for non-streaming responses
        content_type = response.headers.get("content-type", "")
        if not content_type.startswith("text/event-stream"):
            content = await response.aread()
            text = content.decode(response.encoding or "utf-8", "replace")
            if text:
                log_lines.append(f"  body: {text}")
            response._content = content

        self._logger.info("\n".join(log_lines))

    def _sanitize_headers(self, headers: dict[str, str]) -> dict[str, str]:
        """Remove sensitive information from headers."""
        sensitive_keys = {"authorization", "x-api-key", "api-key", "bearer"}
        sanitized = {}
        for k, v in headers.items():
            if k.lower() in sensitive_keys:
                sanitized[k] = "[REDACTED]"
            else:
                sanitized[k] = v
        return sanitized

    def _sanitize_body(self, body: str) -> dict[str, Any] | str:
        """Remove sensitive information from request/response body."""
        try:
            data = json.loads(body)
            if isinstance(data, dict):
                # Remove or mask sensitive fields
                sanitized = data.copy()
                # Common sensitive fields to redact
                sensitive_fields = {"api_key", "authorization", "token", "secret", "password"}
                for field in sensitive_fields:
                    if field in sanitized:
                        sanitized[field] = "[REDACTED]"
                return sanitized
            return data
        except (json.JSONDecodeError, UnicodeDecodeError):
            # If not JSON or can't decode, return truncated string
            return body[:1000] + "..." if len(body) > 1000 else body

    async def _log_request_jsonl(self, request: httpx.Request) -> None:
        """Log outgoing HTTP request in JSONL format for production use."""
        body_str = ""
        if request.content:
            try:
                body_str = request.content.decode()
            except UnicodeDecodeError:
                body_str = "<binary data>"

        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "http_request",
            "method": request.method,
            "url": str(request.url),
            "headers": self._sanitize_headers(dict(request.headers)),
            "body": self._sanitize_body(body_str) if body_str else None,
            "provider": self.cfg.provider,
            "model": self.cfg.model,
        }

        self._logger.debug(json.dumps(log_data, separators=(",", ":")))

    async def _log_response_jsonl(self, response: httpx.Response) -> None:
        """Log incoming HTTP response in JSONL format for production use."""
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "http_response",
            "status_code": response.status_code,
            "reason_phrase": response.reason_phrase,
            "url": str(response.url),
            "headers": self._sanitize_headers(dict(response.headers)),
            "provider": self.cfg.provider,
            "model": self.cfg.model,
        }

        # Only read content for non-streaming responses
        content_type = response.headers.get("content-type", "")
        if not content_type.startswith("text/event-stream"):
            content = await response.aread()
            text = content.decode(response.encoding or "utf-8", "replace")
            log_data["body"] = self._sanitize_body(text) if text else None
            response._content = content
        else:
            log_data["body"] = "<streaming response>"

        self._logger.debug(json.dumps(log_data, separators=(",", ":")))

    async def arun(self, params: RunParams) -> AsyncGenerator[Union[ModelResponse, StreamingModelResponse], None]:
        """Execute the LLM call with dynamic ``params``.
        
        Returns:
            AsyncGenerator yielding ModelResponse for non-streaming calls or 
            StreamingResponse for streaming calls.
        """
        is_error = False
        self._inflight.labels(self.cfg.provider, "false").inc()
        result = "success"
        start = perf_counter()
        first = True
        last = start
        attrs = {
            "provider": self.cfg.provider,
            "model": self.cfg.model,
        }

        # 初始化或更新遥测上下文，包含通用请求数据
        if "llm_request_body" not in params.trace_context:
            request_body = {
                "model": self.cfg.model,
                "provider": self.cfg.provider,
                "messages": [msg.model_dump() for msg in params.messages],
                "stream": params.stream,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            # 添加tool_params信息
            if params.tool_params is not None:
                request_body["tool_params"] = params.tool_params.model_dump() if hasattr(params.tool_params, 'model_dump') else params.tool_params
            params.trace_context["llm_request_body"] = request_body
        # 初始化响应体容器
        params.trace_context["llm_response_body"] = {}
        params.trace_context["responses"] = []

        if params.request_id:
            attrs["http.request_id"] = params.request_id
        if params.app_id:
            attrs["http.app_id"] = params.app_id
        if params.session_id:
            attrs["user.session_id"] = params.session_id
        if params.conversation_id:
            attrs["user.conversation_id"] = params.conversation_id
        if params.user_id:
            attrs["user.id"] = params.user_id

        for key, val in (
            ("request_id", params.request_id),
            ("app_id", params.app_id),
            ("session_id", params.session_id),  # Keep for backward compatibility
            ("conversation_id", params.conversation_id),  # New field
            ("user_id", params.user_id),
        ):
            if val:
                set_baggage(key, val)

        with (
            self._tracer.start_as_current_span("llm.call", attributes=attrs),
            self._histogram.labels(self.cfg.provider).time(),
        ):
            params.trace_context["perf_metrics"] = {}
            try:
                async for response in self._run(params):
                    now = perf_counter()
                    if first:
                        self._first_token.labels(self.cfg.provider, self.cfg.model).observe(now - start)
                        params.trace_context["perf_metrics"]["first_package_latency"] = now - start
                        params.trace_context["perf_metrics"]["total_latency"] = now - start
                        first = False
                    else:
                        self._token_gap.labels(self.cfg.provider, self.cfg.model).observe(now - last)
                        params.trace_context["perf_metrics"]["total_latency"] = now - start
                    last = now
                    yield response

            except Exception as e:
                is_error = True
                result = "error"
                raise
            finally:
                self._inflight.labels(self.cfg.provider, "false").dec()
                self._request_counter.labels(self.cfg.provider, result, str(is_error).lower()).inc()

    async def _run(self, params: RunParams) -> AsyncGenerator[Union[ModelResponse, StreamingModelResponse], None]:
        """Internal method to be implemented by subclasses.
        
        Args:
            params: Runtime parameters for the model call
            
        Returns:
            AsyncGenerator yielding ModelResponse for non-streaming calls or 
            StreamingResponse for streaming calls.
        """
        raise NotImplementedError
        yield  # pragma: no cover - satisfies generator type

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()
    
    # Backward compatibility alias
    async def close(self) -> None:
        """Deprecated: use aclose() instead."""
        import warnings
        warnings.warn("ModelClient.close() is deprecated, use aclose() instead", DeprecationWarning, stacklevel=2)
        await self.aclose()
    
    # Backward compatibility alias
    async def run(self, params: RunParams) -> AsyncGenerator[Union[ModelResponse, StreamingModelResponse], None]:
        """Deprecated: use arun() instead."""
        import warnings
        warnings.warn("ModelClient.run() is deprecated, use arun() instead", DeprecationWarning, stacklevel=2)
        async for response in self.arun(params):
            yield response


class SyncModelClient:
    """Synchronous base class for model clients."""

    provider: str = "generic"

    # Reuse the same metrics from async version to avoid duplication
    _counter = ModelClient._counter
    _histogram = ModelClient._histogram
    _inflight = ModelClient._inflight
    _request_counter = ModelClient._request_counter
    _first_token = ModelClient._first_token
    _token_gap = ModelClient._token_gap
    _prompt_tokens = ModelClient._prompt_tokens
    _completion_tokens = ModelClient._completion_tokens

    def __init__(
        self, cfg: ModelConfig, client: httpx.Client | None = None, is_debug: bool = False, **_: Any
    ) -> None:
        """Create the client with static :class:`ModelConfig` and optional HTTP client."""
        self.cfg = cfg
        self._client = client or httpx.Client(http2=True, timeout=httpx.Timeout(600))
        self._tracer = trace.get_tracer(__name__)
        self._logger = get_logger("model_client")
        self._is_debug = is_debug

        if self._is_debug:
            self._client.event_hooks.setdefault("request", []).append(self._log_request)
            self._client.event_hooks.setdefault("response", []).append(self._log_response)
        else:
            self._client.event_hooks.setdefault("request", []).append(self._log_request_jsonl)
            self._client.event_hooks.setdefault("response", []).append(self._log_response_jsonl)

    def _log_request(self, request: httpx.Request) -> None:
        """Log outgoing HTTP request details as a cURL command."""
        import shlex

        command = f"curl -X {request.method} '{request.url}'"
        for k, v in request.headers.items():
            command += f" \\\n  -H '{k}: {v}'"

        body_bytes = request.content
        if body_bytes:
            body_str = ""
            try:
                body_str = body_bytes.decode()
            except UnicodeDecodeError:
                body_str = "<...binary data...>"

            command += f" \\\n  -d {shlex.quote(body_str)}"

        self._logger.info(f"http request as curl:\n{command}")

    def _log_response(self, response: httpx.Response) -> None:
        """Log incoming HTTP response details."""
        log_lines = [
            f"http response: {response.status_code} {response.reason_phrase}",
            f"  url: {response.url}"
        ]

        for k, v in response.headers.items():
            log_lines.append(f"  header '{k}: {v}'")

        content_type = response.headers.get("content-type", "")
        if not content_type.startswith("text/event-stream"):
            content = response.read()
            text = content.decode(response.encoding or "utf-8", "replace")
            if text:
                log_lines.append(f"  body: {text}")
            response._content = content

        self._logger.info("\n".join(log_lines))

    def _log_request_jsonl(self, request: httpx.Request) -> None:
        """Log outgoing HTTP request in JSONL format for production use."""
        body_str = ""
        if request.content:
            try:
                body_str = request.content.decode()
            except UnicodeDecodeError:
                body_str = "<binary data>"

        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "http_request",
            "method": request.method,
            "url": str(request.url),
            "headers": self._sanitize_headers(dict(request.headers)),
            "body": self._sanitize_body(body_str) if body_str else None,
            "provider": self.cfg.provider,
            "model": self.cfg.model,
        }

        self._logger.debug(json.dumps(log_data, separators=(",", ":"), ensure_ascii=False))

    def _log_response_jsonl(self, response: httpx.Response) -> None:
        """Log incoming HTTP response in JSONL format for production use."""
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "http_response",
            "status_code": response.status_code,
            "reason_phrase": response.reason_phrase,
            "url": str(response.url),
            "headers": self._sanitize_headers(dict(response.headers)),
            "provider": self.cfg.provider,
            "model": self.cfg.model,
        }

        content_type = response.headers.get("content-type", "")
        if not content_type.startswith("text/event-stream"):
            content = response.read()
            text = content.decode(response.encoding or "utf-8", "replace")
            log_data["body"] = self._sanitize_body(text) if text else None
            response._content = content
        else:
            log_data["body"] = "<streaming response>"

        self._logger.debug(json.dumps(log_data, separators=(",", ":"), ensure_ascii=False))

    def _sanitize_headers(self, headers: dict[str, str]) -> dict[str, str]:
        """Remove sensitive information from headers."""
        sensitive_keys = {"authorization", "x-api-key", "api-key", "bearer"}
        sanitized = {}
        for k, v in headers.items():
            if k.lower() in sensitive_keys:
                sanitized[k] = "[REDACTED]"
            else:
                sanitized[k] = v
        return sanitized

    def _sanitize_body(self, body: str) -> dict[str, Any] | str:
        """Remove sensitive information from request/response body."""
        try:
            data = json.loads(body)
            if isinstance(data, dict):
                sanitized = data.copy()
                sensitive_fields = {"api_key", "authorization", "token", "secret", "password"}
                for field in sensitive_fields:
                    if field in sanitized:
                        sanitized[field] = "[REDACTED]"
                return sanitized
            return data
        except (json.JSONDecodeError, UnicodeDecodeError):
            return body[:1000] + "..." if len(body) > 1000 else body

    def run(self, params: RunParams) -> Generator[Union[ModelResponse, StreamingModelResponse], None, None]:
        """Execute the LLM call with dynamic ``params``.
        
        Returns:
            Generator yielding ModelResponse for non-streaming calls or 
            StreamingResponse for streaming calls.
        """
        is_error = False
        self._inflight.labels(self.cfg.provider, "false").inc()
        result = "success"
        start = perf_counter()
        first = True
        last = start
        attrs = {
            "provider": self.cfg.provider,
            "model": self.cfg.model,
        }

        params.trace_context["llm_response_body"] = {}
        params.trace_context["responses"] = []

        if params.request_id:
            attrs["http.request_id"] = params.request_id
        if params.app_id:
            attrs["http.app_id"] = params.app_id
        if params.session_id:
            attrs["user.session_id"] = params.session_id
        if params.conversation_id:
            attrs["user.conversation_id"] = params.conversation_id
        if params.user_id:
            attrs["user.id"] = params.user_id

        for key, val in (
            ("request_id", params.request_id),
            ("app_id", params.app_id),
            ("session_id", params.session_id),  # Keep for backward compatibility
            ("conversation_id", params.conversation_id),  # New field
            ("user_id", params.user_id),
        ):
            if val:
                set_baggage(key, val)

        with (
            self._tracer.start_as_current_span("llm.call", attributes=attrs),
            self._histogram.labels(self.cfg.provider).time(),
        ):
            params.trace_context["perf_metrics"] = {}
            try:
                for response in self._run(params):
                    now = perf_counter()
                    if first:
                        self._first_token.labels(self.cfg.provider, self.cfg.model).observe(now - start)
                        params.trace_context["perf_metrics"]["first_package_latency"] = now - start
                        params.trace_context["perf_metrics"]["total_latency"] = now - start
                        first = False
                    else:
                        self._token_gap.labels(self.cfg.provider, self.cfg.model).observe(now - last)
                        params.trace_context["perf_metrics"]["total_latency"] = now - start
                    last = now
                    yield response

            except Exception as e:
                is_error = True
                result = "error"
                raise
            finally:
                self._inflight.labels(self.cfg.provider, "false").dec()
                self._request_counter.labels(self.cfg.provider, result, str(is_error).lower()).inc()

    def _run(self, params: RunParams) -> Generator[Union[ModelResponse, StreamingModelResponse], None, None]:
        """Internal method to be implemented by subclasses.
        
        Args:
            params: Runtime parameters for the model call
            
        Returns:
            Generator yielding ModelResponse for non-streaming calls or 
            StreamingResponse for streaming calls.
        """
        raise NotImplementedError
        yield  # pragma: no cover - satisfies generator type

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()
