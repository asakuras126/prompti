"""
Tool trace service for reporting custom tool calls to external services.
"""

import asyncio
import time
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import httpx
from opentelemetry import trace

from .logger import get_logger


# Setup logger
logger = get_logger(__name__)
# Get tracer for this module
_tracer = trace.get_tracer(__name__)




@dataclass
class ToolTraceEvent:
    """Event data structure for custom tool call trace."""
    
    # Tool information
    name: str  # 自定义打点工具名
    inputs: Dict[str, Any] = field(default_factory=dict)  # 自定义打点工具调用的输入
    outputs: Optional[Dict[str, Any]] = None  # 自定义打点工具调用的输出
    error: Optional[str] = None  # 自定义打点工具错误信息
    
    # Timing information
    start_at: float = field(default_factory=time.time)  # 开始时间
    end_at: Optional[float] = None  # 结束时间
    
    # Context information
    source: str = ""  # 来源
    request_id: str = ""  # 请求ID
    conversation_id: str = ""  # 会话ID
    user_id: str = ""  # 用户ID
    app_id: str = ""  # 应用ID
    
    # Tracing information
    span_id: Optional[str] = None  # span id
    parent_span_id: Optional[str] = None  # 父span id


class ToolTraceService:
    """
    Service for reporting custom tool call trace to external endpoints.
    """
    
    def __init__(
        self, 
        endpoint_url: str,
        api_key: Optional[str] = None,
        timeout: float = 10.0,
        max_retries: int = 3,
        enabled: bool = True
    ):
        """
        Initialize the tool trace service.
        
        Args:
            endpoint_url: URL of the trace endpoint
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts for failed reports
            enabled: Whether trace reporting is enabled
        """
        self.endpoint_url = endpoint_url
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.enabled = enabled
        self._http_client = None
        self._sync_http_client = None
        
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._http_client is None:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
                
            self._http_client = httpx.AsyncClient(
                timeout=self.timeout,
                headers=headers
            )
        return self._http_client
    
    def _get_sync_client(self) -> httpx.Client:
        """Get or create the synchronous HTTP client."""
        if self._sync_http_client is None:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
                
            self._sync_http_client = httpx.Client(
                timeout=self.timeout,
                headers=headers
            )
        return self._sync_http_client

    async def aclose(self):
        """Close the HTTP client if it exists."""
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None
    
    def close(self):
        """Close the synchronous HTTP client if it exists."""
        if self._sync_http_client is not None:
            self._sync_http_client.close()
            self._sync_http_client = None
    
    async def areport(self, event: ToolTraceEvent) -> bool:
        """
        Report a tool call event to the trace endpoint asynchronously.
        
        Args:
            event: The tool trace event to report
            
        Returns:
            True if the report was successfully sent, False otherwise
        """
        if not self.enabled:
            logger.debug("Tool trace reporting is disabled")
            return True
            
        if not event.conversation_id:
            logger.warning("Tool trace report skipped: conversation_id is empty")
            return False
            
        with _tracer.start_as_current_span("tool_trace.report"):
            client = await self._get_client()
            
            # Convert event to serializable dict
            payload = {
                "name": event.name,
                "inputs": event.inputs,
                "outputs": event.outputs,
                "error": event.error,
                "source": event.source,
                "start_at": event.start_at,
                "end_at": event.end_at or time.time(),
                "request_id": event.request_id,
                "conversation_id": event.conversation_id,
                "user_id": event.user_id,
                "app_id": event.app_id,
                "span_id": event.span_id,
                "parent_span_id": event.parent_span_id
            }
            
            url = self.endpoint_url + "/trace/tool/dump"
            
            # Try to send the report with retries
            for attempt in range(self.max_retries):
                try:
                    response = await client.post(
                        url,
                        json=payload,
                        timeout=self.timeout
                    )
                    
                    if response.status_code < 400:
                        logger.debug(f"Tool trace report sent successfully: {response.status_code}")
                        return True
                    
                    logger.warning(
                        f"Tool trace report failed (attempt {attempt+1}/{self.max_retries}): "
                        f"Status {response.status_code}, {response.text}"
                    )
                    
                    # Wait before retry (exponential backoff)
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(2 ** attempt * 0.1)
                        
                except Exception as e:
                    logger.warning(
                        f"Tool trace report exception (attempt {attempt+1}/{self.max_retries}): {str(e)}"
                    )
                    
                    # Wait before retry (exponential backoff)
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(2 ** attempt * 0.1)
            
            # All attempts failed
            logger.error(f"Tool trace report failed after {self.max_retries} attempts")
            return False
    
    def report(self, event: ToolTraceEvent) -> bool:
        """
        Synchronous version: Report a tool call event to the trace endpoint.
        使用异步线程执行，不阻塞调用者
        
        Args:
            event: The tool trace event to report
            
        Returns:
            True (总是返回True，因为使用异步线程执行)
        """
        if not self.enabled:
            logger.debug("Tool trace reporting is disabled")
            return True
        
        # 使用异步线程执行
        self.async_report(event)
        return True
        
    def _sync_report_internal(self, event: ToolTraceEvent) -> bool:
        """
        内部同步上报方法（仅供异步线程调用）
        
        Args:
            event: The tool trace event to report
            
        Returns:
            True if the report was successfully sent, False otherwise
        """
        client = self._get_sync_client()
        
        # Convert event to serializable dict
        payload = {
            "name": event.name,
            "inputs": event.inputs,
            "outputs": event.outputs,
            "error": event.error,
            "source": event.source,
            "start_at": event.start_at,
            "end_at": event.end_at or time.time(),
            "request_id": event.request_id,
            "conversation_id": event.conversation_id,
            "user_id": event.user_id,
            "app_id": event.app_id,
            "span_id": event.span_id,
            "parent_span_id": event.parent_span_id
        }
        
        url = self.endpoint_url + "/trace/tool/dump"

        # Try to send the report with retries
        for attempt in range(self.max_retries):
            try:
                response = client.post(
                    url,
                    json=payload,
                    timeout=self.timeout
                )
                
                if response.status_code < 400:
                    logger.debug(f"Tool trace report sent successfully: {response.status_code}")
                    return True
                
                logger.warning(
                    f"Tool trace report failed (attempt {attempt+1}/{self.max_retries}): "
                    f"Status {response.status_code}, {response.text}"
                )
                
                # Wait before retry (exponential backoff)
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt * 0.1)
                    
            except Exception as e:
                logger.warning(
                    f"Tool trace report exception (attempt {attempt+1}/{self.max_retries}): {str(e)}"
                )
                
                # Wait before retry (exponential backoff)
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt * 0.1)
        
        # All attempts failed
        logger.error(f"Tool trace report failed after {self.max_retries} attempts")
        return False

    def async_report(self, event: ToolTraceEvent):
        """
        启动一个不阻塞的线程，异步上报工具调用数据
        
        Args:
            event: The tool trace event to report
        """
        def _report_in_thread():
            try:
                self._sync_report_internal(event)
            except Exception as e:
                logger.error(f"async_report error: {e}", exc_info=True)
        
        t = threading.Thread(target=_report_in_thread, daemon=True)
        t.start()

    def report_payload(self, payload: Dict[str, Any]):
        """
        直接接收 payload 字典并上报
        
        Args:
            payload: 包含工具调用信息的字典
        """
        # 将 payload 转换为 ToolTraceEvent
        event = ToolTraceEvent(
            name=payload.get("name", ""),
            inputs=payload.get("inputs", {}),
            outputs=payload.get("outputs"),
            error=payload.get("error"),
            source=payload.get("source", "prompti"),
            start_at=payload.get("start_at", time.time()),
            end_at=payload.get("end_at"),
            request_id=payload.get("request_id", ""),
            conversation_id=payload.get("conversation_id", ""),
            user_id=payload.get("user_id", ""),
            app_id=payload.get("app_id", ""),
            span_id=payload.get("span_id"),
            parent_span_id=payload.get("parent_span_id")
        )
        
        # 异步上报
        self.async_report(event)


class ToolTraceContext:
    """工具追踪上下文管理器，简化工具调用的追踪"""
    
    def __init__(
        self,
        service: ToolTraceService,
        tool_name: str,
        inputs: Dict[str, Any],
        request_id: str = "",
        conversation_id: str = "",
        user_id: str = "",
        app_id: str = "",
        span_id: Optional[str] = None,
        parent_span_id: Optional[str] = None
    ):
        self.service = service
        self.event = ToolTraceEvent(
            name=tool_name,
            inputs=inputs,
            start_at=time.time(),
            request_id=request_id,
            conversation_id=conversation_id,
            user_id=user_id,
            app_id=app_id,
            span_id=span_id,
            parent_span_id=parent_span_id
        )
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.event.end_at = time.time()
        
        if exc_type is not None:
            # 发生异常
            self.event.error = f"{exc_type.__name__}: {str(exc_val)}"
        
        # 异步上报
        self.service.async_report(self.event)
        return False  # Don't suppress exceptions
    
    def set_outputs(self, outputs: Dict[str, Any]):
        """设置工具输出"""
        self.event.outputs = outputs
    
    def set_error(self, error: str):
        """设置错误信息"""
        self.event.error = error


# 便捷函数
def create_tool_trace_service(endpoint_url: str, **kwargs) -> ToolTraceService:
    """创建工具追踪服务实例"""
    return ToolTraceService(endpoint_url=endpoint_url, **kwargs)


def trace_tool_call(
    service: ToolTraceService,
    tool_name: str,
    inputs: Dict[str, Any],
    **context
) -> ToolTraceContext:
    """创建工具调用追踪上下文"""
    return ToolTraceContext(
        service=service,
        tool_name=tool_name,
        inputs=inputs,
        **context
    )