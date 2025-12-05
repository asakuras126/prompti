"""
Trace service for reporting model calls to external services.
"""
from typing import Any, Dict, List, Optional, Union
import asyncio
import logging
import time
from dataclasses import dataclass, field

import httpx
from opentelemetry import trace


def decode_unicode_content(content: str) -> str:
    """解码unicode编码的内容，使其可读，类似json.loads处理\\uXXXX的方式，但只处理unicode序列，保留其他转义序列如\\n
    
    这个函数模拟了JSON解析器中处理unicode转义序列的行为，但只针对\\uXXXX格式，
    不会像codecs.decode('unicode-escape')那样处理所有转义序列。
    """
    if not content or not isinstance(content, str):
        return content
    
    # 快速检查：只有当字符串包含unicode转义序列时才进行解码
    if '\\u' not in content:
        return content
    
    try:
        import re
        
        def process_unicode_sequences(text):
            """处理所有的\\uXXXX序列，包括emoji的代理对"""
            unicode_pattern = r'\\u[0-9a-fA-F]{4}'
            matches = list(re.finditer(unicode_pattern, text))
            
            if not matches:
                return text
            
            result = []
            last_end = 0
            i = 0
            
            while i < len(matches):
                match = matches[i]
                # 添加前面的非unicode部分
                result.append(text[last_end:match.start()])
                
                # 获取当前unicode码点
                hex_code = match.group(0)[2:]  # 去掉\\u前缀
                code_point = int(hex_code, 16)
                
                # 处理UTF-16代理对（emoji等字符需要）
                if 0xD800 <= code_point <= 0xDBFF and i + 1 < len(matches):
                    # 当前是高代理项，检查下一个是否是低代理项
                    next_match = matches[i + 1]
                    if next_match.start() == match.end():  # 必须紧邻
                        next_hex = next_match.group(0)[2:]
                        next_code = int(next_hex, 16)
                        if 0xDC00 <= next_code <= 0xDFFF:  # 确实是低代理项
                            # 组合成完整的unicode字符（参考UTF-16解码算法）
                            combined_code = 0x10000 + ((code_point - 0xD800) << 10) + (next_code - 0xDC00)
                            try:
                                result.append(chr(combined_code))
                                last_end = next_match.end()
                                i += 2  # 跳过这两个序列
                                continue
                            except (ValueError, OverflowError):
                                # 如果组合失败，按单个字符处理
                                pass
                
                # 处理单个unicode字符或失败的代理项
                try:
                    if 0xD800 <= code_point <= 0xDFFF:
                        # 孤立的代理项，保留原样（避免编码错误）
                        result.append(match.group(0))
                    else:
                        # 正常的unicode字符
                        result.append(chr(code_point))
                except (ValueError, OverflowError):
                    # 无效的unicode码点，保留原样
                    result.append(match.group(0))
                
                last_end = match.end()
                i += 1
            
            # 添加剩余的文本
            result.append(text[last_end:])
            return ''.join(result)
        
        return process_unicode_sequences(content)
        
    except Exception:
        # 如果任何步骤失败，返回原内容
        return content


def decode_unicode_in_data(data: Any) -> Any:
    """递归地在数据结构中解码unicode内容"""
    if isinstance(data, str):
        return decode_unicode_content(data)
    elif isinstance(data, dict):
        return {key: decode_unicode_in_data(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [decode_unicode_in_data(item) for item in data]
    else:
        return data


def mask_sensitive_fields(data: Any) -> Any:
    """递归地在数据结构中mask敏感字段，如api_key"""
    if isinstance(data, dict):
        masked_data = {}
        for key, value in data.items():
            if key.lower() == 'api_key' and isinstance(value, str) and value:
                # mask api_key字段，只显示前4位和后4位
                if len(value) <= 8:
                    masked_data[key] = '***'
                else:
                    masked_data[key] = value[:4] + '***' + value[-4:]
            else:
                masked_data[key] = mask_sensitive_fields(value)
        return masked_data
    elif isinstance(data, list):
        return [mask_sensitive_fields(item) for item in data]
    else:
        return data


def restore_original_urls_in_messages(messages: Any) -> Any:
    """恢复messages中的原始图片URL，用于trace上报
    
    通过base64 hash查找映射表，将base64 data URL恢复为原始URL
    
    Args:
        messages: 可能包含base64图片URL的messages数据
        
    Returns:
        恢复原始URL的messages数据
    """
    # Import here to avoid circular imports
    try:
        from .model_client.image_utils import _compute_base64_hash, get_original_url_by_hash
    except ImportError:
        # Fallback if import fails
        logger.warning("Failed to import image_utils functions for URL restoration")
        return messages
    
    if isinstance(messages, list):
        result = []
        for msg in messages:
            if isinstance(msg, dict):
                msg_copy = dict(msg)
                content = msg_copy.get("content")
                
                if isinstance(content, list):
                    # 处理多模态内容
                    new_content = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "image_url":
                            item_copy = dict(item)
                            image_url_obj = item_copy.get("image_url", {})
                            if isinstance(image_url_obj, dict):
                                image_url_copy = dict(image_url_obj)
                                url = image_url_copy.get("url", "")
                                
                                # 如果是base64 data URL，尝试从映射表中恢复原始URL
                                if url.startswith("data:"):
                                    hash_key = _compute_base64_hash(url)
                                    if hash_key:
                                        original_url = get_original_url_by_hash(hash_key)
                                        if original_url:
                                            image_url_copy["url"] = original_url
                                            logger.debug(f"Restored original URL for trace: {original_url}")
                                        else:
                                            # 如果映射表中找不到，使用占位符
                                            image_url_copy["url"] = "<converted_from_original_url_to_base64>"
                                            logger.debug("URL mapping not found, using placeholder")
                                    else:
                                        # 如果无法计算hash，保持原样或使用占位符
                                        image_url_copy["url"] = "<converted_from_original_url_to_base64>"
                                        logger.debug("Failed to compute hash, using placeholder")
                                
                                item_copy["image_url"] = image_url_copy
                            new_content.append(item_copy)
                        else:
                            new_content.append(item)
                    msg_copy["content"] = new_content
                
                result.append(msg_copy)
            else:
                result.append(msg)
        return result
    else:
        return messages

from .logger import get_logger

logger = get_logger(__name__)

# Setup logger

# Get tracer for this module
_tracer = trace.get_tracer(__name__)


def _concatenate_streaming_responses(original_responses: List[Dict[str, Any]], final_responses: List[Dict[str, Any]], hook_response_data: Dict[str, List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    将流式响应拼接成完整的结构化数据。
    
    Args:
        original_responses: 原始模型响应的列表（未经hook处理）
        final_responses: 最终响应的列表（经过hook处理）
        hook_response_data: 每个hook的完整响应数据，key为hook类名，value为响应数据列表
        
    Returns:
        包含拼接后的完整数据的字典，包括:
        - original_llm_response: dict包含content，reasoning_content，tool_calls
        - hook_responses: list[dict] 每个hook的响应内容
        - final_response: dict包含content，reasoning_content，tool_calls
    """
    
    def extract_content_from_responses(responses):
        """从响应列表中提取内容"""
        concatenated_content = ""
        concatenated_reasoning_content = ""
        tool_calls_map = {}
        
        for response in responses:
            choices = response.get("choices", [])
            if not choices:
                continue
                
            first_choice = choices[0]
            
            # 处理流式响应 (delta) 或非流式响应 (message)
            data_source = first_choice.get("delta") or first_choice.get("message", {})
            
            # 提取content
            content = data_source.get("content")
            if content is not None:
                if isinstance(content, str):
                    concatenated_content += content
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text = item.get("text", "")
                            if text:
                                concatenated_content += text
            
            # 提取reasoning_content
            reasoning_content = data_source.get("reasoning_content")
            if reasoning_content is not None and isinstance(reasoning_content, str):
                concatenated_reasoning_content += reasoning_content
            
            # 处理tool_calls (需要按index正确拼接arguments)
            tool_calls = data_source.get("tool_calls")
            if tool_calls and isinstance(tool_calls, list):
                for tool_call in tool_calls:
                    if isinstance(tool_call, dict):
                        tool_call_id = tool_call.get("id")
                        tool_call_index = tool_call.get("index")
                        
                        # 对于有id的tool call（通常是第一个chunk）
                        if tool_call_id:
                            if tool_call_id not in tool_calls_map:
                                tool_calls_map[tool_call_id] = {
                                    "id": tool_call_id,
                                    "type": tool_call.get("type", "function"),
                                    "function": {
                                        "name": tool_call.get("function", {}).get("name", ""),
                                        "arguments": tool_call.get("function", {}).get("arguments", "")
                                    }
                                }
                            else:
                                # 追加arguments内容
                                if "function" in tool_call and "arguments" in tool_call["function"]:
                                    tool_calls_map[tool_call_id]["function"]["arguments"] += tool_call["function"]["arguments"]
                        
                        # 对于只有index的tool call（后续chunks）
                        elif tool_call_index is not None:
                            # 尝试根据index找到对应的tool call
                            for existing_id, existing_call in tool_calls_map.items():
                                # 简单匹配：假设index顺序与创建顺序一致
                                if list(tool_calls_map.keys()).index(existing_id) == tool_call_index:
                                    if "function" in tool_call and "arguments" in tool_call["function"]:
                                        existing_call["function"]["arguments"] += tool_call["function"]["arguments"]
                                    break
        
        # 转换tool_calls为列表
        final_tool_calls = []
        for tool_call in tool_calls_map.values():
            final_call = {
                "id": tool_call.get("id", ""),
                "type": tool_call.get("type", "function"),
                "function": tool_call.get("function", {})
            }
            final_tool_calls.append(final_call)
        
        return concatenated_content, concatenated_reasoning_content, final_tool_calls
    
    # 如果没有响应数据
    if not original_responses and not final_responses:
        result = {
            "original_llm_response": {
                "content": "",
                "reasoning_content": "",
                "tool_calls": []
            },
            "hook_responses": [],
            "final_response": {
                "content": "",
                "reasoning_content": "",
                "tool_calls": []
            }
        }
        return result
    
    # 提取原始响应内容
    original_content, original_reasoning, original_tool_calls = extract_content_from_responses(original_responses or [])
    
    # 提取最终响应内容  
    final_content, final_reasoning, final_tool_calls = extract_content_from_responses(final_responses or [])
    
    # 修复被截断的tool_calls - 如果final response的tool_calls arguments比original短，可能是被截断了
    if original_tool_calls and final_tool_calls:
        for i, final_tool_call in enumerate(final_tool_calls):
            if i < len(original_tool_calls):
                original_args = original_tool_calls[i].get("function", {}).get("arguments", "")
                final_args = final_tool_call.get("function", {}).get("arguments", "")
                
                # 如果final的arguments明显比original短且不是完整的JSON，可能被截断了
                if (len(final_args) < len(original_args) and 
                    final_args and 
                    not final_args.strip().endswith('}')):
                    # 记录警告但不自动修复，因为可能是预期的处理结果
                    logger.warning(f"Tool call arguments may be truncated: final={len(final_args)}, original={len(original_args)}")
    
    # 构建hook_responses数组 - 使用真实的hook响应数据
    hook_responses = []
    if hook_response_data:
        for hook_name, hook_data_list in hook_response_data.items():
            # 从hook的所有响应数据中提取内容
            hook_content, hook_reasoning, hook_tool_calls = extract_content_from_responses(hook_data_list)
            
            hook_responses.append({
                "hook_name": hook_name,
                "content": hook_content,
                "reasoning_content": hook_reasoning,
                "tool_calls": hook_tool_calls
            })
    
    # 构建结果
    result = {
        "original_llm_response": {
            "content": original_content,
            "reasoning_content": original_reasoning,
            "tool_calls": original_tool_calls
        },
        "hook_responses": hook_responses,
        "final_response": {
            "content": final_content,
            "reasoning_content": final_reasoning,
            "tool_calls": final_tool_calls
        }
    }
    
    return result


@dataclass
class TraceEvent:
    """Event data structure for model call trace."""
    
    # Request information
    template_name: str
    template_id: Optional[str] = ""
    template_version: Optional[str] = None
    variant: Optional[str] = None

    model: str = ""
    messages_template: List[Dict[str, Any]] = field(default_factory=list)
    variables: Optional[Dict[str, Any]] = None
    query: str = ""

    # Request information
    llm_request_body: Dict[str, Any] = field(default_factory=dict)  # messages, original_messages, mapping等
    llm_response_body: Dict[str, Any] = field(default_factory=dict)  # responses, final_responses, concatenated_response

    # Metadatao
    request_id: str = ""
    app_id: str = ""
    timestamp: float = field(default_factory=time.time)
    conversation_id: str = ""
    user_id: str = ""
    duration_ms: Optional[float] = None
    token_usage: Dict[str, int] = field(default_factory=dict)
    error: Optional[str] = None
    source: Optional[str] = None
    span_id: Optional[str] = None
    parent_span_id: Optional[str] = None
    perf_metrics: Optional[Dict[str, Any]] = None
    
    # Additional context
    ext: Dict[str, Any] = field(default_factory=dict)


class TraceService:
    """
    Service for reporting model call trace to external endpoints.
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
        Initialize the trace service.
        
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
            headers = {}
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
            headers = {}
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
    
    async def areport(self, event: TraceEvent) -> bool:
        """
        Report a model call event to the trace endpoint.
        
        Args:
            event: The trace event to report
            
        Returns:
            True if the report was successfully sent, False otherwise
        """
        if not self.enabled:
            logger.debug("Trace reporting is disabled")
            return True
            
        with _tracer.start_as_current_span("trace.report"):
            client = await self._get_client()
            
            # Convert event to serializable dict and decode unicode content for better readability
            # Restore original URLs in llm_request_body for trace reporting
            llm_request_body_for_trace = decode_unicode_in_data(event.llm_request_body)
            if llm_request_body_for_trace and isinstance(llm_request_body_for_trace, dict):
                # Restore original URLs in call_llm_params.messages if exists
                if "call_llm_params" in llm_request_body_for_trace and isinstance(llm_request_body_for_trace["call_llm_params"], dict):
                    call_llm_params = llm_request_body_for_trace["call_llm_params"]
                    if "messages" in call_llm_params:
                        call_llm_params["messages"] = restore_original_urls_in_messages(call_llm_params["messages"])
                        logger.debug("Restored original URLs for trace reporting in call_llm_params.messages")
            
            payload = {
                "template_name": event.template_name,
                "template_id": event.template_id,
                "template_version": event.template_version,
                "variant": event.variant,
                "model": event.model,
                "messages_template": decode_unicode_in_data(event.messages_template),
                "variables": decode_unicode_in_data(event.variables),
                "query": decode_unicode_content(event.query),
                "llm_request_body": mask_sensitive_fields(llm_request_body_for_trace),
                "llm_response_body": decode_unicode_in_data(event.llm_response_body),
                "request_id": event.request_id,
                "app_id": event.app_id,
                "user_id": event.user_id,
                "timestamp": event.timestamp,
                "conversation_id": event.conversation_id,
                "token_usage": event.token_usage,
                "error": decode_unicode_content(event.error) if event.error else None,
                "source": event.source,
                "span_id": event.span_id,
                "parent_span_id": event.parent_span_id,
                "ext": decode_unicode_in_data(event.ext),
                "perf_metrics": event.perf_metrics
            }
            url = self.endpoint_url + "/trace/llm-message/dump"

            # Try to send the report with retries
            for attempt in range(self.max_retries):
                try:
                    response = await client.post(
                        url,
                        json=payload,
                        timeout=self.timeout
                    )
                    
                    if response.status_code < 400:
                        logger.debug(f"Trace report sent successfully: {response.status_code}")
                        return True
                    
                    logger.warning(
                        f"Trace report failed (attempt {attempt+1}/{self.max_retries}): "
                        f"Status {response.status_code}, {response.text}"
                    )
                    
                    # Wait before retry (exponential backoff)
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(2 ** attempt * 0.1)
                        
                except Exception as e:
                    logger.warning(
                        f"Trace report exception (attempt {attempt+1}/{self.max_retries}): {str(e)}"
                    )
                    
                    # Wait before retry (exponential backoff)
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(2 ** attempt * 0.1)
            
            # All attempts failed
            logger.error(f"Trace report failed after {self.max_retries} attempts")
            return False
    
    def report(self, event: TraceEvent) -> bool:
        """
        Synchronous version: Report a model call event to the trace endpoint.
        
        Args:
            event: The trace event to report
            
        Returns:
            True if the report was successfully sent, False otherwise
        """
        if not self.enabled:
            logger.debug("Trace reporting is disabled")
            return True
            
        client = self._get_sync_client()
        
        # Convert event to serializable dict and decode unicode content for better readability
        # Restore original URLs in llm_request_body for trace reporting
        llm_request_body_for_trace = decode_unicode_in_data(event.llm_request_body)
        if llm_request_body_for_trace and isinstance(llm_request_body_for_trace, dict):
            # Restore original URLs in call_llm_params.messages if exists
            if "call_llm_params" in llm_request_body_for_trace and isinstance(llm_request_body_for_trace["call_llm_params"], dict):
                call_llm_params = llm_request_body_for_trace["call_llm_params"]
                if "messages" in call_llm_params:
                    call_llm_params["messages"] = restore_original_urls_in_messages(call_llm_params["messages"])
                    logger.debug("Restored original URLs for trace reporting in call_llm_params.messages")
        
        payload = {
            "template_name": event.template_name,
            "template_id": event.template_id,
            "template_version": event.template_version,
            "variant": event.variant,
            "model": event.model,
            "messages_template": decode_unicode_in_data(event.messages_template),
            "variables": decode_unicode_in_data(event.variables),
            "query": decode_unicode_content(event.query),
            "llm_request_body": mask_sensitive_fields(llm_request_body_for_trace),
            "llm_response_body": decode_unicode_in_data(event.llm_response_body),
            "request_id": event.request_id,
            "app_id": event.app_id,
            "user_id": event.user_id,
            "timestamp": event.timestamp,
            "conversation_id": event.conversation_id,
            "token_usage": event.token_usage,
            "error": decode_unicode_content(event.error) if event.error else None,
            "source": event.source,
            "span_id": event.span_id,
            "parent_span_id": event.parent_span_id,
            "ext": decode_unicode_in_data(event.ext),
            "perf_metrics": event.perf_metrics
        }
        url = self.endpoint_url + "/trace/llm-message/dump"

        # Try to send the report with retries
        for attempt in range(self.max_retries):
            try:
                response = client.post(
                    url,
                    json=payload,
                    timeout=self.timeout
            )
                
                if response.status_code < 400:
                    logger.debug(f"Trace report sent successfully: {response.status_code}")
                    return True
                
                logger.warning(
                    f"Trace report failed (attempt {attempt+1}/{self.max_retries}): "
                    f"Status {response.status_code}, {response.text}, payload: {payload}"
                )
                
                # Wait before retry (exponential backoff)
                if attempt < self.max_retries - 1:
                    import time
                    time.sleep(2 ** attempt * 0.1)
                    
            except Exception as e:
                logger.warning(
                    f"Trace report exception (attempt {attempt+1}/{self.max_retries}): {str(e)}, payload: {payload}"
                )
                
                # Wait before retry (exponential backoff)
                if attempt < self.max_retries - 1:
                    import time
                    time.sleep(2 ** attempt * 0.1)
        
        # All attempts failed
        logger.error(f"Trace report failed after {self.max_retries} attempts")
        return False
