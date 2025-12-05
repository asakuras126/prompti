"""Hook基类定义。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Union
from collections.abc import Iterator, AsyncIterator

from ..message import ModelResponse, StreamingModelResponse


class HookResult:
    """Hook处理结果对象，包含处理后的数据和元数据。"""

    def __init__(self, data: Any, metadata: dict[str, Any] | None = None):
        self.data = data
        self.metadata = metadata or {}


class BeforeRunHook(ABC):
    """运行前钩子基类，用于数据预处理（如脱敏）。"""

    @abstractmethod
    def process(self, params: "RunParams") -> HookResult:
        """处理输入参数，返回处理后的参数和元数据。"""
        pass

    @abstractmethod
    async def aprocess(self, params: "RunParams") -> HookResult:
        """异步处理输入参数，返回处理后的参数和元数据。"""
        pass


class AfterRunHook(ABC):
    """运行后钩子基类，用于数据后处理（如反脱敏、安全检查等）。
    
    支持两种模式：
    1. 传统模式：process_non_streaming_response / process_streaming_response
    2. 流式会话模式：start_streaming_session / process_streaming_chunk / finish_streaming_session
    
    流式会话模式支持更复杂的状态管理和Hook内部并发处理。
    """

    def process_non_streaming_response(self, response: "ModelResponse",
                                       hook_metadata: dict[str, Any]) -> HookResult:
        """处理非流式响应数据。
        
        Args:
            response: 非流式模型响应对象
            hook_metadata: Hook元数据
            
        Returns:
            HookResult: 处理结果
        """
        # 默认实现：直接返回原响应
        return HookResult(data=response)

    async def aprocess_non_streaming_response(self, response: "ModelResponse",
                                              hook_metadata: dict[str, Any]) -> HookResult:
        """异步处理非流式响应数据。
        
        Args:
            response: 非流式模型响应对象
            hook_metadata: Hook元数据
            
        Returns:
            HookResult: 处理结果
        """
        # 默认调用同步版本
        return self.process_non_streaming_response(response, hook_metadata)

    # ========== 流式会话接口（新模式，支持Hook内部并发处理） ==========

    def start_streaming_session(self, session_id: str, hook_metadata: dict[str, Any]) -> None:
        """开始流式处理会话。
        
        用于初始化会话状态，为并发处理做准备。
        
        Args:
            session_id: 唯一的会话标识符
            hook_metadata: Hook元数据
        """
        pass  # 默认实现：无操作

    async def astart_streaming_session(self, session_id: str, hook_metadata: dict[str, Any]) -> None:
        """异步开始流式处理会话。
        
        Args:
            session_id: 唯一的会话标识符
            hook_metadata: Hook元数据
        """
        # 默认调用同步版本
        self.start_streaming_session(session_id, hook_metadata)

    def process_streaming_chunk(self, chunk: "StreamingModelResponse", session_id: str, is_final: bool = False) -> Iterator[HookResult]:
        """处理流式chunk，支持中间处理和最终处理。
        
        这是统一的核心接口，支持：
        1. 立即接收新chunk，不等待之前的处理完成
        2. 一个输入chunk可能产生多个输出chunk（通过Iterator）
        3. Hook内部并发处理
        4. 通过is_final参数控制是否为最终处理（清理缓冲区）
        
        Args:
            chunk: 流式模型响应chunk
            session_id: 会话标识符
            is_final: 是否为最终处理，True时会清理缓冲区并输出所有剩余内容
            
        Returns:
            Iterator[HookResult]: 可能产生的多个处理结果的迭代器
        """
        # 默认实现：直接返回原chunk
        yield HookResult(data=chunk)

    async def aprocess_streaming_chunk(self, chunk: "StreamingModelResponse", session_id: str, is_final: bool = False) -> AsyncIterator[HookResult]:
        """异步处理流式chunk，支持中间处理和最终处理。
        
        Args:
            chunk: 流式模型响应chunk
            session_id: 会话标识符
            is_final: 是否为最终处理，True时会清理缓冲区并输出所有剩余内容
            
        Returns:
            AsyncIterator[HookResult]: 可能产生的多个处理结果的异步迭代器
        """
        # 默认调用同步版本并转换为异步迭代器
        for result in self.process_streaming_chunk(chunk, session_id, is_final):
            yield result

