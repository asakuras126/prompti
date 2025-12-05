"""Wordlist anonymization hook implementation."""

from typing import Dict, Any, Union, List
from collections.abc import Iterator, AsyncIterator
from .base import BeforeRunHook, AfterRunHook, HookResult
from ..model_client import RunParams
from ..message import Message, ModelResponse, StreamingModelResponse, StreamingChoice


class WordlistAnonymizationHook(BeforeRunHook, AfterRunHook):
    """Wordlist-based anonymization hook for sensitive data anonymization and deanonymization.
    
    Supports wordlist matching anonymization, replacing specified sensitive terms with anonymized terms.
    
    Workflow:
    1. BeforeRun stage: Anonymize input messages and save mapping
    2. AfterRun stage: Deanonymize model output to restore original content
    3. For streaming responses, supports real-time deanonymization
    """
    
    def __init__(self, wordlist: Dict[str, str] = None):
        """Initialize wordlist anonymization hook.
        
        Args:
            wordlist: Anonymization wordlist, format: {source_string: replacement_string}
        """
        self.wordlist = wordlist or {}
        
        # Streaming response state management
        self._streaming_buffer = ""  # Single buffer for current session
        self._tool_call_buffers = {}  # Tool call specific buffers: {tool_call_id: buffer}
        self._reverse_mapping = {}  # Reverse mapping cache
        self._current_tool_call_ids = {}  # Track current active tool call IDs by index: {index: id}
    
    def _get_reverse_mapping(self) -> Dict[str, str]:
        """获取反向映射关系，优先使用缓存的映射"""
        return {replacement: original for original, replacement in self.wordlist.items()}

    # ========== BeforeRun 接口方法 ==========
    
    def process(self, params: RunParams) -> HookResult:
        """Synchronously process run parameters for anonymization."""
        processed_messages, mapping = self._process_messages(params.messages)
        
        # 创建新的RunParams对象
        new_params = params.model_copy()
        new_params.messages = processed_messages
        
        # Only save anonymization mapping for subsequent deanonymization
        metadata = {'anonymization_mapping': mapping}
        
        return HookResult(data=new_params, metadata=metadata)
    
    async def aprocess(self, params: RunParams) -> HookResult:
        """Asynchronously process run parameters for anonymization."""
        # Anonymization is CPU-intensive, call synchronous method directly
        return self.process(params)

    # ========== AfterRun 接口方法 ==========
    

    # ========== 私有辅助方法 ==========
    
    def _anonymize_text(self, text: str) -> tuple[str, Dict[str, str]]:
        """Perform wordlist anonymization on text."""
        mapping = {}
        result = text
        
        if not self.wordlist:
            return result, mapping
        
        # Sort by source string length in descending order to ensure long matches take priority
        sorted_wordlist = sorted(self.wordlist.items(), key=lambda x: len(x[0]), reverse=True)
        
        for source_text, replacement in sorted_wordlist:
            if source_text in result:
                # Check if already replaced by other replacement strings
                already_replaced = False
                for existing_replacement in mapping.keys():
                    if source_text in existing_replacement:
                        already_replaced = True
                        break
                
                if not already_replaced:
                    result = result.replace(source_text, replacement)
                    # Record reverse mapping for recovery
                    mapping[replacement] = source_text
        
        return result, mapping

    def _deanonymize_text(self, text: str, mapping: Dict[str, str]) -> str:
        """Recover original data from anonymized text."""
        result = text
        for replacement, original in mapping.items():
            result = result.replace(replacement, original)
        return result

    def _anonymize_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], Dict[str, str]]:
        """Anonymize tool_calls data."""
        if not tool_calls:
            return tool_calls, {}
        
        anonymized_tool_calls = []
        combined_mapping = {}
        
        for tool_call in tool_calls:
            new_tool_call = tool_call.copy()
            
            # Process function arguments if present
            if 'function' in tool_call and 'arguments' in tool_call['function']:
                arguments = tool_call['function']['arguments']
                if isinstance(arguments, str) and arguments:
                    anonymized_args, mapping = self._anonymize_text(arguments)
                    new_tool_call['function'] = tool_call['function'].copy()
                    new_tool_call['function']['arguments'] = anonymized_args
                    combined_mapping.update(mapping)
            
            anonymized_tool_calls.append(new_tool_call)
        
        return anonymized_tool_calls, combined_mapping

    def _deanonymize_tool_calls(self, tool_calls: List[Dict[str, Any]], mapping: Dict[str, str]) -> List[Dict[str, Any]]:
        """De-anonymize tool_calls data."""
        if not tool_calls or not mapping:
            return tool_calls
        
        deanonymized_tool_calls = []
        
        for tool_call in tool_calls:
            new_tool_call = tool_call.copy()
            
            # Process function arguments if present
            if 'function' in tool_call and 'arguments' in tool_call['function']:
                arguments = tool_call['function']['arguments']
                if isinstance(arguments, str):
                    deanonymized_args = self._deanonymize_text(arguments, mapping)
                    new_tool_call['function'] = tool_call['function'].copy()
                    new_tool_call['function']['arguments'] = deanonymized_args
            
            deanonymized_tool_calls.append(new_tool_call)
        
        return deanonymized_tool_calls

    def _process_messages(self, messages: list[Message]) -> tuple[list[Message], Dict[str, str]]:
        """处理消息列表进行脱敏。"""
        processed_messages = []
        combined_mapping = {}
        
        for msg in messages:
            new_msg = msg.model_copy()
            
            # Process content
            if msg.content:
                if isinstance(msg.content, str):
                    anonymized_content, mapping = self._anonymize_text(msg.content)
                    new_msg.content = anonymized_content
                    combined_mapping.update(mapping)
                elif isinstance(msg.content, list):
                    # 处理多模态内容
                    new_content = []
                    for item in msg.content:
                        if isinstance(item, dict) and item.get('type') == 'text':
                            text_content = item.get('text', '')
                            anonymized_text, mapping = self._anonymize_text(text_content)
                            new_item = item.copy()
                            new_item['text'] = anonymized_text
                            new_content.append(new_item)
                            combined_mapping.update(mapping)
                        else:
                            new_content.append(item)
                    new_msg.content = new_content
            
            # Process tool_calls
            if msg.tool_calls:
                anonymized_tool_calls, tool_mapping = self._anonymize_tool_calls(msg.tool_calls)
                new_msg.tool_calls = anonymized_tool_calls
                combined_mapping.update(tool_mapping)
            
            processed_messages.append(new_msg)
        
        return processed_messages, combined_mapping

    def _recover_response_content(self, response: Union[ModelResponse, StreamingModelResponse], 
                                mapping: Dict[str, str]) -> Union[ModelResponse, StreamingModelResponse]:
        """恢复响应内容（用于非流式响应）。"""
        new_response = response.model_copy()

        # 处理choices字段
        if hasattr(new_response, 'choices') and new_response.choices:
            for choice in new_response.choices:
                if hasattr(choice, 'message') and choice.message:
                    # De-anonymize content
                    if choice.message.content:
                        choice.message.content = self._deanonymize_text(choice.message.content, mapping)
                    
                    # De-anonymize tool_calls
                    if choice.message.tool_calls:
                        choice.message.tool_calls = self._deanonymize_tool_calls(choice.message.tool_calls, mapping)
        
        return new_response

    def _process_streaming_chunk(self, response: StreamingModelResponse, 
                               reverse_mapping: Dict[str, str], is_final: bool = False) -> StreamingModelResponse:
        """处理流式响应chunk的实时反脱敏。"""
        new_response = response.model_copy()
        
        # 处理choices字段中的delta内容
        if hasattr(new_response, 'choices') and new_response.choices:
            for choice in new_response.choices:
                if hasattr(choice, 'delta') and choice.delta:
                    # 处理content
                    if choice.delta.content or is_final:
                        # 实时处理当前chunk，is_final时即使content为空也要处理buffer
                        recovered_chunk = self._process_chunk_with_buffer(
                            choice.delta.content or "", reverse_mapping, is_final
                        )
                        choice.delta.content = recovered_chunk
                    
                    # 处理tool_calls (streaming中的tool_calls通常包含部分arguments)
                    if choice.delta.tool_calls:
                        # 对于流式tool_calls，需要逐个处理arguments字段
                        processed_tool_calls = []
                        for tool_call in choice.delta.tool_calls:
                            new_tool_call = tool_call.copy()
                            
                            # 获取tool call的index
                            tool_call_index = tool_call.get('index', 0)
                            
                            # 更新当前活跃的tool call ID（基于index）
                            if tool_call.get('id') and tool_call['id'].strip():
                                self._current_tool_call_ids[tool_call_index] = tool_call['id']
                            
                            if 'function' in tool_call and 'arguments' in tool_call['function']:
                                args = tool_call['function']['arguments']
                                if isinstance(args, str) and (args or is_final):
                                    # 使用当前index对应的tool call ID，如果没有则使用index作为默认ID
                                    tool_call_id = self._current_tool_call_ids.get(tool_call_index, f'tool_call_{tool_call_index}')
                                    recovered_args = self._process_tool_call_chunk_with_buffer(
                                        args or "", reverse_mapping, tool_call_id, is_final
                                    )
                                    if recovered_args:  # 只有当有输出内容时才设置
                                        new_tool_call['function'] = tool_call['function'].copy()
                                        new_tool_call['function']['arguments'] = recovered_args
                            processed_tool_calls.append(new_tool_call)
                        choice.delta.tool_calls = processed_tool_calls
        return new_response

    def _process_chunk_with_buffer(self, chunk: str, reverse_mapping: Dict[str, str], is_final: bool = False) -> str:
        """使用缓冲区实时处理流式响应块。
        
        Args:
            chunk: 当前输入的文本块
            reverse_mapping: 反脱敏映射字典
            is_final: 是否为最终处理，True时会输出缓冲区中的所有剩余内容
        """
        if not reverse_mapping:
            return chunk or ""
        
        # 将新块添加到缓冲区
        if chunk:
            self._streaming_buffer += chunk
        
        output = ""
        
        # 按照替换字符串长度排序，优先处理长的替换字符串
        sorted_replacements = sorted(reverse_mapping.keys(), key=len, reverse=True)
        
        while True:
            # 查找所有替换字符串在缓冲区中的位置
            found_matches = []
            for replacement in sorted_replacements:
                pos = self._streaming_buffer.find(replacement)
                if pos != -1:
                    found_matches.append((pos, replacement))
            
            if not found_matches:
                break  # 没有更多完整替换字符串可处理
            
            # 按位置排序，处理最早出现的替换字符串
            found_matches.sort(key=lambda x: x[0])
            pos, found_replacement = found_matches[0]
            
            # 输出替换字符串前的内容
            output += self._streaming_buffer[:pos]
            # 输出恢复的原始内容
            output += reverse_mapping[found_replacement]
            # 保留替换字符串后的内容
            self._streaming_buffer = self._streaming_buffer[pos + len(found_replacement):]
        
        # 如果不是最终处理，应用缓冲区策略
        if not is_final and self._streaming_buffer:
            # 获取最长替换字符串的长度
            max_replacement_len = max(len(r) for r in sorted_replacements) if sorted_replacements else 0
            
            if max_replacement_len > 0:
                # 缓冲区只保留最长key长度-1的内容
                max_buffer_len = max_replacement_len - 1
                
                if len(self._streaming_buffer) > max_buffer_len:
                    # 输出超出部分，只保留必要的缓冲区内容
                    safe_output_len = len(self._streaming_buffer) - max_buffer_len
                    output += self._streaming_buffer[:safe_output_len]
                    self._streaming_buffer = self._streaming_buffer[safe_output_len:]
        
        # 如果是最终处理，输出缓冲区中的所有剩余内容
        if is_final and self._streaming_buffer:
            # 剩余内容都是无法匹配的，直接输出
            output += self._streaming_buffer
            self._streaming_buffer = ""
        
        return output

    def _process_tool_call_chunk_with_buffer(self, chunk: str, reverse_mapping: Dict[str, str], tool_call_id: str, is_final: bool = False) -> str:
        """使用专门的tool call缓冲区实时处理流式tool call参数。
        
        Args:
            chunk: 当前输入的文本块
            reverse_mapping: 反脱敏映射字典
            tool_call_id: tool call的唯一标识符
            is_final: 是否为最终处理，True时会输出缓冲区中的所有剩余内容
        """
        if not reverse_mapping:
            return chunk or ""
        
        # 初始化tool call专用缓冲区
        if tool_call_id not in self._tool_call_buffers:
            self._tool_call_buffers[tool_call_id] = ""
        
        # 将新块添加到对应的tool call缓冲区
        if chunk:
            self._tool_call_buffers[tool_call_id] += chunk
        
        output = ""
        
        # 按照替换字符串长度排序，优先处理长的替换字符串
        sorted_replacements = sorted(reverse_mapping.keys(), key=len, reverse=True)
        
        while True:
            # 查找所有替换字符串在缓冲区中的位置
            found_matches = []
            for replacement in sorted_replacements:
                pos = self._tool_call_buffers[tool_call_id].find(replacement)
                if pos != -1:
                    found_matches.append((pos, replacement))
            
            if not found_matches:
                break  # 没有更多完整替换字符串可处理
            
            # 按位置排序，处理最早出现的替换字符串
            found_matches.sort(key=lambda x: x[0])
            pos, found_replacement = found_matches[0]
            
            # 输出替换字符串前的内容
            output += self._tool_call_buffers[tool_call_id][:pos]
            # 输出恢复的原始内容
            output += reverse_mapping[found_replacement]
            # 保留替换字符串后的内容
            self._tool_call_buffers[tool_call_id] = self._tool_call_buffers[tool_call_id][pos + len(found_replacement):]
        
        # 如果不是最终处理，应用缓冲区策略
        if not is_final and self._tool_call_buffers[tool_call_id]:
            # 获取最长替换字符串的长度
            max_replacement_len = max(len(r) for r in sorted_replacements) if sorted_replacements else 0
            
            if max_replacement_len > 0:
                # 缓冲区只保留最长key长度-1的内容
                max_buffer_len = max_replacement_len - 1
                
                if len(self._tool_call_buffers[tool_call_id]) > max_buffer_len:
                    # 输出超出部分，只保留必要的缓冲区内容
                    safe_output_len = len(self._tool_call_buffers[tool_call_id]) - max_buffer_len
                    output += self._tool_call_buffers[tool_call_id][:safe_output_len]
                    self._tool_call_buffers[tool_call_id] = self._tool_call_buffers[tool_call_id][safe_output_len:]
        
        # 如果是最终处理，输出缓冲区中的所有剩余内容
        if is_final and self._tool_call_buffers[tool_call_id]:
            # 剩余内容都是无法匹配的，直接输出
            output += self._tool_call_buffers[tool_call_id]
            self._tool_call_buffers[tool_call_id] = ""
        
        return output


    # ========== 新的流式会话接口实现 ==========
    
    def start_streaming_session(self, session_id: str, hook_metadata: Dict[str, Any]) -> None:
        """开始流式处理会话"""
        # 重置会话状态
        self._streaming_buffer = ""
        self._tool_call_buffers = {}
        self._current_tool_call_ids = {}
        
        # 缓存反脱敏映射
        anonymization_mapping = hook_metadata.get('anonymization_mapping', {})
        self._reverse_mapping = {v: k for k, v in anonymization_mapping.items()}
    
    async def astart_streaming_session(self, session_id: str, hook_metadata: Dict[str, Any]) -> None:
        """异步开始流式处理会话"""
        self.start_streaming_session(session_id, hook_metadata)
    
    def process_streaming_chunk(self, chunk: "StreamingModelResponse", session_id: str, is_final: bool = False) -> Iterator[HookResult]:
        """处理流式chunk，支持中间处理和最终处理"""
        if not self.wordlist:
            yield HookResult(data=chunk)
            return
        
        # 对于response为空的判断，创建空对象继续处理
        if chunk is None:
            chunk = StreamingModelResponse(
                id="", 
                object="chat.completion.chunk", 
                created=0, 
                model="", 
                choices=[StreamingChoice(
                    index=0,
                    delta=Message(content="", role="assistant"),
                    finish_reason=None
                )]
            )
        
        # 使用统一的映射获取逻辑（优先使用缓存的会话映射）
        reverse_mapping = self._get_reverse_mapping()
        
        # 使用统一的处理逻辑，is_final参数会传递给底层缓冲区方法
        new_response = self._process_streaming_chunk(chunk, reverse_mapping, is_final)
        
        # 如果是最终处理，清理会话状态
        if is_final:
            self._streaming_buffer = ""
            self._tool_call_buffers = {}
            self._current_tool_call_ids = {}
            self._reverse_mapping = {}
        
        yield HookResult(data=new_response)
    
    async def aprocess_streaming_chunk(self, chunk: "StreamingModelResponse", session_id: str, is_final: bool = False) -> AsyncIterator[HookResult]:
        """异步处理流式chunk，支持中间处理和最终处理"""
        # 反脱敏是CPU密集型操作，直接调用同步方法
        for result in self.process_streaming_chunk(chunk, session_id, is_final):
            yield result
    
    def process_non_streaming_response(self, response: "ModelResponse", hook_metadata: Dict[str, Any]) -> HookResult:
        """处理非流式响应数据"""
        if not self.wordlist:
            return HookResult(data=response)
        
        # 对于response为空的判断
        if response is None:
            return HookResult(data=None)
        
        # 使用统一的映射获取逻辑
        reverse_mapping = self._get_reverse_mapping()
        
        # 处理非流式响应
        recovered_response = self._recover_response_content(response, reverse_mapping)
        
        return HookResult(data=recovered_response)
    
    async def aprocess_non_streaming_response(self, response: "ModelResponse", hook_metadata: Dict[str, Any]) -> HookResult:
        """异步处理非流式响应数据"""
        # 反脱敏是CPU密集型操作，直接调用同步方法
        return self.process_non_streaming_response(response, hook_metadata)