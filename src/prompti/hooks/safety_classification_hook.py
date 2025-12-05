"""安全分类检查钩子实现。"""

import re
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from typing import Dict, Any, Union, Optional
from collections.abc import Iterator, AsyncIterator
from dataclasses import dataclass
from collections import deque
from .base import AfterRunHook, HookResult
from prompti.message import ModelResponse, StreamingModelResponse, StreamingChoice, Message
from ..logger import get_logger

logger = get_logger(__name__)

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False


@dataclass
class SafetyCheckTask:
    """异步安全检查任务"""
    sequence_id: int
    content: str
    future: asyncio.Future
    submitted_time: float


@dataclass
class SyncSafetyCheckTask:
    """同步安全检查任务"""
    sequence_id: int
    content: str
    future: Future
    submitted_time: float


class SafetyClassificationException(Exception):
    """安全分类检查异常"""
    def __init__(self, blocked_message: str):
        self.blocked_message = blocked_message
        super().__init__(f"Content blocked: {blocked_message}")


class SafetyClassificationHook(AfterRunHook):
    """安全分类检查钩子，用于检查模型输出的安全性。
    
    该钩子会缓存完整的句子，然后调用安全分类API进行检查。
    如果检测到不安全内容，会中断请求并返回预设的安全提示信息。
    
    安全检查只使用当前查询内容，不使用历史句子作为上下文。
    
    对于流式响应，会对每个完整句子进行实时检查；
    对于非流式响应，会对完整内容进行一次性检查。
    """
    
    def __init__(self, 
                 api_url: str = "https://aisecurity.baidu-int.com/output_safety_multi_classification_service",
                 blocked_message: str = "",
                 text_terminators: list[str] = None,
                 timeout: int = 5,
                 max_retries: int = 3,
                 retry_delay: float = 1.0,
                 max_concurrent_checks: int = 10):
        """初始化安全分类钩子。
        
        Args:
            api_url: 安全分类API的URL
            blocked_message: 当内容被阻止时返回的消息
            text_terminators: 文本终止符列表，用于分割句子和段落
            timeout: API请求超时时间（秒）
            max_retries: 最大重试次数
            retry_delay: 重试间隔时间（秒）
            max_concurrent_checks: 最大并发安全检查数量
        """
        self.api_url = api_url
        self.blocked_message = blocked_message
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_concurrent_checks = max_concurrent_checks
        
        # 线程池用于Hook内部并发API调用
        self._thread_pool: Optional[ThreadPoolExecutor] = None
        
        # 会话状态管理
        self._sessions: Dict[str, Dict[str, Any]] = {}  # 会话状态存储
        
        # 默认的文本终止符
        if text_terminators is None:
            self.text_terminators = [
                # 行终止符
                "\n", "\r\n", "\r",
                # 句子终止符
                "；", "，", "!", "?", "。", "！", "？"
            ]
        else:
            self.text_terminators = text_terminators

    # ========== 线程池管理 ==========
    
    def _ensure_thread_pool(self):
        """确保线程池已初始化"""
        if self._thread_pool is None:
            self._thread_pool = ThreadPoolExecutor(max_workers=self.max_concurrent_checks)

    # ========== 主要接口方法 ==========
    
    def __del__(self):
        """析构函数，清理线程池资源"""
        if hasattr(self, '_thread_pool') and self._thread_pool:
            try:
                self._thread_pool.shutdown(wait=False)
            except Exception:
                pass  # 忽略清理时的异常
    
    def shutdown_thread_pool(self):
        """手动关闭线程池"""
        if self._thread_pool:
            self._thread_pool.shutdown(wait=True)
            self._thread_pool = None

    # ========== 新的流式会话接口实现 ==========
    
    def start_streaming_session(self, session_id: str, hook_metadata: Dict[str, Any]) -> None:
        """开始流式处理会话"""
        self._sessions[session_id] = {
            'sentence_buffer': "",  # 缓存当前正在构建的句子
            'sentence_history': [],  # 缓存之前的N个完整句子
            'original_finish_reason': None,  # 保存原始的finish_reason
            'last_response_metadata': {},  # 保存最后一个响应的元数据
            'usage_info': None,  # 保存usage信息
            'hook_metadata': hook_metadata  # 保存hook元数据
        }
        logger.debug(f"SafetyClassificationHook: Started session {session_id}")
    
    async def astart_streaming_session(self, session_id: str, hook_metadata: Dict[str, Any]) -> None:
        """异步开始流式处理会话"""
        self.start_streaming_session(session_id, hook_metadata)
    
    def process_streaming_chunk(self, chunk: "StreamingModelResponse", session_id: str, is_final: bool = False) -> Iterator[HookResult]:
        """处理流式chunk，支持中间处理和最终处理
        
        关键特性：Hook内部并发处理
        1. 立即接收新chunk，不等待之前的安全API响应
        2. 并发提交多个安全检查任务
        3. 按顺序输出已完成的安全内容
        4. is_final=True时，处理缓冲区剩余内容并清理会话状态
        """
        if session_id not in self._sessions:
            logger.warning(f"SafetyClassificationHook: Session {session_id} not found, starting new session")
            self.start_streaming_session(session_id, {})
        
        session_state = self._sessions[session_id]
        content = self._extract_content_from_response(chunk)
        
        if not content and not is_final:
            self._save_response_metadata_to_session(chunk, session_state)
            yield HookResult(data=chunk)
            return
        
        # 处理传入的内容
        if content:
            # 立即提取句子并提交并发安全检查
            sentences, _ = self._extract_complete_sentences_for_session(content, session_state)
            self._save_response_metadata_to_session(chunk, session_state)
            
            # 立即提交所有句子的安全检查任务（非阻塞）
            for sentence in sentences:
                # 将当前句子添加到历史记录中
                self._add_to_sentence_history(sentence, session_state)
                # 构建包含历史上下文的查询内容
                contextual_query = self._build_contextual_query(sentence, session_state)
                self._submit_concurrent_safety_check(contextual_query, sentence, session_state)
            
            # 立即返回当前已完成的安全检查结果
            yield from self._collect_ready_safety_outputs(session_state, chunk)
        
        # 如果是最终处理，处理缓冲区剩余内容
        if is_final:
            # 等待所有剩余的并发安全检查完成
            self._wait_for_all_concurrent_checks(session_state)
            
            # 收集所有剩余的安全内容
            remaining_outputs = []
            if 'concurrent_pending_tasks' in session_state:
                # 创建一个临时chunk用于收集剩余输出
                temp_chunk = type('TempChunk', (), {
                    'model_copy': lambda self, deep=True: self,
                    'choices': [type('Choice', (), {'delta': type('Delta', (), {'content': ''})})()]
                })()
                
                for hook_result in self._collect_ready_safety_outputs(session_state, temp_chunk):
                    remaining_outputs.append(hook_result.data.choices[0].delta.content if hasattr(hook_result.data, 'choices') else "")
            
            # 提取最终内容（包括缓冲区中的不完整句子）
            final_content = self._extract_final_content_for_session(session_state)
            
            # 添加剩余的输出内容
            if remaining_outputs:
                final_content = "".join(remaining_outputs) + final_content
            
            # 对合并后的最终内容进行安全检查
            if final_content.strip():
                # 将最终内容添加到历史记录并构建上下文查询
                self._add_to_sentence_history(final_content.strip(), session_state)
                contextual_query = self._build_contextual_query(final_content.strip(), session_state)
                if not self._call_safety_api(contextual_query):
                    logger.error(f"SafetyClassificationHook: Final content unsafe in session {session_id}: {final_content}")
                    final_content = ""  # 不安全的内容不输出
            
            # 生成最终响应
            if final_content:
                response = self._create_final_response_for_session(final_content, session_state)
                yield HookResult(data=response)
            elif chunk:
                # 如果没有最终内容但有传入响应，需要创建一个空内容的响应
                response = self._update_response_content(chunk, "")
                yield HookResult(data=response)
            
            # 清理会话状态
            del self._sessions[session_id]
            logger.debug(f"SafetyClassificationHook: Finished session {session_id}")
    
    async def aprocess_streaming_chunk(self, chunk: "StreamingModelResponse", session_id: str, is_final: bool = False) -> AsyncIterator[HookResult]:
        """异步处理流式chunk，支持中间处理和最终处理"""
        if session_id not in self._sessions:
            logger.warning(f"SafetyClassificationHook: Session {session_id} not found, starting new session")
            await self.astart_streaming_session(session_id, {})
        
        session_state = self._sessions[session_id]
        content = self._extract_content_from_response(chunk)
        
        if not content and not is_final:
            self._save_response_metadata_to_session(chunk, session_state)
            yield HookResult(data=chunk)
            return
        
        # 处理传入的内容
        if content:
            sentences, _ = self._extract_complete_sentences_for_session(content, session_state)
            self._save_response_metadata_to_session(chunk, session_state)
            
            # 异步检查每个完整句子，可能产生多个输出chunk
            for sentence in sentences:
                # 将当前句子添加到历史记录中
                self._add_to_sentence_history(sentence, session_state)
                # 构建包含历史上下文的查询内容
                contextual_query = self._build_contextual_query(sentence, session_state)
                if await self._call_safety_api_async(contextual_query):
                    logger.debug(f"SafetyClassificationHook: Async sentence safe in session {session_id}: {sentence[:20]}...")
                    # 为每个安全的句子创建单独的输出chunk
                    output_chunk = self._update_response_content(chunk, sentence, clear_finish_reason=True)
                    yield HookResult(data=output_chunk)
                else:
                    logger.error(f"SafetyClassificationHook: Async sentence unsafe in session {session_id}: {sentence}")
                    # 不安全的句子不输出
        
        # 如果是最终处理，处理缓冲区剩余内容
        if is_final:
            final_content = self._extract_final_content_for_session(session_state)
            
            if final_content.strip():
                # 将最终内容添加到历史记录并构建上下文查询
                self._add_to_sentence_history(final_content.strip(), session_state)
                contextual_query = self._build_contextual_query(final_content.strip(), session_state)
                if not await self._call_safety_api_async(contextual_query):
                    logger.error(f"SafetyClassificationHook: Async final content unsafe in session {session_id}: {final_content}")
                    final_content = ""
            
            # 生成最终响应
            if final_content:
                response = self._create_final_response_for_session(final_content, session_state)
                yield HookResult(data=response)
            elif chunk:
                # 如果没有最终内容但有传入响应，需要创建一个空内容的响应
                response = self._update_response_content(chunk, "")
                yield HookResult(data=response)
            
            # 清理会话状态
            del self._sessions[session_id]
            logger.debug(f"SafetyClassificationHook: Async finished session {session_id}")

    # ========== 非流式响应处理方法 ==========
    
    def process_non_streaming_response(self, response: "ModelResponse", hook_metadata: Dict[str, Any]) -> HookResult:
        """处理非流式响应的安全检查。
        
        Args:
            response: 非流式模型响应对象
            hook_metadata: Hook元数据
            
        Returns:
            HookResult: 处理结果，包含安全检查后的响应
            
        Raises:
            SafetyClassificationException: 当内容被安全检查阻止时
        """
        # 提取响应内容
        content = self._extract_content_from_response(response)
        
        if not content or not content.strip():
            # 空内容直接返回
            logger.debug(f"SafetyClassificationHook: Empty content in non-streaming response, skipping safety check")
            return HookResult(data=response)
        
        logger.debug(f"SafetyClassificationHook: Processing non-streaming response, content length: {len(content)}")
        
        # 对完整内容进行安全检查
        if self._call_safety_api(content):
            logger.debug(f"SafetyClassificationHook: Non-streaming content is safe")
            return HookResult(data=response)
        else:
            logger.error(f"SafetyClassificationHook: Non-streaming content blocked: {content[:50]}...")
            # 统一抛出安全异常，由Engine的_handle_safety_exception方法处理
            raise SafetyClassificationException(blocked_message="")
    
    async def aprocess_non_streaming_response(self, response: "ModelResponse", hook_metadata: Dict[str, Any]) -> HookResult:
        """异步处理非流式响应的安全检查。
        
        Args:
            response: 非流式模型响应对象
            hook_metadata: Hook元数据
            
        Returns:
            HookResult: 处理结果，包含安全检查后的响应
            
        Raises:
            SafetyClassificationException: 当内容被安全检查阻止时
        """
        # 提取响应内容
        content = self._extract_content_from_response(response)
        
        if not content or not content.strip():
            # 空内容直接返回
            logger.debug(f"SafetyClassificationHook: Empty content in async non-streaming response, skipping safety check")
            return HookResult(data=response)
        
        logger.debug(f"SafetyClassificationHook: Processing async non-streaming response, content length: {len(content)}")
        
        # 对完整内容进行异步安全检查
        if await self._call_safety_api_async(content):
            logger.debug(f"SafetyClassificationHook: Async non-streaming content is safe")
            return HookResult(data=response)
        else:
            logger.error(f"SafetyClassificationHook: Async non-streaming content blocked: {content[:50]}...")
            # 统一抛出安全异常，由Engine的_handle_safety_exception方法处理
            raise SafetyClassificationException(blocked_message="")

    

    # ========== Hook内部并发安全检查方法（会话版本） ==========
    
    def _submit_concurrent_safety_check(self, contextual_query: str, original_sentence: str, session_state: Dict[str, Any]):
        """为会话提交并发安全检查任务（非阻塞），使用上下文查询"""
        self._ensure_thread_pool()
        
        # 获取或初始化会话的并发状态
        if 'concurrent_sequence_counter' not in session_state:
            session_state['concurrent_sequence_counter'] = 0
            session_state['concurrent_pending_tasks'] = {}
            session_state['concurrent_completed_results'] = {}
            session_state['concurrent_output_queue'] = deque()
            session_state['concurrent_next_output_sequence'] = 0
        
        # 背压控制：如果pending检查太多，等待一些完成
        if len(session_state['concurrent_pending_tasks']) >= self.max_concurrent_checks:
            logger.debug(f"SafetyCheck: max concurrent reached, waiting for completion...")
            self._wait_for_any_concurrent_completion(session_state)
        
        # 分配序列号
        sequence_id = session_state['concurrent_sequence_counter']
        session_state['concurrent_sequence_counter'] += 1
        
        # 提交异步任务
        future = self._thread_pool.submit(self._call_safety_api, contextual_query)
        
        # 创建任务记录
        task = SyncSafetyCheckTask(
            sequence_id=sequence_id,
            content=original_sentence,
            future=future,
            submitted_time=time.time()
        )
        
        session_state['concurrent_pending_tasks'][sequence_id] = task
        session_state['concurrent_output_queue'].append((sequence_id, original_sentence))
        
        logger.debug(f"SafetyCheck: submitted concurrent check {sequence_id}: '{original_sentence[:20]}...' (with context)")
    
    def _collect_ready_safety_outputs(self, session_state: Dict[str, Any], chunk: "StreamingModelResponse") -> Iterator[HookResult]:
        """收集会话中已完成的安全检查结果（按顺序输出）"""
        if 'concurrent_pending_tasks' not in session_state:
            return
        
        # 检查已完成的pending任务
        completed_seq_ids = []
        for seq_id, task in list(session_state['concurrent_pending_tasks'].items()):
            if task.future.done():
                try:
                    is_safe = task.future.result()
                    processing_time = time.time() - task.submitted_time
                    
                    session_state['concurrent_completed_results'][seq_id] = is_safe
                    completed_seq_ids.append(seq_id)
                    
                    logger.debug(f"SafetyCheck: completed concurrent check {seq_id}: {'SAFE' if is_safe else 'BLOCKED'} ({processing_time:.3f}s)")
                    
                except Exception as e:
                    logger.error(f"SafetyCheck: concurrent check {seq_id} failed: {e}")
                    # 失败时认为不安全
                    session_state['concurrent_completed_results'][seq_id] = False
                    completed_seq_ids.append(seq_id)
        
        # 清理已完成的pending任务
        for seq_id in completed_seq_ids:
            session_state['concurrent_pending_tasks'].pop(seq_id, None)
        
        # 按顺序输出连续的已完成结果
        while (session_state['concurrent_output_queue'] and 
               session_state['concurrent_output_queue'][0][0] == session_state['concurrent_next_output_sequence'] and
               session_state['concurrent_next_output_sequence'] in session_state['concurrent_completed_results']):
            
            seq_id, sentence = session_state['concurrent_output_queue'].popleft()
            is_safe = session_state['concurrent_completed_results'].pop(seq_id)
            
            if is_safe:
                logger.debug(f"SafetyCheck: output safe sentence {seq_id}: '{sentence}'")
                # 为每个安全的句子创建单独的输出chunk
                output_chunk = self._update_response_content(chunk, sentence, clear_finish_reason=True)
                yield HookResult(data=output_chunk)
            else:
                logger.error(f"SafetyCheck: blocked unsafe sentence {seq_id}: '{sentence}'")
            
            session_state['concurrent_next_output_sequence'] += 1
    
    def _wait_for_any_concurrent_completion(self, session_state: Dict[str, Any]):
        """等待会话中任意一个并发检查完成"""
        if not session_state.get('concurrent_pending_tasks'):
            return
        
        pending_futures = [task.future for task in session_state['concurrent_pending_tasks'].values()]
        
        try:
            # 等待任意一个完成
            for future in as_completed(pending_futures, timeout=self.timeout):
                break  # 只要有一个完成就退出
        except Exception as e:
            logger.warning(f"SafetyCheck: error waiting for concurrent completion: {e}")
    
    def _wait_for_all_concurrent_checks(self, session_state: Dict[str, Any], max_wait_time: float = 10.0):
        """等待会话中所有并发检查完成"""
        if 'concurrent_pending_tasks' not in session_state or not session_state['concurrent_pending_tasks']:
            return
        
        logger.debug(f"SafetyCheck: waiting for {len(session_state['concurrent_pending_tasks'])} concurrent checks...")
        
        pending_futures = [task.future for task in session_state['concurrent_pending_tasks'].values()]
        
        try:
            # 等待所有任务完成
            for future in as_completed(pending_futures, timeout=max_wait_time):
                try:
                    future.result()  # 获取结果，如果有异常会抛出
                except Exception as e:
                    logger.warning(f"SafetyCheck: concurrent task failed: {e}")
        except Exception as e:
            logger.warning(f"SafetyCheck: some concurrent checks failed or timed out: {e}")
        
        logger.debug("SafetyCheck: all concurrent checks completed or timed out")

    # ========== 会话相关的辅助方法 ==========
    
    def _save_response_metadata_to_session(self, response: Union["ModelResponse", "StreamingModelResponse"], session_state: Dict[str, Any]) -> None:
        """保存响应的元数据和usage信息到会话状态"""
        # 保存响应元数据
        if hasattr(response, 'id'):
            session_state['last_response_metadata']['id'] = response.id
        if hasattr(response, 'object'):
            session_state['last_response_metadata']['object'] = response.object
        if hasattr(response, 'created'):
            session_state['last_response_metadata']['created'] = response.created
        if hasattr(response, 'model'):
            session_state['last_response_metadata']['model'] = response.model
        
        # 保存usage信息（通常在最后一个chunk中）
        if hasattr(response, 'usage') and response.usage:
            session_state['usage_info'] = response.usage
        
        # 保存原始的finish_reason（如果存在）
        if hasattr(response, 'choices') and response.choices:
            for choice in response.choices:
                if hasattr(choice, 'finish_reason') and choice.finish_reason:
                    session_state['original_finish_reason'] = choice.finish_reason
                    break
    
    def _extract_complete_sentences_for_session(self, text: str, session_state: Dict[str, Any]) -> tuple[list[str], str]:
        """从文本中提取完整的句子，使用会话状态"""
        if not text:
            return [], session_state['sentence_buffer']
        
        # 将新文本添加到缓冲区
        session_state['sentence_buffer'] += text
        
        # 查找终止符并分割文本
        sentences = []
        remaining_text = session_state['sentence_buffer']
        
        # 按照终止符长度排序，优先匹配较长的终止符（如 \r\n）
        sorted_terminators = sorted(self.text_terminators, key=len, reverse=True)
        
        # 构建正则表达式模式，转义特殊字符
        escaped_terminators = [re.escape(term) for term in sorted_terminators]
        pattern = f'({"|".join(escaped_terminators)})'
        
        # 分割文本，保留分隔符
        parts = re.split(pattern, remaining_text)
        
        current_sentence = ""
        for part in parts:
            if part in self.text_terminators:
                # 找到终止符
                current_sentence += part
                # 保留所有终止符内容，包括换行符等空白字符
                if current_sentence:  # 只要不是完全空字符串就保留
                    sentences.append(current_sentence)
                current_sentence = ""
            else:
                current_sentence += part
        
        # 更新缓冲区为剩余的不完整句子
        session_state['sentence_buffer'] = current_sentence
        
        return sentences, current_sentence
    
    def _extract_final_content_for_session(self, session_state: Dict[str, Any]) -> str:
        """从会话状态中提取最终内容"""
        final_content = session_state['sentence_buffer']
        
        # 清空缓冲区
        session_state['sentence_buffer'] = ""
        session_state['sentence_history'] = []
        
        return final_content
    
    def _create_final_response_for_session(self, final_content: str, session_state: Dict[str, Any]) -> "StreamingModelResponse":
        """为会话创建最终响应对象"""
        delta_message = Message(role="assistant", content=final_content)
        # 使用保存的finish_reason，如果没有则使用"stop"
        finish_reason = session_state['original_finish_reason'] if session_state['original_finish_reason'] else "stop"
        choice = StreamingChoice(index=0, delta=delta_message, finish_reason=finish_reason)
        
        return StreamingModelResponse(
            id=session_state['last_response_metadata'].get('id', f"safety-final-{int(time.time())}"),
            object=session_state['last_response_metadata'].get('object', "chat.completion.chunk"),
            created=session_state['last_response_metadata'].get('created', int(time.time())),
            model=session_state['last_response_metadata'].get('model', "unknown"),
            choices=[choice],
            usage=session_state['usage_info']  # 包含usage信息
        )


    # ========== 私有辅助方法 ==========
    
    def _extract_content_from_response(self, response: Union[ModelResponse, StreamingModelResponse]) -> str:
        """从响应中提取文本内容。"""
        content = ""
        if hasattr(response, 'content') and response.content:
            content = response.content
        elif hasattr(response, 'choices') and response.choices:
            for choice in response.choices:
                if hasattr(choice, 'message') and choice.message and choice.message.content:
                    content += choice.message.content
                elif hasattr(choice, 'delta') and choice.delta and choice.delta.content:
                    content += choice.delta.content
        return content




    
    
    
    

    def _update_response_content(self, response: Union[ModelResponse, StreamingModelResponse], 
                               new_content: str, clear_finish_reason: bool = False) -> Union[ModelResponse, StreamingModelResponse]:
        """更新响应对象的内容。"""
        new_response = response.model_copy(deep=True)
        if hasattr(new_response, 'content'):
            new_response.content = new_content
        elif hasattr(new_response, 'choices') and new_response.choices:
            for choice in new_response.choices:
                if hasattr(choice, 'message') and choice.message:
                    choice.message.content = new_content
                elif hasattr(choice, 'delta') and choice.delta:
                    choice.delta.content = new_content
                    # 根据需要清除finish_reason
                    if clear_finish_reason:
                        choice.finish_reason = None
        return new_response


    def _add_to_sentence_history(self, sentence: str, session_state: Dict[str, Any]) -> None:
        """将句子添加到历史记录中（但不再使用历史上下文）"""
        if 'sentence_history' not in session_state:
            session_state['sentence_history'] = []
        
        # 添加新句子到历史记录（保留功能但不在安全检查中使用）
        session_state['sentence_history'].append(sentence)
        
        logger.debug(f"SafetyCheck: Added sentence to history (not used in safety check), current size: {len(session_state['sentence_history'])}")
    
    def _build_contextual_query(self, current_sentence: str, session_state: Dict[str, Any]) -> str:
        """构建查询内容，只使用当前句子"""
        # 直接返回当前句子，不使用历史上下文
        logger.debug(f"SafetyCheck: Using current query only, length: {len(current_sentence)}")
        return current_sentence

    def _call_safety_api(self, query: str) -> bool:
        """同步调用安全分类API，支持重试。"""
        if not HAS_HTTPX:
            logger.warning("缺少 httpx 库，无法进行安全检查，默认认为内容不安全")
            return False
        if not query.strip():
            # 空白符跳过检查
            return True
        payload = {"query": query}
        last_exception = None
        
        client_params = {"timeout": self.timeout, "verify": False}
        for attempt in range(self.max_retries):
            try:
                with httpx.Client(**client_params) as client:
                    response = client.post(
                        self.api_url,
                        headers={"Content-Type": "application/json"},
                        json=payload
                    )
                    logger.debug(f"call safety classification api, request: {payload}")
                    response.raise_for_status()
                    result = response.json()
                    # 检查返回的label_id是否为3或4（安全正常）
                    # 返回格式: {'data': {'label_id': 4}, 'error_code': 0, 'error_msg': 'Success', 'requestId': '1762739779106722344'}
                    # 分类: {"1": "涉政高危", "2": "涉政负面", "3": "违法色情价值观", "4": "安全正常"}
                    data = result.get("data", {})
                    output_label_id = data.get("label_id", "")
                    if output_label_id and output_label_id not in [3, 4]:
                        logger.error(f"request {payload} failed with label {output_label_id}")
                    return output_label_id in [3, 4]
                    
            except Exception as e:
                last_exception = e
                logger.warning(f"安全分类API调用失败 (尝试 {attempt + 1}/{self.max_retries}): {e}")
                
                # 如果不是最后一次尝试，等待后重试
                if attempt < self.max_retries - 1:
                    logger.info(f"等待 {self.retry_delay} 秒后重试...")
                    time.sleep(self.retry_delay)
        
        # 所有重试都失败，放行，避免影响后续生成
        logger.error(f"安全分类API调用最终失败，已重试 {self.max_retries} 次: {last_exception}")
        return True

    async def _call_safety_api_async(self, query: str) -> bool:
        """异步调用安全分类API，支持重试。"""
        if not HAS_HTTPX:
            logger.warning("缺少 httpx 库，无法进行安全检查，默认认为内容不安全")
            return False
            
        payload = {"query": query}
        last_exception = None
        
        client_params = {"timeout": self.timeout, "verify": False}

        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(**client_params) as client:
                    response = await client.post(
                        self.api_url,
                        headers={"Content-Type": "application/json"},
                        json=payload
                    )
                    response.raise_for_status()
                    
                    result = response.json()
                    
                    # 检查返回的output_label_id是否为"4"（安全正常）
                    data = result.get("data", {})
                    output_label_id = data.get("label_id", "")
                    return output_label_id == 4 or output_label_id == 3
                    
            except Exception as e:
                last_exception = e
                logger.warning(f"安全分类API调用失败 (尝试 {attempt + 1}/{self.max_retries}): {e}")
                
                # 如果不是最后一次尝试，等待后重试
                if attempt < self.max_retries - 1:
                    logger.info(f"等待 {self.retry_delay} 秒后重试...")
                    await asyncio.sleep(self.retry_delay)
        
        # 所有重试都失败，放行，避免影响后续生成
        logger.error(f"安全分类API调用最终失败，已重试 {self.max_retries} 次: {last_exception}")
        return True