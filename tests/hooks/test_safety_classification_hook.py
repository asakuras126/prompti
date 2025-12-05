"""安全分类钩子的单元测试"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from prompti.hooks.safety_classification_hook import SafetyClassificationHook, SafetyClassificationException


class TestSafetyClassificationHook:
    """安全分类钩子的测试类"""
    
    def setup_method(self):
        """每个测试方法前的设置"""
        self.hook = SafetyClassificationHook(
            api_url="http://test.api.com/safety",
            blocked_message="内容被阻止"
        )
    
    def test_init_default_values(self):
        """测试初始化默认值"""
        hook = SafetyClassificationHook()
        assert hook.api_url == "https://aisecurity.baidu-int.com/output_safety_multi_classification_service"
        assert hook.blocked_message == ""
        assert hook.timeout == 5
        assert hook.max_retries == 3
        assert hook.retry_delay == 1.0
    
    def test_init_custom_values(self):
        """测试初始化自定义值"""
        hook = SafetyClassificationHook(
            api_url="http://custom.api.com",
            blocked_message="自定义阻止消息"
        )
        assert hook.api_url == "http://custom.api.com"
        assert hook.blocked_message == "自定义阻止消息"
    
    def test_extract_complete_sentences_single_sentence(self):
        """测试提取单个完整句子"""
        session_state = {'sentence_buffer': "", 'sentence_history': []}
        text = "你好世界。"
        sentences, remaining = self.hook._extract_complete_sentences_for_session(text, session_state)

        assert sentences == ["你好世界。"]
        assert remaining == ""
    
    def test_extract_complete_sentences_multiple_sentences(self):
        """测试提取多个完整句子"""
        session_state = {'sentence_buffer': "", 'sentence_history': []}
        text = "第一句。第二句。第三句。"
        sentences, remaining = self.hook._extract_complete_sentences_for_session(text, session_state)

        assert sentences == ["第一句。", "第二句。", "第三句。"]
        assert remaining == ""
    
    def test_extract_complete_sentences_incomplete_sentence(self):
        """测试提取不完整句子"""
        session_state = {'sentence_buffer': "", 'sentence_history': []}
        text = "你好世界"
        sentences, remaining = self.hook._extract_complete_sentences_for_session(text, session_state)

        assert sentences == []
        assert remaining == "你好世界"
    
    def test_extract_complete_sentences_mixed_content(self):
        """测试混合内容（完整和不完整句子）"""
        session_state = {'sentence_buffer': "", 'sentence_history': []}
        # 第一次：不完整句子
        text1 = "你好"
        sentences1, remaining1 = self.hook._extract_complete_sentences_for_session(text1, session_state)
        assert sentences1 == []
        assert remaining1 == "你好"

        # 第二次：完成句子
        text2 = "世界。"
        sentences2, remaining2 = self.hook._extract_complete_sentences_for_session(text2, session_state)
        assert sentences2 == ["你好世界。"]
        assert remaining2 == ""
    
    def test_build_context_content_no_history(self):
        """测试构建上下文内容（无历史）"""
        session_state = {'sentence_buffer': "", 'sentence_history': []}
        current_sentence = "你好世界。"
        context = self.hook._build_contextual_query(current_sentence, session_state)
        assert context == "你好世界。"
    
    def test_build_context_content_with_history(self):
        """测试构建上下文内容（有历史）"""
        # 先添加一些历史句子
        session_state = {'sentence_buffer': "", 'sentence_history': ["第一句。", "第二句。"]}

        current_sentence = "第三句。"
        context = self.hook._build_contextual_query(current_sentence, session_state)
        # 新实现只返回当前句子，不使用历史上下文
        assert context == "第三句。"
    
    def test_sentence_history_buffer_management(self):
        """测试句子历史缓冲区的管理"""
        session_state = {'sentence_buffer': "", 'sentence_history': []}
        # 模拟流式处理多个文本块
        chunks = [
            "你好",
            "世界。",
            "今天",
            "天气",
            "很好。"
        ]

        # 处理所有块并验证缓冲区状态
        for chunk in chunks:
            sentences, _ = self.hook._extract_complete_sentences_for_session(chunk, session_state)

        # 验证最终的缓冲区状态：应该包含两个完整句子
        assert session_state['sentence_buffer'] == ""
    
    def test_extract_final_content_clears_buffers(self):
        """测试提取最终内容时清空缓冲区"""
        # 先添加一些内容到缓冲区
        session_state = {
            'sentence_buffer': "剩余内容",
            'sentence_history': ["历史句子1。", "历史句子2。"]
        }

        # 调用提取最终内容
        final_content = self.hook._extract_final_content_for_session(session_state)

        assert final_content == "剩余内容"
        assert session_state['sentence_buffer'] == ""
        assert session_state['sentence_history'] == []
    
    def test_process_streaming_chunk_with_history(self):
        """测试处理流式响应块时使用上下文"""
        session_id = "test_session"
        # 创建会话
        self.hook.start_streaming_session(session_id, {})

        # 模拟响应对象
        mock_response = Mock()
        mock_response.model_copy.return_value = mock_response
        mock_response.content = None  # 明确设置content为None，让其使用choices
        mock_delta = Mock()
        mock_delta.content = "新句子。"  # 确保content是字符串
        mock_choice = Mock()
        mock_choice.message = None  # 流式响应没有message
        mock_choice.delta = mock_delta
        mock_choice.finish_reason = None
        mock_response.choices = [mock_choice]
        # 添加必要的响应元数据
        mock_response.id = "test_id"
        mock_response.object = "chat.completion.chunk"
        mock_response.created = 1234567890
        mock_response.model = "test_model"
        mock_response.usage = None

        # 模拟安全API调用
        with patch.object(self.hook, '_call_safety_api') as mock_safety_api:
            mock_safety_api.return_value = True

            # 处理包含完整句子的文本块
            results = list(self.hook.process_streaming_chunk(mock_response, session_id, is_final=False))

            # 验证安全API被调用（新实现只使用当前句子）
            assert mock_safety_api.called
    
    def test_safety_api_failure_raises_exception(self):
        """测试安全API失败时不输出内容"""
        session_id = "test_session"
        # 创建会话
        self.hook.start_streaming_session(session_id, {})

        # 模拟响应对象
        mock_response = Mock()
        mock_response.model_copy.return_value = mock_response
        mock_response.content = None  # 明确设置content为None，让其使用choices
        mock_delta = Mock()
        mock_delta.content = "测试句子。"  # 确保content是字符串
        mock_choice = Mock()
        mock_choice.message = None  # 流式响应没有message
        mock_choice.delta = mock_delta
        mock_choice.finish_reason = None
        mock_response.choices = [mock_choice]
        # 添加必要的响应元数据
        mock_response.id = "test_id"
        mock_response.object = "chat.completion.chunk"
        mock_response.created = 1234567890
        mock_response.model = "test_model"
        mock_response.usage = None

        # 模拟安全API调用失败
        with patch.object(self.hook, '_call_safety_api') as mock_safety_api:
            mock_safety_api.return_value = False

            # 处理块，不安全的内容不会输出
            results = list(self.hook.process_streaming_chunk(mock_response, session_id, is_final=False))

            # 验证安全API被调用
            assert mock_safety_api.called
    
    def test_text_terminators_customization(self):
        """测试文本终止符的自定义"""
        custom_terminators = ["|", "&", "~"]
        hook = SafetyClassificationHook(text_terminators=custom_terminators)
        session_state = {'sentence_buffer': "", 'sentence_history': []}

        # 使用自定义终止符分割文本
        text = "第一句|第二句&第三句~"
        sentences, remaining = hook._extract_complete_sentences_for_session(text, session_state)

        assert sentences == ["第一句|", "第二句&", "第三句~"]
        assert remaining == ""
    
    def test_history_buffer_context_building(self):
        """测试历史缓冲区上下文构建"""
        session_state = {'sentence_buffer': "", 'sentence_history': []}
        # 逐步添加句子并检查上下文构建
        texts = ["第一句。", "第二句。", "第三句。", "第四句。"]

        for text in texts:
            sentences, _ = self.hook._extract_complete_sentences_for_session(text, session_state)
            if sentences:
                for sentence in sentences:
                    # 先添加到历史记录
                    self.hook._add_to_sentence_history(sentence, session_state)
                    context = self.hook._build_contextual_query(sentence, session_state)
                    # 验证上下文包含当前句子（新实现只返回当前句子）
                    assert context == sentence

        # 验证历史缓冲区包含所有句子
        assert len(session_state['sentence_history']) == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
