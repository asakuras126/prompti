"""Test LiteLLM reasoning content extraction from Claude responses."""

import pytest
from unittest.mock import MagicMock

from prompti.model_client.base import ModelConfig
from prompti.model_client.litellm import LiteLLMClient, SyncLiteLLMClient


class TestLiteLLMReasoningExtraction:
    """Test extraction of reasoning/thinking content from Claude responses."""

    def test_sync_client_extracts_reasoning_content(self):
        """Test that sync client correctly extracts reasoning_content from response."""
        config = ModelConfig(
            provider="litellm",
            model="claude-sonnet-4-5-20250929",
            api_key="test_key",
            ext={"thinking": {"type": "enabled", "budget_tokens": 10000}}
        )

        client = SyncLiteLLMClient(config)

        # Mock response with reasoning_content
        # Note: litellm already extracts thinking_blocks into reasoning_content
        mock_response = MagicMock()
        mock_response.id = "chatcmpl-test"
        mock_response.created = 1234567890
        mock_response.model = "claude-sonnet-4-5-20250929"

        # Mock message with reasoning_content
        mock_message = MagicMock()
        mock_message.role = "assistant"
        mock_message.content = "你好！很高兴见到你。"
        mock_message.tool_calls = None
        # litellm puts the thinking content in reasoning_content
        mock_message.reasoning_content = "The user has greeted me in Chinese with \"你好\"."

        mock_choice = MagicMock()
        mock_choice.index = 0
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"

        mock_response.choices = [mock_choice]

        # Mock usage
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 38
        mock_usage.completion_tokens = 81
        mock_usage.total_tokens = 119
        mock_usage.cache_creation_input_tokens = None
        mock_usage.cache_read_input_tokens = None
        mock_usage.cached_tokens = None
        mock_response.usage = mock_usage

        # Process the response
        result = client._process_non_streaming_response(mock_response)

        # Verify content is extracted
        assert result.choices[0].message.content == "你好！很高兴见到你。"

        # Verify reasoning_content is extracted
        assert result.choices[0].message.reasoning_content == "The user has greeted me in Chinese with \"你好\"."

    @pytest.mark.asyncio
    async def test_async_client_extracts_reasoning_content(self):
        """Test that async client correctly extracts reasoning_content from response."""
        config = ModelConfig(
            provider="litellm",
            model="claude-sonnet-4-5-20250929",
            api_key="test_key",
            ext={"thinking": {"type": "enabled", "budget_tokens": 10000}}
        )

        client = LiteLLMClient(config)

        # Mock response with reasoning_content
        mock_response = MagicMock()
        mock_response.id = "chatcmpl-test"
        mock_response.created = 1234567890
        mock_response.model = "claude-sonnet-4-5-20250929"

        # Mock message with reasoning_content
        mock_message = MagicMock()
        mock_message.role = "assistant"
        mock_message.content = "Complex answer here"
        mock_message.tool_calls = None
        # litellm puts all thinking content in reasoning_content
        mock_message.reasoning_content = "Let me think about this step by step..."

        mock_choice = MagicMock()
        mock_choice.index = 0
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"

        mock_response.choices = [mock_choice]

        # Mock usage
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 50
        mock_usage.completion_tokens = 100
        mock_usage.total_tokens = 150
        mock_usage.cache_creation_input_tokens = None
        mock_usage.cache_read_input_tokens = None
        mock_usage.cached_tokens = None
        mock_response.usage = mock_usage

        # Process the response
        result = client._process_non_streaming_response(mock_response)

        # Verify content is extracted
        assert result.choices[0].message.content == "Complex answer here"

        # Verify reasoning_content is extracted
        assert result.choices[0].message.reasoning_content == "Let me think about this step by step..."

    def test_response_without_reasoning_content(self):
        """Test that responses without reasoning_content work correctly."""
        config = ModelConfig(
            provider="litellm",
            model="gpt-4",
            api_key="test_key"
        )

        client = SyncLiteLLMClient(config)

        # Mock response WITHOUT reasoning_content
        mock_response = MagicMock()
        mock_response.id = "chatcmpl-test"
        mock_response.created = 1234567890
        mock_response.model = "gpt-4"

        # Mock message WITHOUT reasoning fields
        mock_message = MagicMock()
        mock_message.role = "assistant"
        mock_message.content = "Simple answer"
        mock_message.tool_calls = None
        # No reasoning_content, thinking_blocks, or provider_specific_fields

        # Simulate hasattr returning False for these fields
        type(mock_message).reasoning_content = property(lambda self: None)
        type(mock_message).thinking_blocks = property(lambda self: None)
        type(mock_message).provider_specific_fields = property(lambda self: None)

        mock_choice = MagicMock()
        mock_choice.index = 0
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"

        mock_response.choices = [mock_choice]

        # Mock usage
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5
        mock_usage.total_tokens = 15
        mock_usage.cache_creation_input_tokens = None
        mock_usage.cache_read_input_tokens = None
        mock_usage.cached_tokens = None
        mock_response.usage = mock_usage

        # Process the response
        result = client._process_non_streaming_response(mock_response)

        # Verify basic fields work and reasoning_content is None
        assert result.choices[0].message.content == "Simple answer"
        assert result.choices[0].message.reasoning_content is None

    def test_empty_reasoning_content_not_added(self):
        """Test that empty reasoning_content is not added."""
        config = ModelConfig(
            provider="litellm",
            model="claude-sonnet-4-5-20250929",
            api_key="test_key"
        )

        client = SyncLiteLLMClient(config)

        # Mock response with empty reasoning_content
        mock_response = MagicMock()
        mock_response.id = "chatcmpl-test"
        mock_response.created = 1234567890
        mock_response.model = "claude-sonnet-4-5-20250929"

        mock_message = MagicMock()
        mock_message.role = "assistant"
        mock_message.content = "Answer"
        mock_message.tool_calls = None
        mock_message.reasoning_content = ""  # Empty string
        mock_message.thinking_blocks = []  # Empty list
        mock_message.provider_specific_fields = {}  # Empty dict

        mock_choice = MagicMock()
        mock_choice.index = 0
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"

        mock_response.choices = [mock_choice]

        # Mock usage
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5
        mock_usage.total_tokens = 15
        mock_usage.cache_creation_input_tokens = None
        mock_usage.cache_read_input_tokens = None
        mock_usage.cached_tokens = None
        mock_response.usage = mock_usage

        # Process the response
        result = client._process_non_streaming_response(mock_response)

        # Verify empty values are handled correctly
        assert result.choices[0].message.content == "Answer"
        # Empty values should not be added (falsy in Python)
        assert result.choices[0].message.reasoning_content is None

    def test_reasoning_content_with_complex_thinking(self):
        """Test extraction of reasoning_content with complex thinking."""
        config = ModelConfig(
            provider="litellm",
            model="claude-sonnet-4-5-20250929",
            api_key="test_key"
        )

        client = SyncLiteLLMClient(config)

        # Mock response with complex reasoning_content
        # Note: litellm combines all thinking_blocks into reasoning_content
        mock_response = MagicMock()
        mock_response.id = "chatcmpl-test"
        mock_response.created = 1234567890
        mock_response.model = "claude-sonnet-4-5-20250929"

        mock_message = MagicMock()
        mock_message.role = "assistant"
        mock_message.content = "Final answer"
        mock_message.tool_calls = None
        # litellm already combines all thinking steps into reasoning_content
        mock_message.reasoning_content = "Step 1: First thought\nStep 2: Second thought\nStep 3: Third thought"

        mock_choice = MagicMock()
        mock_choice.index = 0
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"

        mock_response.choices = [mock_choice]

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 20
        mock_usage.completion_tokens = 100
        mock_usage.total_tokens = 120
        mock_usage.cache_creation_input_tokens = None
        mock_usage.cache_read_input_tokens = None
        mock_usage.cached_tokens = None
        mock_response.usage = mock_usage

        # Process the response
        result = client._process_non_streaming_response(mock_response)

        # Verify reasoning_content is extracted
        assert result.choices[0].message.reasoning_content is not None
        assert "Step 1: First thought" in result.choices[0].message.reasoning_content
        assert "Step 2: Second thought" in result.choices[0].message.reasoning_content
        assert "Step 3: Third thought" in result.choices[0].message.reasoning_content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
