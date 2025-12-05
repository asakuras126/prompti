"""Unit tests for OpenAI Response API client."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any, Dict, List

import pytest

from prompti.model_client.base import ModelConfig, RunParams
from prompti.model_client.openai_response import (
    OpenAIResponseClient,
    SyncOpenAIResponseClient,
    messages_to_responses_items,
    responses_to_completion_format,
)
from prompti.message import Message


class TestMessagesToResponsesItems:
    """Test messages_to_responses_items conversion function."""

    def test_convert_system_message(self):
        """Test converting system message to instructions."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"}
        ]

        items, instructions = messages_to_responses_items(messages)

        assert instructions == "You are a helpful assistant."
        assert len(items) == 1
        assert items[0]["role"] == "user"
        assert items[0]["content"] == "Hello"

    def test_convert_user_message(self):
        """Test converting user message."""
        messages = [
            {"role": "user", "content": "What is AI?"}
        ]

        items, instructions = messages_to_responses_items(messages)

        assert instructions == ""
        assert len(items) == 1
        assert items[0]["role"] == "user"
        assert items[0]["content"] == "What is AI?"

    def test_convert_assistant_message_with_text(self):
        """Test converting assistant message with text content."""
        messages = [
            {"role": "assistant", "content": "AI stands for Artificial Intelligence."}
        ]

        items, instructions = messages_to_responses_items(messages)

        assert len(items) == 1
        assert items[0]["role"] == "assistant"
        assert items[0]["content"] == "AI stands for Artificial Intelligence."

    def test_convert_assistant_message_with_tool_calls(self):
        """Test converting assistant message with tool calls."""
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "Beijing"}'
                        }
                    }
                ]
            }
        ]

        items, instructions = messages_to_responses_items(messages)

        assert len(items) == 1
        assert items[0]["type"] == "function_call"
        assert items[0]["call_id"] == "call_123"
        assert items[0]["name"] == "get_weather"
        assert items[0]["arguments"] == '{"location": "Beijing"}'

    def test_convert_tool_message(self):
        """Test converting tool response message."""
        messages = [
            {
                "role": "tool",
                "tool_call_id": "call_123",
                "content": '{"temperature": "25°C"}'
            }
        ]

        items, instructions = messages_to_responses_items(messages)

        assert len(items) == 1
        assert items[0]["type"] == "function_call_output"
        assert items[0]["call_id"] == "call_123"
        assert items[0]["output"] == '{"temperature": "25°C"}'

    def test_convert_multimodal_message(self):
        """Test converting multimodal message with image."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
                ]
            }
        ]

        items, instructions = messages_to_responses_items(messages)

        assert len(items) == 1
        assert items[0]["role"] == "user"
        assert isinstance(items[0]["content"], list)
        assert items[0]["content"][0]["type"] == "input_text"
        assert items[0]["content"][0]["text"] == "What's in this image?"
        assert items[0]["content"][1]["type"] == "input_image"

    def test_empty_messages(self):
        """Test with empty messages list."""
        messages = []

        items, instructions = messages_to_responses_items(messages)

        assert items == []
        assert instructions == ""


class TestResponsesToCompletionFormat:
    """Test responses_to_completion_format conversion function."""

    def test_convert_text_response(self):
        """Test converting text response."""
        # Mock response object
        mock_response = MagicMock()
        mock_response.id = "resp_123"
        mock_response.created_at = 1234567890
        mock_response.model = "o1-preview"

        # Mock output items
        mock_message = MagicMock()
        mock_message.type = "message"
        mock_message.content = [MagicMock(type="output_text", text="Hello, how can I help?")]
        mock_response.output = [mock_message]

        # Mock usage
        mock_usage = MagicMock()
        mock_usage.input_tokens = 10
        mock_usage.output_tokens = 20
        mock_usage.total_tokens = 30
        mock_usage.output_tokens_details = None
        mock_response.usage = mock_usage

        result = responses_to_completion_format(mock_response)

        assert result["id"] == "resp_123"
        assert result["model"] == "o1-preview"
        assert result["choices"][0]["message"]["content"] == "Hello, how can I help?"
        assert result["choices"][0]["finish_reason"] == "stop"
        assert result["usage"]["prompt_tokens"] == 10
        assert result["usage"]["completion_tokens"] == 20

    def test_convert_function_call_response(self):
        """Test converting function call response."""
        # Mock response object
        mock_response = MagicMock()
        mock_response.id = "resp_456"
        mock_response.created = 1234567890
        mock_response.model = "o1-preview"

        # Mock function call
        mock_function_call = MagicMock()
        mock_function_call.type = "function_call"
        mock_function_call.call_id = "call_789"
        mock_function_call.name = "get_weather"
        mock_function_call.arguments = '{"location": "Beijing"}'
        mock_response.output = [mock_function_call]

        # Mock usage
        mock_usage = MagicMock()
        mock_usage.input_tokens = 15
        mock_usage.output_tokens = 5
        mock_usage.total_tokens = 20
        mock_usage.output_tokens_details = None
        mock_response.usage = mock_usage

        result = responses_to_completion_format(mock_response)

        assert result["choices"][0]["finish_reason"] == "tool_calls"
        assert len(result["choices"][0]["message"]["tool_calls"]) == 1
        assert result["choices"][0]["message"]["tool_calls"][0]["id"] == "call_789"
        assert result["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "get_weather"

    def test_convert_reasoning_response(self):
        """Test converting response with reasoning content."""
        # Mock response object
        mock_response = MagicMock()
        mock_response.id = "resp_999"
        mock_response.created_at = 1234567890
        mock_response.model = "o1-preview"

        # Mock reasoning and message
        mock_reasoning = MagicMock()
        mock_reasoning.type = "reasoning"
        mock_reasoning.content = [MagicMock(text="Let me think about this...")]
        mock_reasoning.summary = None

        mock_message = MagicMock()
        mock_message.type = "message"
        mock_message.content = [MagicMock(type="output_text", text="The answer is 42.")]

        mock_response.output = [mock_reasoning, mock_message]

        # Mock usage with reasoning tokens
        mock_usage = MagicMock()
        mock_usage.input_tokens = 10
        mock_usage.output_tokens = 25
        mock_usage.total_tokens = 35
        mock_output_details = MagicMock()
        mock_output_details.reasoning_tokens = 15
        mock_usage.output_tokens_details = mock_output_details
        mock_response.usage = mock_usage

        result = responses_to_completion_format(mock_response)

        assert result["choices"][0]["message"]["reasoning_content"] == "Let me think about this..."
        assert result["choices"][0]["message"]["content"] == "The answer is 42."
        assert result["usage"]["completion_tokens_details"]["reasoning_tokens"] == 15


class TestOpenAIResponseClient:
    """Test OpenAIResponseClient async client."""

    @pytest.mark.asyncio
    async def test_basic_completion(self):
        """Test basic non-streaming completion."""
        # Setup
        config = ModelConfig(
            provider="openai_response",
            model="o1-preview",
            api_key="test_key",
            api_url="https://api.openai.com/v1"
        )

        client = OpenAIResponseClient(config)

        # Mock OpenAI client
        mock_response = MagicMock()
        mock_response.id = "resp_test"
        mock_response.created_at = 1234567890
        mock_response.model = "o1-preview"

        mock_message = MagicMock()
        mock_message.type = "message"
        mock_message.content = [MagicMock(type="output_text", text="Test response")]
        mock_response.output = [mock_message]

        mock_usage = MagicMock()
        mock_usage.input_tokens = 10
        mock_usage.output_tokens = 20
        mock_usage.total_tokens = 30
        mock_usage.output_tokens_details = None
        mock_response.usage = mock_usage

        with patch.object(client.openai_client.responses, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response

            # Execute
            params = RunParams(
                messages=[Message(role="user", content="Hello")],
                stream=False
            )

            results = []
            async for response in client._run(params):
                results.append(response)

            # Verify
            assert len(results) == 1
            assert results[0].choices[0].message.content == "Test response"
            assert results[0].usage.prompt_tokens == 10
            assert results[0].usage.completion_tokens == 20

    @pytest.mark.asyncio
    async def test_reasoning_effort_parameter(self):
        """Test that reasoning.effort parameter is passed correctly."""
        # Setup with reasoning_effort in ext
        config = ModelConfig(
            provider="openai_response",
            model="o1-preview",
            api_key="test_key",
            api_url="https://api.openai.com/v1",
            ext={"reasoning_effort": "high"}
        )

        client = OpenAIResponseClient(config)

        # Mock OpenAI client
        mock_response = MagicMock()
        mock_response.id = "resp_test"
        mock_response.created_at = 1234567890
        mock_response.model = "o1-preview"

        mock_message = MagicMock()
        mock_message.type = "message"
        mock_message.content = [MagicMock(type="output_text", text="Test response")]
        mock_response.output = [mock_message]

        mock_usage = MagicMock()
        mock_usage.input_tokens = 10
        mock_usage.output_tokens = 20
        mock_usage.total_tokens = 30
        mock_usage.output_tokens_details = None
        mock_response.usage = mock_usage

        with patch.object(client.openai_client.responses, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response

            # Execute
            params = RunParams(
                messages=[Message(role="user", content="Solve this complex problem")],
                stream=False
            )

            results = []
            async for response in client._run(params):
                results.append(response)

            # Verify reasoning parameter was passed
            call_kwargs = mock_create.call_args[1]
            assert "reasoning" in call_kwargs
            assert call_kwargs["reasoning"]["effort"] == "high"

    @pytest.mark.asyncio
    async def test_invalid_reasoning_effort(self):
        """Test handling of invalid reasoning_effort value."""
        # Setup with invalid reasoning_effort
        config = ModelConfig(
            provider="openai_response",
            model="o1-preview",
            api_key="test_key",
            api_url="https://api.openai.com/v1",
            ext={"reasoning_effort": "invalid"}
        )

        client = OpenAIResponseClient(config)

        # Mock OpenAI client
        mock_response = MagicMock()
        mock_response.id = "resp_test"
        mock_response.created_at = 1234567890
        mock_response.model = "o1-preview"

        mock_message = MagicMock()
        mock_message.type = "message"
        mock_message.content = [MagicMock(type="output_text", text="Test response")]
        mock_response.output = [mock_message]

        mock_usage = MagicMock()
        mock_usage.input_tokens = 10
        mock_usage.output_tokens = 20
        mock_usage.total_tokens = 30
        mock_usage.output_tokens_details = None
        mock_response.usage = mock_usage

        with patch.object(client.openai_client.responses, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response

            # Execute
            params = RunParams(
                messages=[Message(role="user", content="Hello")],
                stream=False
            )

            results = []
            async for response in client._run(params):
                results.append(response)

            # Verify reasoning parameter was NOT passed (invalid value)
            call_kwargs = mock_create.call_args[1]
            assert "reasoning" not in call_kwargs


class TestSyncOpenAIResponseClient:
    """Test SyncOpenAIResponseClient synchronous client."""

    def test_basic_completion(self):
        """Test basic non-streaming completion."""
        # Setup
        config = ModelConfig(
            provider="openai_response",
            model="o1-preview",
            api_key="test_key",
            api_url="https://api.openai.com/v1"
        )

        client = SyncOpenAIResponseClient(config)

        # Mock OpenAI client
        mock_response = MagicMock()
        mock_response.id = "resp_test"
        mock_response.created = 1234567890
        mock_response.model = "o1-preview"

        mock_message = MagicMock()
        mock_message.type = "message"
        mock_message.content = [MagicMock(type="output_text", text="Test response")]
        mock_response.output = [mock_message]

        mock_usage = MagicMock()
        mock_usage.input_tokens = 10
        mock_usage.output_tokens = 20
        mock_usage.total_tokens = 30
        mock_usage.output_tokens_details = None
        mock_response.usage = mock_usage

        with patch.object(client.openai_client.responses, 'create') as mock_create:
            mock_create.return_value = mock_response

            # Execute
            params = RunParams(
                messages=[Message(role="user", content="Hello")],
                stream=False
            )

            results = list(client._run(params))

            # Verify
            assert len(results) == 1
            assert results[0].choices[0].message.content == "Test response"
            assert results[0].usage.prompt_tokens == 10
            assert results[0].usage.completion_tokens == 20

    def test_reasoning_effort_low(self):
        """Test reasoning.effort=low parameter."""
        config = ModelConfig(
            provider="openai_response",
            model="o1-preview",
            api_key="test_key",
            api_url="https://api.openai.com/v1",
            ext={"reasoning_effort": "low"}
        )

        client = SyncOpenAIResponseClient(config)

        # Mock OpenAI client
        mock_response = MagicMock()
        mock_response.id = "resp_test"
        mock_response.created = 1234567890
        mock_response.model = "o1-preview"

        mock_message = MagicMock()
        mock_message.type = "message"
        mock_message.content = [MagicMock(type="output_text", text="Quick answer")]
        mock_response.output = [mock_message]

        mock_usage = MagicMock()
        mock_usage.input_tokens = 5
        mock_usage.output_tokens = 10
        mock_usage.total_tokens = 15
        mock_usage.output_tokens_details = None
        mock_response.usage = mock_usage

        with patch.object(client.openai_client.responses, 'create') as mock_create:
            mock_create.return_value = mock_response

            # Execute
            params = RunParams(
                messages=[Message(role="user", content="Simple question")],
                stream=False
            )

            results = list(client._run(params))

            # Verify reasoning parameter
            call_kwargs = mock_create.call_args[1]
            assert "reasoning" in call_kwargs
            assert call_kwargs["reasoning"]["effort"] == "low"

    def test_reasoning_effort_medium(self):
        """Test reasoning.effort=medium parameter."""
        config = ModelConfig(
            provider="openai_response",
            model="o1-preview",
            api_key="test_key",
            api_url="https://api.openai.com/v1",
            ext={"reasoning_effort": "medium"}
        )

        client = SyncOpenAIResponseClient(config)

        # Mock OpenAI client
        mock_response = MagicMock()
        mock_response.id = "resp_test"
        mock_response.created = 1234567890
        mock_response.model = "o1-preview"

        mock_message = MagicMock()
        mock_message.type = "message"
        mock_message.content = [MagicMock(type="output_text", text="Balanced answer")]
        mock_response.output = [mock_message]

        mock_usage = MagicMock()
        mock_usage.input_tokens = 10
        mock_usage.output_tokens = 20
        mock_usage.total_tokens = 30
        mock_usage.output_tokens_details = None
        mock_response.usage = mock_usage

        with patch.object(client.openai_client.responses, 'create') as mock_create:
            mock_create.return_value = mock_response

            # Execute
            params = RunParams(
                messages=[Message(role="user", content="Moderate question")],
                stream=False
            )

            results = list(client._run(params))

            # Verify reasoning parameter
            call_kwargs = mock_create.call_args[1]
            assert "reasoning" in call_kwargs
            assert call_kwargs["reasoning"]["effort"] == "medium"

    def test_tool_calling(self):
        """Test tool calling functionality."""
        config = ModelConfig(
            provider="openai_response",
            model="o1-preview",
            api_key="test_key",
            api_url="https://api.openai.com/v1"
        )

        client = SyncOpenAIResponseClient(config)

        # Mock OpenAI client with function call response
        mock_response = MagicMock()
        mock_response.id = "resp_tool"
        mock_response.created = 1234567890
        mock_response.model = "o1-preview"

        mock_function_call = MagicMock()
        mock_function_call.type = "function_call"
        mock_function_call.call_id = "call_123"
        mock_function_call.name = "get_weather"
        mock_function_call.arguments = '{"location": "Beijing"}'
        mock_response.output = [mock_function_call]

        mock_usage = MagicMock()
        mock_usage.input_tokens = 15
        mock_usage.output_tokens = 10
        mock_usage.total_tokens = 25
        mock_usage.output_tokens_details = None
        mock_response.usage = mock_usage

        with patch.object(client.openai_client.responses, 'create') as mock_create:
            mock_create.return_value = mock_response

            # Execute with tools
            from prompti.model_client.base import ToolParams, ToolSpec

            tool_spec = ToolSpec(
                name="get_weather",
                description="Get current weather",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    }
                }
            )

            params = RunParams(
                messages=[Message(role="user", content="What's the weather in Beijing?")],
                stream=False,
                tool_params=ToolParams(tools=[tool_spec])
            )

            results = list(client._run(params))

            # Verify
            assert len(results) == 1
            assert len(results[0].choices[0].message.tool_calls) == 1
            assert results[0].choices[0].message.tool_calls[0]["function"]["name"] == "get_weather"
            assert results[0].choices[0].finish_reason == "tool_calls"

    def test_temperature_parameter(self):
        """Test temperature parameter is passed correctly."""
        config = ModelConfig(
            provider="openai_response",
            model="o1-preview",
            api_key="test_key",
            api_url="https://api.openai.com/v1",
            temperature=0.7
        )

        client = SyncOpenAIResponseClient(config)

        # Mock OpenAI client
        mock_response = MagicMock()
        mock_response.id = "resp_test"
        mock_response.created = 1234567890
        mock_response.model = "o1-preview"

        mock_message = MagicMock()
        mock_message.type = "message"
        mock_message.content = [MagicMock(type="output_text", text="Test")]
        mock_response.output = [mock_message]

        mock_usage = MagicMock()
        mock_usage.input_tokens = 10
        mock_usage.output_tokens = 20
        mock_usage.total_tokens = 30
        mock_usage.output_tokens_details = None
        mock_response.usage = mock_usage

        with patch.object(client.openai_client.responses, 'create') as mock_create:
            mock_create.return_value = mock_response

            # Execute
            params = RunParams(
                messages=[Message(role="user", content="Hello")],
                stream=False
            )

            results = list(client._run(params))

            # Verify temperature was passed
            call_kwargs = mock_create.call_args[1]
            assert call_kwargs["temperature"] == 0.7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
