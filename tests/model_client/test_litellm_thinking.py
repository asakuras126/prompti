"""Unit tests for LiteLLM Claude thinking parameter support."""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from prompti.model_client.base import ModelConfig, RunParams
from prompti.model_client.litellm import LiteLLMClient, SyncLiteLLMClient
from prompti.message import Message


class TestLiteLLMThinkingParameter:
    """Test LiteLLM client thinking parameter support for Claude models."""

    def test_thinking_enabled_with_budget_tokens(self):
        """Test thinking parameter with type='enabled' and budget_tokens."""
        config = ModelConfig(
            provider="litellm",
            model="claude-sonnet-4-5-20250929",
            api_key="test_key",
            api_url="https://api.anthropic.com",
            ext={
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": 10000
                }
            }
        )

        client = SyncLiteLLMClient(config)

        params = RunParams(
            messages=[Message(role="user", content="Solve this complex problem")],
            stream=False
        )

        # Build request data
        request_data = client._build_request_data(params)

        # Verify thinking parameter is included
        assert "thinking" in request_data
        assert request_data["thinking"]["type"] == "enabled"
        assert request_data["thinking"]["budget_tokens"] == 10000

    def test_thinking_disabled(self):
        """Test thinking parameter with type='disabled'."""
        config = ModelConfig(
            provider="litellm",
            model="claude-sonnet-4-5-20250929",
            api_key="test_key",
            ext={
                "thinking": {
                    "type": "disabled"
                }
            }
        )

        client = SyncLiteLLMClient(config)

        params = RunParams(
            messages=[Message(role="user", content="Simple question")],
            stream=False
        )

        request_data = client._build_request_data(params)

        assert "thinking" in request_data
        assert request_data["thinking"]["type"] == "disabled"

    def test_no_thinking_parameter(self):
        """Test request without thinking parameter."""
        config = ModelConfig(
            provider="litellm",
            model="claude-sonnet-4-5-20250929",
            api_key="test_key"
        )

        client = SyncLiteLLMClient(config)

        params = RunParams(
            messages=[Message(role="user", content="Hello")],
            stream=False
        )

        request_data = client._build_request_data(params)

        # Verify thinking parameter is not included
        assert "thinking" not in request_data

    def test_invalid_thinking_type(self):
        """Test invalid thinking type is rejected."""
        config = ModelConfig(
            provider="litellm",
            model="claude-sonnet-4-5-20250929",
            api_key="test_key",
            ext={
                "thinking": {
                    "type": "invalid_type"  # Invalid type
                }
            }
        )

        client = SyncLiteLLMClient(config)

        params = RunParams(
            messages=[Message(role="user", content="Hello")],
            stream=False
        )

        request_data = client._build_request_data(params)

        # Verify thinking parameter is NOT included due to invalid type
        assert "thinking" not in request_data

    def test_thinking_with_minimal_config(self):
        """Test thinking parameter with minimal configuration (only type)."""
        config = ModelConfig(
            provider="litellm",
            model="claude-sonnet-4-5-20250929",
            api_key="test_key",
            ext={
                "thinking": {
                    "type": "enabled"
                    # No budget_tokens specified
                }
            }
        )

        client = SyncLiteLLMClient(config)

        params = RunParams(
            messages=[Message(role="user", content="Question")],
            stream=False
        )

        request_data = client._build_request_data(params)

        assert "thinking" in request_data
        assert request_data["thinking"]["type"] == "enabled"
        assert "budget_tokens" not in request_data["thinking"]

    @pytest.mark.asyncio
    async def test_async_client_thinking_parameter(self):
        """Test thinking parameter with async LiteLLM client."""
        config = ModelConfig(
            provider="litellm",
            model="claude-sonnet-4-5-20250929",
            api_key="test_key",
            ext={
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": 5000
                }
            }
        )

        client = LiteLLMClient(config)

        params = RunParams(
            messages=[Message(role="user", content="Complex reasoning task")],
            stream=False
        )

        request_data = client._build_request_data(params)

        assert "thinking" in request_data
        assert request_data["thinking"]["type"] == "enabled"
        assert request_data["thinking"]["budget_tokens"] == 5000

    def test_thinking_parameter_not_modified_by_extra_params(self):
        """Test that thinking parameter is not overridden by extra_params."""
        config = ModelConfig(
            provider="litellm",
            model="claude-sonnet-4-5-20250929",
            api_key="test_key",
            ext={
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": 10000
                }
            }
        )

        client = SyncLiteLLMClient(config)

        # Extra params should not override thinking from ext
        params = RunParams(
            messages=[Message(role="user", content="Test")],
            stream=False,
            extra_params={
                "some_other_param": "value"
            }
        )

        request_data = client._build_request_data(params)

        # Verify thinking is still from ext config
        assert "thinking" in request_data
        assert request_data["thinking"]["type"] == "enabled"
        assert request_data["thinking"]["budget_tokens"] == 10000
        assert request_data["some_other_param"] == "value"

    def test_thinking_with_non_dict_value(self):
        """Test that non-dict thinking values are ignored."""
        config = ModelConfig(
            provider="litellm",
            model="claude-sonnet-4-5-20250929",
            api_key="test_key",
            ext={
                "thinking": "invalid_string"  # Should be a dict
            }
        )

        client = SyncLiteLLMClient(config)

        params = RunParams(
            messages=[Message(role="user", content="Test")],
            stream=False
        )

        request_data = client._build_request_data(params)

        # Non-dict values should be ignored
        assert "thinking" not in request_data

    def test_thinking_parameter_copied_not_referenced(self):
        """Test that thinking config is copied, not referenced."""
        thinking_config = {
            "type": "enabled",
            "budget_tokens": 10000
        }

        config = ModelConfig(
            provider="litellm",
            model="claude-sonnet-4-5-20250929",
            api_key="test_key",
            ext={"thinking": thinking_config}
        )

        client = SyncLiteLLMClient(config)

        params = RunParams(
            messages=[Message(role="user", content="Test")],
            stream=False
        )

        request_data = client._build_request_data(params)

        # Modify request_data thinking
        request_data["thinking"]["budget_tokens"] = 99999

        # Original config should not be modified
        assert thinking_config["budget_tokens"] == 10000

    @pytest.mark.asyncio
    async def test_thinking_with_streaming(self):
        """Test thinking parameter with streaming enabled."""
        config = ModelConfig(
            provider="litellm",
            model="claude-sonnet-4-5-20250929",
            api_key="test_key",
            ext={
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": 8000
                }
            }
        )

        client = LiteLLMClient(config)

        params = RunParams(
            messages=[Message(role="user", content="Stream this response")],
            stream=True  # Streaming enabled
        )

        request_data = client._build_request_data(params)

        # Thinking should work with streaming
        assert "thinking" in request_data
        assert request_data["thinking"]["type"] == "enabled"
        assert request_data["thinking"]["budget_tokens"] == 8000
        assert request_data["stream"] is True

    def test_thinking_with_different_models(self):
        """Test thinking parameter with different Claude model versions."""
        models = [
            "claude-sonnet-4-5-20250929",
            "claude-haiku-4-5-20251001",
            "claude-opus-4",
            "anthropic/claude-3-5-sonnet"
        ]

        for model in models:
            config = ModelConfig(
                provider="litellm",
                model=model,
                api_key="test_key",
                ext={
                    "thinking": {
                        "type": "enabled",
                        "budget_tokens": 10000
                    }
                }
            )

            client = SyncLiteLLMClient(config)

            params = RunParams(
                messages=[Message(role="user", content="Test")],
                stream=False
            )

            request_data = client._build_request_data(params)

            # All models should support thinking
            assert "thinking" in request_data
            assert request_data["thinking"]["type"] == "enabled"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
