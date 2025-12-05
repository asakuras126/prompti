"""Unit tests for MemoryModelConfigLoader."""

import unittest
from collections import Counter

from prompti.loader.model_config_loader import MemoryModelConfigLoader, ModelConfigNotFoundError


class TestMemoryModelConfigLoader(unittest.TestCase):
    """Test cases for MemoryModelConfigLoader."""

    def setUp(self):
        """Set up test fixtures."""
        # Standard grouped models data
        self.grouped_models = {
            "gpt-4": [
                {
                    "name": "gpt-4",
                    "provider": "openai",
                    "url": "https://api.openai.com/v1",
                    "weight": 60,
                    "llm_tokens": ["openai_token"]
                },
                {
                    "name": "gpt-4",
                    "provider": "azure",
                    "url": "https://your-resource.openai.azure.com",
                    "weight": 40,
                    "llm_tokens": ["azure_token"]
                }
            ],
            "gpt-3.5-turbo": [
                {
                    "name": "gpt-3.5-turbo",
                    "provider": "openai",
                    "url": "https://api.openai.com/v1",
                    "weight": 80,
                    "llm_tokens": ["openai_token"]
                },
                {
                    "name": "gpt-3.5-turbo",
                    "provider": "litellm",
                    "url": "https://api.litellm.ai/v1",
                    "weight": 20,
                    "llm_tokens": ["litellm_token"]
                }
            ],
            "claude-3-sonnet": [
                {
                    "name": "claude-3-sonnet",
                    "provider": "anthropic",
                    "url": "https://api.anthropic.com",
                    "weight": 100,
                    "llm_tokens": ["anthropic_token"]
                }
            ]
        }
        
        # Token data
        self.tokens = [
            {
                "name": "openai_token",
                "token_config": {
                    "api_key": "sk-openai-api-key-here"
                }
            },
            {
                "name": "azure_token", 
                "token_config": {
                    "api_key": "azure-api-key-here"
                }
            },
            {
                "name": "litellm_token",
                "token_config": {
                    "api_key": "litellm-api-key-here"
                }
            },
            {
                "name": "anthropic_token",
                "token_config": {
                    "api_key": "anthropic-api-key-here"
                }
            }
        ]

    def test_initialization(self):
        """Test loader initialization."""
        loader = MemoryModelConfigLoader(
            grouped_models=self.grouped_models,
            tokens=self.tokens
        )
        
        # Verify loader was initialized correctly
        self.assertEqual(loader.grouped_models, self.grouped_models)
        self.assertEqual(loader.tokens, self.tokens)
        self.assertIsInstance(loader.models, dict)

    def test_list_models(self):
        """Test listing available models."""
        loader = MemoryModelConfigLoader(
            grouped_models=self.grouped_models,
            tokens=self.tokens
        )
        
        models = loader.list_models()
        expected_models = ["gpt-4", "gpt-3.5-turbo", "claude-3-sonnet"]
        
        self.assertEqual(set(models), set(expected_models))
        self.assertEqual(len(models), 3)

    def test_get_model_config_by_provider(self):
        """Test getting model config by specifying provider."""
        loader = MemoryModelConfigLoader(
            grouped_models=self.grouped_models,
            tokens=self.tokens
        )

        # Test OpenAI provider for GPT-4 (returns list, get first element)
        openai_configs = loader.get_model_config("gpt-4", provider="openai")
        self.assertIsInstance(openai_configs, list)
        self.assertGreater(len(openai_configs), 0)
        openai_config = openai_configs[0]
        self.assertEqual(openai_config.provider, "openai")
        self.assertEqual(openai_config.model, "gpt-4")
        self.assertEqual(openai_config.api_url, "https://api.openai.com/v1")
        self.assertEqual(openai_config.weight, 60)
        self.assertEqual(openai_config.api_key, "sk-openai-api-key-here")

        # Test Azure provider for GPT-4 (returns list, get first element)
        azure_configs = loader.get_model_config("gpt-4", provider="azure")
        self.assertIsInstance(azure_configs, list)
        self.assertGreater(len(azure_configs), 0)
        azure_config = azure_configs[0]
        self.assertEqual(azure_config.provider, "azure")
        self.assertEqual(azure_config.model, "gpt-4")
        self.assertEqual(azure_config.api_url, "https://your-resource.openai.azure.com")
        self.assertEqual(azure_config.weight, 40)
        self.assertEqual(azure_config.api_key, "azure-api-key-here")

    def test_get_model_config_load_balancing(self):
        """Test that configs are returned sorted by weight (highest first)."""
        loader = MemoryModelConfigLoader(
            grouped_models=self.grouped_models,
            tokens=self.tokens
        )

        # API now returns sorted list of all configs (highest weight first)
        configs = loader.get_model_config("gpt-4")
        self.assertIsInstance(configs, list)
        self.assertEqual(len(configs), 2)  # Two providers for gpt-4

        # Verify configs are sorted by weight (OpenAI weight 60 should be first)
        self.assertEqual(configs[0].provider, "openai")
        self.assertEqual(configs[0].weight, 60)
        self.assertEqual(configs[1].provider, "azure")
        self.assertEqual(configs[1].weight, 40)

        # Verify both providers are present
        providers = [c.provider for c in configs]
        self.assertIn("openai", providers)
        self.assertIn("azure", providers)

    def test_token_association(self):
        """Test API key association from tokens."""
        loader = MemoryModelConfigLoader(
            grouped_models=self.grouped_models,
            tokens=self.tokens
        )

        # Test different models have correct API keys (returns list, get first element)
        gpt4_configs = loader.get_model_config("gpt-4", provider="openai")
        self.assertGreater(len(gpt4_configs), 0)
        self.assertEqual(gpt4_configs[0].api_key, "sk-openai-api-key-here")

        gpt35_configs = loader.get_model_config("gpt-3.5-turbo", provider="litellm")
        self.assertGreater(len(gpt35_configs), 0)
        self.assertEqual(gpt35_configs[0].api_key, "litellm-api-key-here")

        claude_configs = loader.get_model_config("claude-3-sonnet")
        self.assertGreater(len(claude_configs), 0)
        self.assertEqual(claude_configs[0].api_key, "anthropic-api-key-here")

    def test_model_not_found_error(self):
        """Test error when model is not found."""
        loader = MemoryModelConfigLoader(
            grouped_models=self.grouped_models,
            tokens=self.tokens
        )
        
        with self.assertRaises(ModelConfigNotFoundError):
            loader.get_model_config("nonexistent-model")

    def test_provider_not_found(self):
        """Test behavior when specified provider is not found.

        New API behavior: provider parameter is used for prioritization,
        not filtering. If provider doesn't exist, configs are still returned
        (just without that provider).
        """
        loader = MemoryModelConfigLoader(
            grouped_models=self.grouped_models,
            tokens=self.tokens
        )

        # API returns all configs even if specified provider doesn't exist
        configs = loader.get_model_config("gpt-4", provider="nonexistent-provider")
        self.assertIsInstance(configs, list)
        self.assertGreater(len(configs), 0)

        # Verify returned configs don't have the nonexistent provider
        for config in configs:
            self.assertNotEqual(config.provider, "nonexistent-provider")

        # Should have the actual providers (openai and azure)
        providers = [c.provider for c in configs]
        self.assertIn("openai", providers)
        self.assertIn("azure", providers)

    def test_empty_loader(self):
        """Test loader with empty data."""
        empty_loader = MemoryModelConfigLoader(grouped_models={}, tokens=[])
        
        # Test empty model list
        models = empty_loader.list_models()
        self.assertEqual(models, [])
        self.assertEqual(len(models), 0)
        
        # Test error on empty loader
        with self.assertRaises(ModelConfigNotFoundError):
            empty_loader.get_model_config("any-model")

    def test_model_without_tokens(self):
        """Test model configuration without associated tokens."""
        model_only_grouped = {
            "test-model": [
                {
                    "name": "test-model",
                    "provider": "test",
                    "url": "https://test.api.com",
                    "weight": 50
                    # No llm_tokens field
                }
            ]
        }

        loader = MemoryModelConfigLoader(
            grouped_models=model_only_grouped,
            tokens=[]
        )

        # API returns list, get first element
        configs = loader.get_model_config("test-model")
        self.assertIsInstance(configs, list)
        self.assertGreater(len(configs), 0)
        config = configs[0]
        self.assertEqual(config.provider, "test")
        self.assertEqual(config.model, "test-model")
        self.assertEqual(config.api_url, "https://test.api.com")
        self.assertEqual(config.weight, 50)
        self.assertIsNone(config.api_key)  # Should be None when no token

    def test_weight_distribution(self):
        """Test that configs are sorted by weight correctly."""
        # Create loader with specific weights
        weighted_models = {
            "test-model": [
                {
                    "name": "test-model",
                    "provider": "provider-a",
                    "url": "https://a.com",
                    "weight": 70,
                    "llm_tokens": ["token-a"]
                },
                {
                    "name": "test-model",
                    "provider": "provider-b",
                    "url": "https://b.com",
                    "weight": 30,
                    "llm_tokens": ["token-b"]
                }
            ]
        }

        tokens = [
            {"name": "token-a", "token_config": {"api_key": "key-a"}},
            {"name": "token-b", "token_config": {"api_key": "key-b"}}
        ]

        loader = MemoryModelConfigLoader(
            grouped_models=weighted_models,
            tokens=tokens
        )

        # API returns sorted list by weight (highest first)
        configs = loader.get_model_config("test-model")
        self.assertIsInstance(configs, list)
        self.assertEqual(len(configs), 2)

        # Verify configs are sorted by weight (provider-a weight 70 should be first)
        self.assertEqual(configs[0].provider, "provider-a")
        self.assertEqual(configs[0].weight, 70)
        self.assertEqual(configs[0].api_key, "key-a")

        self.assertEqual(configs[1].provider, "provider-b")
        self.assertEqual(configs[1].weight, 30)
        self.assertEqual(configs[1].api_key, "key-b")

    def test_single_model_variant(self):
        """Test model with single variant (returns list with one element)."""
        single_model = {
            "claude-only": [
                {
                    "name": "claude-only",
                    "provider": "anthropic",
                    "url": "https://api.anthropic.com",
                    "weight": 100,
                    "llm_tokens": ["anthropic_token"]
                }
            ]
        }

        loader = MemoryModelConfigLoader(
            grouped_models=single_model,
            tokens=self.tokens
        )

        # Multiple calls should return the same list with one config
        for _ in range(10):
            configs = loader.get_model_config("claude-only")
            self.assertIsInstance(configs, list)
            self.assertEqual(len(configs), 1)
            config = configs[0]
            self.assertEqual(config.provider, "anthropic")
            self.assertEqual(config.model, "claude-only")
            self.assertEqual(config.weight, 100)


if __name__ == "__main__":
    unittest.main()