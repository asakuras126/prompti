"""Memory-based model configuration loader."""

from __future__ import annotations

from typing import List, Dict

from ...model_client.base import ModelConfig
from .base import ModelConfigLoader, ModelConfigNotFoundError
from ...logger import get_logger

logger = get_logger(__name__)


class MemoryModelConfigLoader(ModelConfigLoader):
    """Load model configurations from memory using grouped model format."""

    def __init__(self, grouped_models: Dict[str, List[dict]], tokens: List[dict],
                 reload_interval: int = 300) -> None:
        """Initialize the loader with grouped model data and token list.

        Args:
            grouped_models: Grouped model data (compatible with HTTPModelConfigLoader):
                {
                    "gpt-4": [
                        {
                            "name": "gpt-4",
                            "provider": "openai",
                            "url": "https://api.openai.com/v1",
                            "weight": 60,
                            "llm_tokens": ["openai_token"]
                        }
                    ]
                }
            tokens: Token data:
                [
                    {
                        "name": "openai_token",
                        "token_config": {
                            "api_key": "sk-..."
                        }
                    }
                ]
            reload_interval: Reload interval in seconds (default: 300)
        """
        super().__init__(reload_interval)
        
        self.grouped_models = grouped_models
        self.tokens = tokens or []
        self.models: Dict[str, List[ModelConfig]] = {}
        self.raw_model_data: Dict[str, List[dict]] = {}  # 保存原始模型数据用于token匹配
        self._do_load()

    def _do_load(self):
        """Load model configurations from memory using grouped model data."""
        try:
            self._load_grouped_format()
        except Exception as e:
            logger.error(f"Error loading model configs from memory: {e}")

    def _load_grouped_format(self):
        """Load model configurations from grouped format data."""
        # Create token lookup dictionary
        token_dict = {token["name"]: token for token in self.tokens}

        new_models = {}
        new_raw_data = {}
        # grouped_models is in grouped format: {"model_name": [model1, model2, ...]}
        for model_name, model_variants in self.grouped_models.items():
            # Create ModelConfig list for each model name
            model_configs = []
            raw_models = []
            for model in model_variants:
                model_config = ModelConfig(
                    provider=model.get("provider", ""),
                    model=model_name,  # 聚合模型名称（来自grouping的key）
                    model_value=model.get("value", model.get("name", model_name)),  # 真实调用的模型名称
                    api_url=model.get("url", ""),
                    weight=model.get("weight", 50),
                )

                # Associate API key from token list if available
                if model.get("llm_tokens"):
                    llm_token_name = model.get("llm_tokens")[0]
                    if llm_token_name in token_dict:
                        token_config = token_dict[llm_token_name].get('token_config', {})
                        model_config.api_key = token_config.get("api_key", "")

                model_configs.append(model_config)
                raw_models.append(model)  # 保存原始数据
            
            new_models[model_name] = model_configs
            new_raw_data[model_name] = raw_models

        self.models = new_models
        self.raw_model_data = new_raw_data

    def get_model_config(self, model: str, provider: str = None, model_value: str = None, llm_token: str = None, model_control=None) -> List[ModelConfig]:
        """Get ordered list of model configs based on priority.
        
        Args:
            model: Required. The aggregated model name to look up.
            provider: Optional. Filter by provider name. If provided, configs with this provider will be prioritized.
            model_value: Optional. Filter by actual model name (real API model name).
            llm_token: Optional. Filter by associated LLM token name.
            model_control: Optional. ModelControlParams for dynamic weight/enable/disable control.
            
        Returns:
            List[ModelConfig]: Ordered list of model configurations sorted by priority.
                              If provider is specified, matching providers come first.
                              Within each group, configs are sorted by weight (highest first).
            
        Raises:
            ModelConfigNotFoundError: If no matching configuration is found.
        """
        if model not in self.models:
            raise ModelConfigNotFoundError(model)
        
        # Get all configs for this model
        all_configs = self.models[model]
        all_raw_data = self.raw_model_data.get(model, [])
        
        # Apply basic filtering first (exclude provider filter for now, will be used for prioritization)
        filtered_configs = self._filter_configs_by_criteria(all_configs, None, model_value)
        
        # Apply llm_token filtering if specified
        if llm_token and filtered_configs:
            # Need to filter based on raw model data since token info isn't in ModelConfig
            filtered_indices = []
            for i, raw_model in enumerate(all_raw_data):
                # Check if this config is still in our filtered list
                if i < len(all_configs) and all_configs[i] in filtered_configs:
                    # Check if llm_token matches
                    model_tokens = raw_model.get("llm_tokens", [])
                    if llm_token in model_tokens:
                        filtered_indices.append(i)
            
            # Update filtered_configs to only include those with matching tokens
            filtered_configs = [all_configs[i] for i in filtered_indices]
        
        if not filtered_configs:
            # Create a more descriptive error message
            criteria = [f"model='{model}'"]
            if provider:
                criteria.append(f"provider='{provider}'")
            if model_value:
                criteria.append(f"model_value='{model_value}'")
            if llm_token:
                criteria.append(f"llm_token='{llm_token}'")
            raise ModelConfigNotFoundError(f"No model config found matching criteria: {', '.join(criteria)}")
        
        # Apply model control filters before final selection
        if model_control:
            # Apply model control filters (enabled/disabled models)
            final_filtered_configs = self._apply_model_control_filters(filtered_configs, model_control)
            if not final_filtered_configs:
                # Create a more descriptive error message
                criteria = [f"model='{model}'"]
                if provider:
                    criteria.append(f"provider='{provider}'")
                if model_value:
                    criteria.append(f"model_value='{model_value}'")
                if llm_token:
                    criteria.append(f"llm_token='{llm_token}'")
                raise ModelConfigNotFoundError(f"No model config found matching criteria after applying model control: {', '.join(criteria)}")
            filtered_configs = final_filtered_configs
        
        # Sort configs by priority: preferred provider first, then by weight
        sorted_configs = self._sort_configs_by_priority(filtered_configs, provider, model_control)
        
        return sorted_configs

    def list_models(self) -> List[str]:
        """List all available model names."""
        return list(self.models.keys())