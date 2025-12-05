"""File-based model configuration loader."""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict

import yaml

from ...model_client.base import ModelConfig
from .base import ModelConfigLoader, ModelConfigNotFoundError


class FileModelConfigLoader(ModelConfigLoader):
    """Load model configurations from a local YAML or JSON file."""

    def __init__(self, path: str | Path = "./configs/models.yaml", reload_interval: int = 300) -> None:
        """Initialize the loader with a path to a local YAML or JSON file."""
        super().__init__(reload_interval)
        self.path = Path(path)
        self.models: Dict[str, List[ModelConfig]] = {}

    def _do_load(self):
        """Load model configurations from a local YAML or JSON file and store in memory."""
        if not self.path.exists():
            raise FileNotFoundError(f"Config file not found: {self.path}")

        text = self.path.read_text()
        data = yaml.safe_load(text)

        if not isinstance(data, dict):
            raise ValueError("Config file must contain a mapping")

        # 支持两种格式：单个模型配置或多个模型配置
        if "models" in data:
            # 多模型配置格式
            models_data = data["models"]
            if not isinstance(models_data, list):
                raise ValueError("'models' field must be a list")

            new_models = {}
            for model_data in models_data:
                if isinstance(model_data, dict):
                    model_config = ModelConfig(**model_data)
                    model_name = model_config.model
                    if model_name not in new_models:
                        new_models[model_name] = []
                    new_models[model_name].append(model_config)
            self.models = new_models
        else:
            self.models = {}

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
        self.load()  # Check if reload is needed
        
        if model in self.models:
            # Apply basic filtering first (exclude provider filter for now, will be used for prioritization)
            filtered_configs = self._filter_configs_by_criteria(self.models[model], None, model_value)
            
            if not filtered_configs:
                # Create a more descriptive error message
                criteria = [f"model='{model}'"]
                if provider:
                    criteria.append(f"provider='{provider}'")
                if model_value:
                    criteria.append(f"model_value='{model_value}'")
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
                    raise ModelConfigNotFoundError(f"No model config found matching criteria after applying model control: {', '.join(criteria)}")
                filtered_configs = final_filtered_configs
            
            # Sort configs by priority: preferred provider first, then by weight
            sorted_configs = self._sort_configs_by_priority(filtered_configs, provider, model_control)
            
            return sorted_configs

        raise ModelConfigNotFoundError(model)

    def list_models(self) -> List[str]:
        """List all available model names."""
        self.load()  # Ensure models are loaded
        return list(self.models.keys())