"""Base classes for model configuration loaders."""

from __future__ import annotations

import threading
import time
import random
from abc import ABC, abstractmethod
from typing import List, Dict

from ...model_client.base import ModelConfig


class ModelConfigNotFoundError(Exception):
    """Raised when a model configuration is not found."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        super().__init__(f"Model configuration '{model_name}' not found")


class ModelConfigLoader(ABC):
    """Base class for loaders that return a :class:`ModelConfig`."""

    models: Dict[str, List[ModelConfig]] = {}
    
    def __init__(self, reload_interval: int = 300) -> None:
        self._last_loaded = 0
        self._reload_interval = reload_interval
        self._lock = threading.Lock()

    @abstractmethod
    def _do_load(self):
        """Internal method to perform the actual loading."""
        raise NotImplementedError
    
    def load(self):
        """Load model configurations with automatic reload every 5 minutes."""
        current_time = time.time()
        
        with self._lock:
            if current_time - self._last_loaded >= self._reload_interval:
                self._do_load()
                self._last_loaded = current_time

    def _select_config_by_weight(self, configs: List[ModelConfig], model_control=None) -> ModelConfig:
        """Select a config based on weighted random selection with dynamic control."""
        if not configs:
            raise ValueError("No configs available for selection")
        
        # Apply dynamic model control filters
        filtered_configs = self._apply_model_control_filters(configs, model_control)
        
        if not filtered_configs:
            raise ValueError("No enabled configs available after applying model control filters")
        
        if len(filtered_configs) == 1:
            return filtered_configs[0]
        
        # Apply dynamic weight overrides
        configs_with_weights = self._apply_weight_overrides(filtered_configs, model_control)
        
        # Calculate total weight
        total_weight = sum(weight for _, weight in configs_with_weights)
        
        if total_weight <= 0:
            # If all weights are 0 or negative, use simple random selection
            return random.choice([config for config, _ in configs_with_weights])
        
        # Generate random number between 0 and total_weight
        random_weight = random.uniform(0, total_weight)
        
        # Select config based on cumulative weight
        cumulative_weight = 0
        for config, weight in configs_with_weights:
            cumulative_weight += weight
            if random_weight <= cumulative_weight:
                return config
        
        # Fallback to last config (should not reach here)
        return configs[-1]

    def _filter_configs_by_criteria(self, configs: List[ModelConfig], provider: str = None, 
                                   model_value: str = None, llm_token: str = None) -> List[ModelConfig]:
        """Filter model configurations based on multiple criteria.
        
        Args:
            configs: List of ModelConfig objects to filter.
            provider: Optional. Filter by provider name.
            model_value: Optional. Filter by actual model name.
            llm_token: Optional. Filter by associated LLM token name.
            
        Returns:
            List[ModelConfig]: Filtered configurations that match all provided criteria.
        """
        filtered_configs = configs.copy()
        
        # Filter by provider
        if provider:
            filtered_configs = [cfg for cfg in filtered_configs if cfg.provider == provider]
        
        # Filter by model_value (actual model name)
        if model_value:
            filtered_configs = [cfg for cfg in filtered_configs if cfg.get_actual_model_name() == model_value]
        
        # Filter by llm_token - need to check in the raw data
        # This will be implemented differently in each loader since token info isn't in ModelConfig
        
        return filtered_configs

    def _apply_model_control_filters(self, configs: List[ModelConfig], model_control) -> List[ModelConfig]:
        """Apply dynamic model control filters (enabled/disabled models).
        
        Args:
            configs: List of ModelConfig objects to filter.
            model_control: ModelControlParams object containing filter rules.
            
        Returns:
            List[ModelConfig]: Filtered configurations after applying enable/disable rules.
        """
        if not model_control:
            return configs
        
        from ...model_client.base import ModelControlParams
        
        # Validate input type
        if not isinstance(model_control, ModelControlParams):
            return configs
        
        filtered_configs = configs.copy()
        
        # Apply enabled_models filter (if specified, only these will be used)
        if model_control.enabled_models:
            enabled_configs = []
            for config in filtered_configs:
                provider_key = config.provider
                provider_model_key = f"{config.provider}/{config.get_aggregated_model_name()}"
                
                # Check if this config matches any enabled pattern
                if (provider_key in model_control.enabled_models or 
                    provider_model_key in model_control.enabled_models or
                    config.get_aggregated_model_name() in model_control.enabled_models):
                    enabled_configs.append(config)
            
            filtered_configs = enabled_configs
        
        # Apply disabled_models filter  
        if model_control.disabled_models and filtered_configs:
            non_disabled_configs = []
            for config in filtered_configs:
                provider_key = config.provider
                provider_model_key = f"{config.provider}/{config.get_aggregated_model_name()}"
                
                # Check if this config is NOT in the disabled list
                if (provider_key not in model_control.disabled_models and 
                    provider_model_key not in model_control.disabled_models and
                    config.get_aggregated_model_name() not in model_control.disabled_models):
                    non_disabled_configs.append(config)
            
            filtered_configs = non_disabled_configs
        
        return filtered_configs

    def _apply_weight_overrides(self, configs: List[ModelConfig], model_control) -> List[tuple[ModelConfig, int]]:
        """Apply dynamic weight overrides.
        
        Args:
            configs: List of ModelConfig objects.
            model_control: ModelControlParams object containing weight overrides.
            
        Returns:
            List[tuple[ModelConfig, int]]: List of tuples containing (config, effective_weight).
        """
        from ...model_client.base import ModelControlParams
        
        configs_with_weights = []
        
        for config in configs:
            original_weight = config.weight or 50
            
            if (model_control and 
                isinstance(model_control, ModelControlParams) and 
                model_control.weight_overrides):
                
                # Check for weight overrides in order of specificity:
                # 1. provider/model specific (e.g., "openai/gpt-4")
                # 2. provider specific (e.g., "openai")
                # 3. model specific (e.g., "gpt-4")
                provider_model_key = f"{config.provider}/{config.get_aggregated_model_name()}"
                provider_key = config.provider
                model_key = config.get_aggregated_model_name()
                
                if provider_model_key in model_control.weight_overrides:
                    weight = model_control.weight_overrides[provider_model_key]
                elif provider_key in model_control.weight_overrides:
                    weight = model_control.weight_overrides[provider_key]
                elif model_key in model_control.weight_overrides:
                    weight = model_control.weight_overrides[model_key]
                else:
                    weight = original_weight
            else:
                weight = original_weight
                
            configs_with_weights.append((config, weight))
        
        return configs_with_weights

    def _sort_configs_by_priority(self, configs: List[ModelConfig], preferred_provider: str = None, model_control=None) -> List[ModelConfig]:
        """Sort configs by priority: preferred provider first, then by weight (highest first).
        
        Args:
            configs: List of ModelConfig objects to sort.
            preferred_provider: Optional. Provider to prioritize.
            model_control: Optional. ModelControlParams for dynamic weight control.
            
        Returns:
            List[ModelConfig]: Sorted list of configurations.
        """
        if not configs:
            return configs
        
        # Apply weight overrides first
        configs_with_weights = self._apply_weight_overrides(configs, model_control)
        
        # Separate configs by provider preference
        preferred_configs = []
        other_configs = []
        
        for config, weight in configs_with_weights:
            if preferred_provider and config.provider == preferred_provider:
                preferred_configs.append((config, weight))
            else:
                other_configs.append((config, weight))
        
        # Sort each group by weight (highest first)
        preferred_configs.sort(key=lambda x: x[1], reverse=True)
        other_configs.sort(key=lambda x: x[1], reverse=True)
        
        # Combine results: preferred provider first, then others
        result = [config for config, _ in preferred_configs] + [config for config, _ in other_configs]
        
        return result

    @abstractmethod
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
        raise NotImplementedError