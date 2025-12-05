"""HTTP-based model configuration loader."""

from __future__ import annotations

import time
from typing import List, Dict

import httpx

from ...model_client.base import ModelConfig
from .base import ModelConfigLoader, ModelConfigNotFoundError
from ...logger import get_logger

logger = get_logger(__name__)


class HTTPModelConfigLoader(ModelConfigLoader):
    """Fetch model configurations from an HTTP endpoint returning JSON."""

    def __init__(self, url: str, client: httpx.Client | None = None, registry_api_key: str=None, reload_interval: int = 300, max_retries: int = 3) -> None:
        """Initialize the loader with an HTTP endpoint returning JSON."""
        super().__init__(reload_interval)
        self.base_url = url
        self.client = client or httpx.Client(timeout=httpx.Timeout(30))
        self.models: Dict[str, List[ModelConfig]] = {}
        self.raw_model_data: Dict[str, List[dict]] = {}  # 保存原始模型数据用于token匹配
        self.registry_api_key = registry_api_key
        self.max_retries = max_retries

    def _retry_request(self, func, *args, **kwargs):
        """Execute an HTTP request with exponential backoff retry."""
        last_exception = None
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                logger.warning(f"http model config loader get http error， retry: {attempt + 1}")
                last_exception = e
                if attempt < self.max_retries:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    time.sleep(wait_time)
                    continue
                break
        # If all retries failed, raise the last exception
        raise last_exception

    def _do_load(self):
        """Fetch model configurations from an HTTP endpoint and store in memory."""
        try:
            headers = {
                "Authorization": f"Bearer {self.registry_api_key}"
            }
            
            # Get model list with retry
            def _get_models():
                model_list_url = f"{self.base_url}/model/grouped"
                resp = self.client.get(url=model_list_url, headers=headers)
                resp.raise_for_status()
                return resp
            
            model_resp = self._retry_request(_get_models)
            logger.debug(f"get model response: {model_resp.text}")
            model_list_data = model_resp.json().get("data") or []

            # Get token list with retry
            def _get_tokens():
                model_token_url = f"{self.base_url}/llm-token/list"
                resp = self.client.get(url=model_token_url, headers=headers)
                resp.raise_for_status()
                return resp
            
            token_resp = self._retry_request(_get_tokens)
            logger.debug(f"get token response: {token_resp.text}")
            token_list_data = token_resp.json().get("data") or []
            token_dict = {token["name"]: token for token in token_list_data}
            
            new_models = {}
            new_raw_data = {}
            # model_list_data 现在是分组格式: {"model_name": [model1, model2, ...]}
            for model_name, model_variants in model_list_data.items():
                # 为每个模型名称创建ModelConfig列表
                model_configs = []
                raw_models = []
                for model in model_variants:
                    model_ext = model.get("ext") or {}
                    model_config = ModelConfig(
                        provider=model["provider"],
                        model=model_name,  # 聚合模型名称（来自grouping的key）
                        model_value=model.get("value", model["name"]),  # 真实调用的模型名称
                        api_url=model["url"],
                        weight=model_ext.get("weight", 50),
                        ext=model_ext,  # 直接传递完整的ext字典
                    )
                    if model.get("llm_tokens"):
                        llm_token_name = model.get("llm_tokens")[0]
                        if llm_token_name in token_dict:
                            token_data = token_dict[llm_token_name]
                            model_config.token_config = token_data.get('token_config', {})
                            model_config.api_key = model_config.token_config.get("api_key", "")
                            logger.debug(f"Loaded token_config for model {model_name} from token {llm_token_name}: {model_config.token_config}")
                        else:
                            logger.warning(f"Token '{llm_token_name}' not found in token_dict for model {model_name}")
                    model_configs.append(model_config)
                    raw_models.append(model)  # 保存原始数据
                new_models[model_name] = model_configs
                new_raw_data[model_name] = raw_models
            
            self.models = new_models
            self.raw_model_data = new_raw_data
            logger.debug(f"get models: {self.models}")
        except Exception as e:
            raise


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
        self.load()  # Ensure models are loaded
        return list(self.models.keys())