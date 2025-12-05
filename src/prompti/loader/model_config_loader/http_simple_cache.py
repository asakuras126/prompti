"""Simple HTTP model config loader with permanent cache fallback."""

from __future__ import annotations

import time
import json
from pathlib import Path
from typing import List, Dict

import httpx

from ...model_client.base import ModelConfig
from .base import ModelConfigLoader, ModelConfigNotFoundError
from ...logger import get_logger

logger = get_logger(__name__)


class HTTPModelConfigLoaderSimpleCache(ModelConfigLoader):
    """HTTP model config loader with simple permanent cache fallback."""

    def __init__(self, url: str, client: httpx.Client | None = None, registry_api_key: str = None, 
                 reload_interval: int = 300, max_retries: int = 3) -> None:
        """Initialize the loader with an HTTP endpoint."""
        super().__init__(reload_interval)
        self.base_url = url
        self.client = client or httpx.Client(timeout=httpx.Timeout(30))
        self.models: Dict[str, List[ModelConfig]] = {}
        self.raw_model_data: Dict[str, List[dict]] = {}
        self.registry_api_key = registry_api_key
        self.max_retries = max_retries
        self.cache_dir = Path(".prompti_cache")
        self.cache_dir.mkdir(exist_ok=True)

    def _get_cache_file(self, key: str) -> Path:
        """Get cache file path for a key."""
        return self.cache_dir / f"{key.replace(':', '_').replace('/', '_')}.json"

    def _save_cache(self, key: str, data: dict):
        """Save data to cache."""
        try:
            cache_file = self._get_cache_file(key)
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache for {key}: {e}")

    def _load_cache(self, key: str) -> dict | None:
        """Load data from cache."""
        try:
            cache_file = self._get_cache_file(key)
            if cache_file.exists():
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache for {key}: {e}")
        return None

    def _retry_request(self, func, *args, **kwargs):
        """Execute an HTTP request with exponential backoff retry."""
        last_exception = None
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                logger.warning(f"HTTP request failed, attempt {attempt + 1}: {e}")
                last_exception = e
                if attempt < self.max_retries:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                    continue
                break
        raise last_exception

    def _do_load(self):
        """Fetch model configurations from HTTP endpoint."""
        cache_key = "model_configs"
        
        try:
            headers = {"Authorization": f"Bearer {self.registry_api_key}"}
            
            # Get model list
            def _get_models():
                model_list_url = f"{self.base_url}/model/grouped"
                resp = self.client.get(url=model_list_url, headers=headers)
                resp.raise_for_status()
                return resp

            model_resp = self._retry_request(_get_models)
            model_list_data = model_resp.json().get("data") or []

            # Get token list
            def _get_tokens():
                model_token_url = f"{self.base_url}/llm-token/list"
                resp = self.client.get(url=model_token_url, headers=headers)
                resp.raise_for_status()
                return resp

            token_resp = self._retry_request(_get_tokens)
            token_list_data = token_resp.json().get("data") or []
            
            # Save to cache
            cache_data = {
                "model_list_data": model_list_data,
                "token_list_data": token_list_data,
                "timestamp": time.time()
            }
            self._save_cache(cache_key, cache_data)
            
            self._process_model_data(model_list_data, token_list_data)
            logger.debug(f"Loaded {len(self.models)} model configs from server")
            
        except Exception as e:
            logger.warning(f"Failed to load model configs from server, trying cache: {e}")
            # Try cache
            cached_data = self._load_cache(cache_key)
            if cached_data:
                logger.info("Using cached model configs")
                model_list_data = cached_data.get("model_list_data", [])
                token_list_data = cached_data.get("token_list_data", [])
                self._process_model_data(model_list_data, token_list_data)
            else:
                logger.error("No cached model configs available")
                raise

    def _process_model_data(self, model_list_data: dict, token_list_data: list):
        """Process model and token data."""
        token_dict = {token["name"]: token for token in token_list_data}
        
        new_models = {}
        new_raw_data = {}
        
        for model_name, model_variants in model_list_data.items():
            model_configs = []
            raw_models = []
            for model in model_variants:
                model_ext = model.get("ext") or {}
                model_config = ModelConfig(
                    provider=model["provider"],
                    model=model_name,
                    model_value=model.get("value", model["name"]),
                    api_url=model["url"],
                    weight=model_ext.get("weight", 50),
                    ext=model_ext,
                )
                if model.get("llm_tokens"):
                    llm_token_name = model.get("llm_tokens")[0]
                    if llm_token_name in token_dict:
                        model_config.api_key = token_dict[llm_token_name].get('token_config', {}).get("api_key", "")
                        model_config.token_config = token_dict[llm_token_name].get('token_config', {})
                model_configs.append(model_config)
                raw_models.append(model)
            new_models[model_name] = model_configs
            new_raw_data[model_name] = raw_models
        
        self.models = new_models
        self.raw_model_data = new_raw_data

    def get_model_config(self, model: str, provider: str = None, model_value: str = None, 
                        llm_token: str = None, model_control=None) -> List[ModelConfig]:
        """Get ordered list of model configs based on priority."""
        self.load()  # Check if reload is needed
        
        if model not in self.models:
            raise ModelConfigNotFoundError(model)
        
        all_configs = self.models[model]
        all_raw_data = self.raw_model_data.get(model, [])
        
        # Apply basic filtering
        filtered_configs = self._filter_configs_by_criteria(all_configs, None, model_value)
        
        # Apply llm_token filtering if specified
        if llm_token and filtered_configs:
            filtered_indices = []
            for i, raw_model in enumerate(all_raw_data):
                if i < len(all_configs) and all_configs[i] in filtered_configs:
                    model_tokens = raw_model.get("llm_tokens", [])
                    if llm_token in model_tokens:
                        filtered_indices.append(i)
            filtered_configs = [all_configs[i] for i in filtered_indices]
        
        if not filtered_configs:
            criteria = [f"model='{model}'"]
            if provider:
                criteria.append(f"provider='{provider}'")
            if model_value:
                criteria.append(f"model_value='{model_value}'")
            if llm_token:
                criteria.append(f"llm_token='{llm_token}'")
            raise ModelConfigNotFoundError(f"No model config found matching criteria: {', '.join(criteria)}")
        
        # Apply model control filters
        if model_control:
            final_filtered_configs = self._apply_model_control_filters(filtered_configs, model_control)
            if not final_filtered_configs:
                raise ModelConfigNotFoundError(f"No model config found after applying model control")
            filtered_configs = final_filtered_configs
        
        # Sort configs by priority
        sorted_configs = self._sort_configs_by_priority(filtered_configs, provider, model_control)
        
        return sorted_configs

    def list_models(self) -> List[str]:
        """List all available model names."""
        self.load()
        return list(self.models.keys())