"""Simple HTTP template loader with permanent cache fallback."""

from __future__ import annotations

import asyncio
import time
import httpx
from urllib.parse import quote
import json
import os
from pathlib import Path

from ...template import PromptTemplate, Variant, ModelConfig
from .base import TemplateLoader, TemplateNotFoundError, VersionEntry
from ...logger import get_logger

logger = get_logger(__name__)


class HTTPLoaderSimpleCache(TemplateLoader):
    """HTTP template loader with simple permanent cache fallback."""

    def __init__(self, base_url: str, auth_token: str, client: httpx.AsyncClient | None = None, max_retries: int = 3) -> None:
        """Initialize with base_url for the template registry."""
        self.base_url = base_url.rstrip("/")
        self.client = client or httpx.AsyncClient(timeout=httpx.Timeout(30))
        self.sync_client = httpx.Client(timeout=httpx.Timeout(30))
        self.headers = {"Authorization": f"Bearer {auth_token}"}
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

    async def _retry_request_async(self, func, *args, **kwargs):
        """Execute an async HTTP request with exponential backoff retry."""
        last_exception = None
        for attempt in range(self.max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                logger.warning(f"HTTP request failed, attempt {attempt + 1}: {e}")
                last_exception = e
                if attempt < self.max_retries:
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
                    continue
                break
        raise last_exception

    def _retry_request_sync(self, func, *args, **kwargs):
        """Execute a sync HTTP request with exponential backoff retry."""
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

    async def alist_versions(self, name: str) -> list[VersionEntry]:
        """List all available versions for a template."""
        cache_key = f"versions_{name}"
        
        try:
            async def _get_versions():
                resp = await self.client.get(f"{self.base_url}/template/{name}/versions", headers=self.headers)
                resp.raise_for_status()
                return resp

            resp = await self._retry_request_async(_get_versions)
            versions_data = resp.json()
            
            # Save to cache
            self._save_cache(cache_key, versions_data)
            
            return [VersionEntry(id=str(v.get("version", "0")),
                               aliases=list(v.get("aliases", []))) for v in versions_data]
        except Exception as e:
            logger.warning(f"Failed to get versions for {name}, trying cache: {e}")
            # Try cache
            cached_data = self._load_cache(cache_key)
            if cached_data:
                logger.info(f"Using cached versions for {name}")
                return [VersionEntry(id=str(v.get("version", "0")),
                                   aliases=list(v.get("aliases", []))) for v in cached_data]
            return []

    async def aget_template(self, name: str, version: str) -> PromptTemplate:
        """Retrieve specific version of template."""
        cache_key = f"template_{name}_{version or 'max_version'}"
        
        try:
            if version:
                encoded_version = quote(version, safe='')
                url = f"{self.base_url}/template/{name}?label={encoded_version}"
            else:
                url = f"{self.base_url}/template/{name}"

            async def _get_template():
                resp = await self.client.get(url=url, headers=self.headers)
                if resp.status_code != 200:
                    raise TemplateNotFoundError(f"Template {name} version {version} not found")
                return resp

            try:
                resp = await self._retry_request_async(_get_template)
            except Exception as e:
                # If versioned URL fails and we have a version, try fallback to base URL
                if version:
                    logger.warning(f"Versioned URL failed for {name}:{version}, trying base URL: {e}")
                    fallback_url = f"{self.base_url}/template/{name}"
                    
                    async def _get_template_fallback():
                        resp = await self.client.get(url=fallback_url, headers=self.headers)
                        if resp.status_code != 200:
                            raise TemplateNotFoundError(f"Template {name} not found")
                        return resp
                    
                    resp = await self._retry_request_async(_get_template_fallback)
                else:
                    raise
            
            data = resp.json().get("data", {})
            
            # Save to cache
            self._save_cache(cache_key, data)
            
            template = self._build_template(data, name)
            logger.info(f"Loaded template {name}:{version} from server")
            return template
            
        except Exception as e:
            logger.warning(f"Failed to get template {name}:{version}, trying cache: {e}")
            # Try cache
            cached_data = self._load_cache(cache_key)
            if cached_data:
                logger.info(f"Using cached template {name}:{version}")
                return self._build_template(cached_data, name)
            return None

    def list_versions_sync(self, name: str) -> list[VersionEntry]:
        """Synchronous version of alist_versions."""
        cache_key = f"versions_{name}"
        
        try:
            def _get_versions():
                resp = self.sync_client.get(f"{self.base_url}/template/{name}/versions", headers=self.headers)
                resp.raise_for_status()
                return resp

            resp = self._retry_request_sync(_get_versions)
            versions_data = resp.json()
            
            # Save to cache
            self._save_cache(cache_key, versions_data)
            
            return [VersionEntry(id=str(v.get("version", "0")),
                               aliases=list(v.get("aliases", []))) for v in versions_data]
        except Exception as e:
            logger.warning(f"Failed to get versions for {name}, trying cache: {e}")
            # Try cache
            cached_data = self._load_cache(cache_key)
            if cached_data:
                logger.info(f"Using cached versions for {name}")
                return [VersionEntry(id=str(v.get("version", "0")),
                                   aliases=list(v.get("aliases", []))) for v in cached_data]
            return []

    def get_template_sync(self, name: str, version: str) -> PromptTemplate:
        """Synchronous version of aget_template."""
        cache_key = f"template_{name}_{version or 'max_version'}"
        
        try:
            if version:
                encoded_version = quote(version, safe='')
                url = f"{self.base_url}/template/{name}?label={encoded_version}"
            else:
                url = f"{self.base_url}/template/{name}"

            def _get_template():
                resp = self.sync_client.get(url=url, headers=self.headers)
                if resp.status_code != 200:
                    raise TemplateNotFoundError(f"Template {name} version {version} not found")
                return resp

            try:
                resp = self._retry_request_sync(_get_template)
            except Exception as e:
                # If versioned URL fails and we have a version, try fallback to base URL
                if version:
                    logger.warning(f"Versioned URL failed for {name}:{version}, trying base URL: {e}")
                    fallback_url = f"{self.base_url}/template/{name}"
                    
                    def _get_template_fallback():
                        resp = self.sync_client.get(url=fallback_url, headers=self.headers)
                        if resp.status_code != 200:
                            raise TemplateNotFoundError(f"Template {name} not found")
                        return resp
                    
                    resp = self._retry_request_sync(_get_template_fallback)
                else:
                    raise
            
            data = resp.json().get("data", {})
            
            # Save to cache
            self._save_cache(cache_key, data)
            
            template = self._build_template(data, name)
            real_version = data.get("version")
            logger.info(f"Loaded template {name}:{version} (real version {real_version}) from server")
            return template
            
        except Exception as e:
            logger.warning(f"Failed to get template {name}:{version}, trying cache: {e}")
            # Try cache
            cached_data = self._load_cache(cache_key)
            if cached_data:
                logger.info(f"Using cached template {name}:{version}")
                return self._build_template(cached_data, name)
            return None

    def _build_template(self, data: dict, name: str) -> PromptTemplate:
        """Build PromptTemplate from API data."""
        template_name = data.get("name")
        template_version = data.get("version")
        variants = data.get("variants", {})
        final_variants = {}
        
        for variant_name, variant in variants.items():
            model_cfg_dict = variant.get("model_cfg") or {}
            model_strategy_data = variant.get("model_strategy")
            
            # Compatibility handling
            if model_strategy_data is None and model_cfg_dict and isinstance(model_cfg_dict, dict):
                model_strategy_data = model_cfg_dict.get("model_strategy")
            
            # Filter out model_strategy field
            clean_model_cfg = {k: v for k, v in model_cfg_dict.items() if k != "model_strategy"}
            
            model_cfg = None
            if clean_model_cfg:
                model_cfg = ModelConfig(
                    provider=clean_model_cfg.get("provider"),
                    model=clean_model_cfg.get("model"),
                    api_key=clean_model_cfg.get("api_key"),
                    api_url=clean_model_cfg.get("api_url"),
                    temperature=clean_model_cfg.get("temperature"),
                    top_p=clean_model_cfg.get("top_p"),
                    max_tokens=clean_model_cfg.get("max_tokens"),
                    ext=clean_model_cfg.get("ext") or {}
                )
            
            final_variants[variant_name] = Variant(
                selector=variant.get("selector", []),
                model_cfg=model_cfg,
                model_strategy=model_strategy_data,
                messages=variant["messages_template"],
                required_variables=variant.get("required_variables") or [],
            )
        
        return PromptTemplate(
            id=data.get("template_id"),
            name=data.get("name", name),
            description="",
            version=template_version,
            aliases=list(data.get("aliases", [])),
            variant_router=data.get("variant_router"),
            variants=final_variants
        )