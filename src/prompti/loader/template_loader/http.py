"""Fetch prompt templates from a remote HTTP service."""

from __future__ import annotations

import asyncio
import time
import httpx
from urllib.parse import quote

from ...template import PromptTemplate, Variant, ModelConfig
from .base import TemplateLoader, TemplateNotFoundError, VersionEntry
from ...logger import get_logger

logger = get_logger(__name__)

class HTTPLoader(TemplateLoader):
    """Fetch templates from an HTTP endpoint."""

    def __init__(self, base_url: str, auth_token: str, client: httpx.AsyncClient | None = None, max_retries: int = 3) -> None:
        """Initialize with ``base_url`` for the template registry."""
        self.base_url = base_url.rstrip("/")
        self.client = client or httpx.AsyncClient(timeout=httpx.Timeout(30))
        self.sync_client = httpx.Client(timeout=httpx.Timeout(30))
        self.headers = {"Authorization": f"Bearer {auth_token}"}
        self.max_retries = max_retries

    async def _retry_request_async(self, func, *args, **kwargs):
        """Execute an async HTTP request with exponential backoff retry."""
        last_exception = None
        for attempt in range(self.max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                logger.warning(f"http template loader get http error， retry: {attempt + 1}")
                last_exception = e
                if attempt < self.max_retries:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    await asyncio.sleep(wait_time)
                    continue
                break
        # If all retries failed, raise the last exception
        raise last_exception

    def _retry_request_sync(self, func, *args, **kwargs):
        """Execute a sync HTTP request with exponential backoff retry."""
        last_exception = None
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                logger.warning(f"http template loader get http error， retry: {attempt + 1}")
                last_exception = e
                if attempt < self.max_retries:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    time.sleep(wait_time)
                    continue
                break
        # If all retries failed, raise the last exception
        raise last_exception

    async def alist_versions(self, name: str) -> list[VersionEntry]:
        """List all available versions for a template from HTTP endpoint."""
        try:
            async def _get_versions():
                resp = await self.client.get(f"{self.base_url}/template/{name}/versions", headers=self.headers)
                resp.raise_for_status()
                return resp
            
            resp = await self._retry_request_async(_get_versions)
            if resp.status_code != 200:
                return []

            versions_data = resp.json()
            return [VersionEntry(id=str(v.get("version", "0")),
                                 aliases=list(v.get("aliases", []))) for v in versions_data]
        except (httpx.RequestError, ValueError, KeyError):
            return []

    async def aget_template(self, name: str, version: str) -> PromptTemplate:
        """Retrieve specific version of template from the remote registry."""
        try:
            if version:
                # URL编码version参数，特别是将#替换为%23
                encoded_version = quote(version, safe='')
                url = f"{self.base_url}/template/{name}?label={encoded_version}"
            else:
                url = f"{self.base_url}/template/{name}"
            
            async def _get_template():
                resp = await self.client.get(url=url, headers=self.headers)
                if resp.status_code != 200:
                    raise TemplateNotFoundError(
                        f"Template {name} version {version} not found"
                    )
                return resp
            
            resp = await self._retry_request_async(_get_template)
            data = resp.json()
            data = data.get("data", {})
            template_name = data.get("name")
            template_version = data.get("version")
            logger.info(f"get template {template_name}, version: {template_version} from promptstore")
            variants = data.get("variants", {})
            final_variants = {}
            for variant_name, variant in variants.items():
                model_cfg_dict = variant.get("model_cfg") or {}
                model_strategy_data = variant.get("model_strategy")
                
                # 兼容性处理：如果model_strategy为空，但model_cfg中包含model_strategy，则从那里获取
                if model_strategy_data is None and model_cfg_dict and isinstance(model_cfg_dict, dict):
                    model_strategy_data = model_cfg_dict.get("model_strategy")
                
                # 过滤掉model_strategy字段，避免传递给ModelConfig
                clean_model_cfg = {k: v for k, v in model_cfg_dict.items() if k != "model_strategy"}
                
                model_cfg = None
                if clean_model_cfg:  # 只有在有其他配置时才创建ModelConfig
                    model_cfg = ModelConfig(
                        provider=clean_model_cfg.get("provider"),
                        model=clean_model_cfg.get("model"),
                        api_key=clean_model_cfg.get("api_key"),
                        api_url=clean_model_cfg.get("api_url"),
                        temperature=clean_model_cfg.get("temperature"),
                        top_p=clean_model_cfg.get("top_p"),
                        max_tokens=clean_model_cfg.get("max_tokens"),
                    )
                
                final_variants[variant_name] = Variant(
                    selector=variant.get("selector", []),
                    model_cfg=model_cfg,
                    model_strategy=model_strategy_data,
                    messages=variant["messages_template"],
                    required_variables=variant.get("required_variables") or [],
                )
            tmpl = PromptTemplate(
                id=data.get("template_id"),
                name=data.get("name", name),
                description="",
                version=template_version,
                aliases=list(data.get("aliases", [])),
                variant_router=data.get("variant_router"),
                variants=final_variants,
            )
            return tmpl
        except Exception as e:
            logger.warning(
                f"Template {name} version {version} not found: {str(e)}"
            )
            return None

    def list_versions_sync(self, name: str) -> list[VersionEntry]:
        """Synchronous version of alist_versions."""
        try:
            def _get_versions():
                resp = self.sync_client.get(f"{self.base_url}/template/{name}/versions", headers=self.headers)
                resp.raise_for_status()
                return resp
            
            resp = self._retry_request_sync(_get_versions)
            if resp.status_code != 200:
                return []

            versions_data = resp.json()
            return [VersionEntry(id=str(v.get("version", "0")),
                                 aliases=list(v.get("aliases", []))) for v in versions_data]
        except (httpx.RequestError, ValueError, KeyError):
            return []

    def get_template_sync(self, name: str, version: str) -> PromptTemplate:
        """Synchronous version of aget_template."""
        try:
            if version:
                # URL编码version参数，特别是将#替换为%23
                encoded_version = quote(version, safe='')
                url = f"{self.base_url}/template/{name}?label={encoded_version}"
            else:
                url = f"{self.base_url}/template/{name}"
            
            def _get_template():
                resp = self.sync_client.get(url=url, headers=self.headers)
                if resp.status_code != 200:
                    raise TemplateNotFoundError(
                        f"Template {name} version {version} not found"
                    )
                return resp
            
            resp = self._retry_request_sync(_get_template)
            data = resp.json()
            data = data.get("data", {})
            template_name = data.get("name")
            template_version = data.get("version")
            logger.info(f"get template {template_name}, version: {template_version} from promptstore, url: {url}")
            variants = data.get("variants", {})
            final_variants = {}
            for variant_name, variant in variants.items():
                model_cfg_dict = variant.get("model_cfg") or {}
                model_strategy_data = variant.get("model_strategy")
                
                # 兼容性处理：如果model_strategy为空，但model_cfg中包含model_strategy，则从那里获取
                if model_strategy_data is None and model_cfg_dict and isinstance(model_cfg_dict, dict):
                    model_strategy_data = model_cfg_dict.get("model_strategy")
                
                # 过滤掉model_strategy字段，避免传递给ModelConfig
                clean_model_cfg = {k: v for k, v in model_cfg_dict.items() if k != "model_strategy"}
                
                model_cfg = None
                if clean_model_cfg:  # 只有在有其他配置时才创建ModelConfig
                    model_cfg = ModelConfig(
                        provider=clean_model_cfg.get("provider"),
                        model=clean_model_cfg.get("model"),
                        api_key=clean_model_cfg.get("api_key"),
                        api_url=clean_model_cfg.get("api_url"),
                        temperature=clean_model_cfg.get("temperature"),
                        top_p=clean_model_cfg.get("top_p"),
                        max_tokens=clean_model_cfg.get("max_tokens"),
                    )
                
                final_variants[variant_name] = Variant(
                    selector=variant.get("selector", []),
                    model_cfg=model_cfg,
                    model_strategy=model_strategy_data,
                    messages=variant["messages_template"],
                    required_variables=variant.get("required_variables") or [],
                )
            tmpl = PromptTemplate(
                id=data.get("template_id"),
                name=data.get("name", name),
                description="",
                version=template_version,
                aliases=list(data.get("aliases", [])),
                variant_router=data.get("variant_router"),
                variants=final_variants,
            )
            return tmpl
        except Exception as e:
            logger.warning(
                f"Template {name} version {version} not found: {str(e)}"
            )
            return None
