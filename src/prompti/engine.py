"""Core engine that resolves templates and executes them with model clients."""

from __future__ import annotations

import asyncio
import json
import os
import hashlib
import threading

import uuid
from functools import lru_cache
from abc import ABC, abstractmethod
from collections import OrderedDict

import yaml
from collections.abc import AsyncGenerator, Generator, Callable, Awaitable, Iterator, AsyncIterator
from typing import Union
from pathlib import Path
from typing import Any, cast, ClassVar, Protocol
import time

from async_lru import alru_cache
from opentelemetry import trace
from pydantic import BaseModel, ConfigDict

from .loader import (
    FileLoader,
    MemoryLoader,
    HTTPLoader,
    TemplateLoader,
    TemplateNotFoundError,
)
# 导入简单缓存版本的HTTP Loader
from .loader.template_loader.http_simple_cache import HTTPLoaderSimpleCache
from .message import Message, ModelResponse, StreamingModelResponse, StreamingChoice, Choice
from .trace import TraceService, TraceEvent, _concatenate_streaming_responses, restore_original_urls_in_messages
from .tool_trace import ToolTraceService
from .otel import init_llm_metrics, get_llm_metrics, shutdown_llm_metrics
from .model_client import ModelConfig, RunParams, ToolParams, ToolSpec
from .model_client.factory import create_client
from .model_client.base import should_retry_error
from .loader import ModelConfigLoader, FileModelConfigLoader, \
    HTTPModelConfigLoader, ModelConfigNotFoundError, MemoryModelConfigLoader
# 导入简单缓存版本的Model Config Loader
from .loader.model_config_loader.http_simple_cache import HTTPModelConfigLoaderSimpleCache
from .template import PromptTemplate
from .hooks import HookResult, BeforeRunHook, AfterRunHook

from .logger import get_logger, set_logger, LoggerProtocol

logger = get_logger(__name__)

_tracer = trace.get_tracer(__name__)


# def _get_simple_variant_route_rules():
#     return [
#         # 规则1: 黑名单 → primary
#         {
#             "name": "white_list",
#             "conditions": {
#                 "condition_list": [
#                     {
#                         "type": "list",
#                         "field": "user_id",
#                         "allow": ["blacklist_user_001", "blacklist_user_002"],
#                     }
#                 ]
#             },
#             "candidates": [{"name": "primary", "weight": 1}],
#             "selector": {"type": "weighted_round_robin"},
#         },

#         # 规则2: 满足所有条件 → hash路由
#         {
#             "name": "coding_routing",
#             "conditions": {
#                 "condition_list": [
#                     {"type": "value", "field": "query_type", "values": ["coding"]},
#                     {"type": "bool", "field": "mmu", "expected": False},
#                     {"type": "value", "field": "turn", "values": ["head"]},
#                 ]
#             },
#             "candidates": [
#                 {"name": "primary", "weight": 90},       
#                 {"name": "qianfan-code", "weight": 10}, 
#             ],
#             "selector": {"type": "hash", "hash_key": "session_id"},
#         },

#         # 规则3: 其他所有情况 → primary
#         {
#             "name": "default",
#             "conditions": [],
#             "candidates": [{"name": "primary", "weight": 1}],
#             "selector": {"type": "weighted_round_robin"},
#         },
#     ]
def _extract_query_from_params(messages: list[Message], variables: dict[str, Any]) -> str:
    """Extract query from messages or variables for trace reporting.
    
    Args:
        messages: List of Message objects from RunParams
        variables: Variables dict from template
        
    Returns:
        Extracted query string, empty string if not found
    """
    # First try to extract from messages - if last message is from user, get its text content
    if messages:
        last_message = messages[-1]
        if last_message.role == "user" and last_message.content:
            # Handle both string content and list content (multimodal)
            if isinstance(last_message.content, str):
                return last_message.content
            elif isinstance(last_message.content, list):
                # Extract text content from list format
                text_parts = []
                for item in last_message.content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                return " ".join(text_parts)
    
    # If no user message found, try to get query from variables
    if variables and "query" in variables:
        query_value = variables["query"]
        if isinstance(query_value, str):
            return query_value
        # Convert non-string values to string
        return str(query_value) if query_value is not None else ""
    
    return ""




class PromptEngine:
    """Resolve templates and generate model responses."""

    def __init__(
        self,
        prompt_loaders: list[TemplateLoader],
        model_loaders: list[ModelConfigLoader] | None = None,
        cache_ttl: int = 60,
        global_model_config: ModelConfig | None = None,
        trace_service: TraceService | None = None,
        tool_trace_service: ToolTraceService | None = None,
        before_run_hooks: list[BeforeRunHook] | None = None,
        after_run_hooks: list[AfterRunHook] | None = None,
        default_hook_configs: dict[str, Any] | None = None,
        router_cache_max_size: int = 100,
        router_cache_ttl: int = 120,
        logger: LoggerProtocol | None = None,
    ) -> None:
        """Initialize the engine with prompt loaders, model loaders and optional global config.

        Args:
            prompt_loaders: List of template loaders
            model_loaders: List of model config loaders
            cache_ttl: Template cache TTL in seconds
            global_model_config: Global model configuration
            trace_service: Trace service for logging
            tool_trace_service: Tool trace service for logging
            before_run_hooks: Hooks to run before model execution
            after_run_hooks: Hooks to run after model execution
            default_hook_configs: Default hook configurations
            router_cache_max_size: Maximum number of routers to cache (LRU policy)
            router_cache_ttl: Router cache TTL in seconds (default: 120s)
            logger: Custom logger instance (optional). If provided, will be used globally.
                   If not provided, uses default loguru logger.
        """
        # Set custom logger if provided
        if logger is not None:
            set_logger(logger)
        self._prompt_loaders = prompt_loaders
        self._model_loaders = model_loaders or []
        self._cache_ttl = cache_ttl
        self._global_cfg = global_model_config
        self._trace_service = trace_service
        self._tool_trace_service = tool_trace_service
        self._before_run_hooks = before_run_hooks or []
        self._after_run_hooks = after_run_hooks or []
        self._default_hook_configs = default_hook_configs or {}
        self._resolve = alru_cache(maxsize=128, ttl=cache_ttl)(self._resolve_impl)
        self._sync_resolve = lru_cache(maxsize=128)(self._sync_resolve_impl)

        # Router cache with LRU policy and TTL (template-level caching)
        # Store tuples of (router, timestamp) for TTL management
        self._router_cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._router_cache_max_size = router_cache_max_size
        self._router_cache_ttl = router_cache_ttl
        self._router_cache_lock = threading.Lock()

        # Global config from PromptStore (Redis etc.)
        self._global_config: dict[str, Any] = {}
        self._global_config_loader = None

        # Redis client cache for sticky routing (to avoid connection leaks)
        self._redis_client = None
        self._redis_client_lock = threading.Lock()

    async def _resolve_impl(self, name: str, version: str | None) -> PromptTemplate:
        for loader in self._prompt_loaders:
            tmpl = await loader.aget_template(name, version)
            if not tmpl:
                continue
            return tmpl
        raise TemplateNotFoundError(name)

    def _sync_resolve_impl(self, name: str, version: str | None) -> PromptTemplate:
        """Synchronous template resolution implementation."""
        for loader in self._prompt_loaders:
            # For sync resolution, we need to handle different loader types
            if hasattr(loader, 'get_template_sync'):
                tmpl = loader.get_template_sync(name, version)
                if tmpl:
                    return tmpl
            # else:
            #     # For loaders that only have async methods, we run them in sync context
            #     import asyncio
            #     try:
            #         loop = asyncio.get_event_loop()
            #         if loop.is_running():
            #             # If we're in an async context, we can't use run_until_complete
            #             # In this case, we should use a different approach
            #             raise RuntimeError("Cannot resolve template synchronously from async context")
            #         else:
            #             tmpl = loop.run_until_complete(loader.aget_template(name, version))
            #     except RuntimeError:
            #         # No event loop, create one
            #         tmpl = asyncio.run(loader.aget_template(name, version))

        raise TemplateNotFoundError(name)

    def _get_or_build_router(
        self,
        template_name: str,
        version: str | None,
        route_rules: list[dict],
        sticky_config: dict[str, Any] | None = None
    ) -> Any:
        """Get or build router with template-level LRU caching and TTL.

        This method implements a template-level router cache with:
        - LRU eviction policy when cache is full
        - TTL-based expiration (default: 120 seconds)

        Cache key format: "{template_name}:{version}:{rules_hash}"

        Args:
            template_name: Template name for cache key
            version: Template version for cache key
            route_rules: Routing rules to build the router
            sticky_config: Sticky routing configuration (optional)

        Returns:
            Router instance (cached or newly built)

        Note:
            Sticky routing is handled at route-level inside RoutePipeline.
            Router cache does NOT include sticky_config in cache key because:
            1. Sticky config rarely changes (global Redis config)
            2. Sticky logic is stateless - same router can be used with different configs
            3. Including sticky_config would fragment the cache unnecessarily
        """
        # Generate cache key: template_name + version + rules_hash
        rules_json = json.dumps(route_rules, sort_keys=True)
        rules_hash = hashlib.md5(rules_json.encode()).hexdigest()[:8]
        cache_key = f"{template_name}:{version or 'default'}:{rules_hash}"

        current_time = time.time()

        with self._router_cache_lock:
            # Check if key exists in cache
            if cache_key in self._router_cache:
                router, timestamp = self._router_cache[cache_key]

                # Check if cache entry is expired
                if current_time - timestamp > self._router_cache_ttl:
                    logger.debug(f"Router cache expired for template '{template_name}' version '{version}'"
                                 f"(age: {current_time - timestamp:.1f}s)")
                    del self._router_cache[cache_key]
                else:
                    # Cache hit - move to end (mark as recently used)
                    self._router_cache.move_to_end(cache_key)
                    logger.debug(f"Router cache hit for template '{template_name}' version '{version}' (age:"
                                 f" {current_time - timestamp:.1f}s)")
                    return router

            # Build new router with sticky routing support
            from .router.conditional import build_pipeline
            router = build_pipeline(
                route_definitions=route_rules,
                sticky_config=sticky_config,
                template_name=template_name,
                global_config_getter=self.get_global_config,
                redis_client=self.get_redis_client()
            )

            # Add to cache with current timestamp
            self._router_cache[cache_key] = (router, current_time)
            self._router_cache.move_to_end(cache_key)

            # Clean up expired entries
            expired_keys = []
            for key, (_, timestamp) in self._router_cache.items():
                if current_time - timestamp > self._router_cache_ttl:
                    expired_keys.append(key)

            for key in expired_keys:
                del self._router_cache[key]
                logger.debug(f"Router cache expired during cleanup: {key}")

            # Evict oldest entry if cache is still full after cleanup
            if len(self._router_cache) > self._router_cache_max_size:
                evicted_key = next(iter(self._router_cache))
                del self._router_cache[evicted_key]
                logger.debug(f"Router cache full, evicted oldest: {evicted_key}")

            logger.info(
                f"Built and cached router for template '{template_name}' version '{version}', "
                f"cache_key={cache_key}, cache_size={len(self._router_cache)}, ttl={self._router_cache_ttl}s"
            )
            return router

    def _select_variant_by_router(
        self,
        variant_router: dict[str, Any] | None,
        tmpl: PromptTemplate,
        ctx: dict[str, Any] | None,
        template_name: str = "",
        version: str | None = None
    ) -> str | None:
        """Select variant using variant_router with fallback to template auto-selection.

        Template stores only routing rules, context is provided at runtime.
        Supports two scenarios:
        1. Runtime provides complete variant_router (rules + context)
        2. Runtime provides only context, uses template's rules
        3. Supports sticky routing via Redis (configured in template or runtime)

        Args:
            variant_router: Runtime variant router configuration (may contain rules and/or context)
            tmpl: Template with routing rules (tmpl.variant_router contains only rules)
            ctx: Context variables for template auto-selection fallback
            template_name: Template name for router caching
            version: Template version for router caching

        Returns:
            Selected variant name, or None if routing fails
        """
        # Record start time for metrics
        start_time = time.time()

        try:
            # Determine route rules and context
            route_rules = None
            route_context = None

            if variant_router:
                # Runtime may provide rules and/or context
                route_rules = variant_router.get("rules")
                route_context = variant_router.get("context")

                # If runtime only provides context, use template's rules
                if route_rules is None and tmpl.variant_router:
                    route_rules = tmpl.variant_router.get("rules")
            elif tmpl.variant_router:
                # Template only has rules, no context
                # This case won't select a variant via routing since context is missing
                route_rules = tmpl.variant_router.get("rules")

            # Execute routing only if we have both rules and context
            if route_rules and route_context:
                try:
                    # Add template_name to route_context for metrics tracking
                    if 'template_name' not in route_context:
                        route_context = dict(route_context)
                        route_context['template_name'] = tmpl.name

                    # Get sticky routing configuration (returns None if disabled)
                    sticky_config = self._get_sticky_config(variant_router, tmpl)

                    # Build router and select variant
                    # Note: Sticky routing is now handled inside RoutePipeline at route-level
                    router = self._get_or_build_router(
                        template_name=template_name,
                        version=version,
                        route_rules=route_rules,
                        sticky_config=sticky_config
                    )
                    selected_candidate = router.select(route_context)
                    variant_name = selected_candidate.name

                    return variant_name
                except Exception as e:
                    logger.warning(f"Failed to select variant via routing: {e}, falling back to auto-selection")

            # Fallback to template auto-selection
            return tmpl.choose_variant(ctx) or next(iter(tmpl.variants))
        finally:
            # Record variant selection duration metrics
            duration_seconds = time.time() - start_time
            duration_milliseconds = duration_seconds * 1000

            # Log variant selection duration
            logger.debug(
                f"Variant selection duration: {duration_seconds:.4f}s ({duration_milliseconds:.2f}ms) "
                f"for template '{template_name or tmpl.name}'"
            )

            llm_metrics = get_llm_metrics()
            if llm_metrics and llm_metrics.enabled:
                metric_attrs = {
                    "template_name": template_name or tmpl.name,
                }
                llm_metrics.variant_selection_duration.record(duration_milliseconds, metric_attrs)

    def _get_sticky_config(self, variant_router: dict[str, Any] | None, tmpl: PromptTemplate) -> dict[str, Any] | None:
        """Get sticky routing configuration from runtime or template.


        Sticky routing is enabled by default if global Redis config is available.
        Default TTL is 30 days.

        Args:
            variant_router: Runtime variant router configuration
            tmpl: Template instance

        Returns:
            Sticky routing configuration dict with defaults, or None if disabled
        """
        sticky_config = None

        global_redis_config = self.get_global_config("redis")
        if global_redis_config and global_redis_config.get("enabled"):
            # 自动启用 sticky routing,使用默认配置
            sticky_config = {"enabled": True}

        # 如果没有任何配置,返回 None
        if not sticky_config:
            return None

        # 检查是否显式禁用
        if sticky_config.get("enabled") is False:
            return None

        # 应用默认值
        sticky_config.setdefault("enabled", True)
        sticky_config.setdefault("ttl_days", 30)

        return sticky_config

    async def aload(self, template_name: str, version: str = None) -> PromptTemplate:
        """Public entry: resolve & cache a template by name."""
        return await self._resolve(template_name, version)

    # Backward compatibility alias
    async def load(self, template_name: str, version: str = None) -> PromptTemplate:
        """Deprecated: use aload() instead."""
        import warnings
        warnings.warn("load() is deprecated, use aload() instead", DeprecationWarning, stacklevel=2)
        return await self.aload(template_name, version)

    def _mask_sensitive_fields(self, cfg_list: list[ModelConfig]) -> list[dict]:
        """Mask sensitive fields in model config list for logging.
        
        Args:
            cfg_list: List of ModelConfig objects
            
        Returns:
            List of dicts with sensitive fields masked
        """
        masked_list = []
        for cfg in cfg_list:
            cfg_dict = cfg.model_dump()
            if cfg_dict.get('api_key'):
                cfg_dict['api_key'] = '***MASKED***'
            if cfg_dict.get('token_config'):
                cfg_dict['token_config'] = '***MASKED***'
            masked_list.append(cfg_dict)
        return masked_list

    def get_model_config(self, model_name: str, provider: str = None, model_value: str = None,
                         llm_token: str = None, model_control=None) -> list[ModelConfig] | None:
        """Get ordered list of model configurations by name from loaded model loaders.
        
        Args:
            model_name: Required. The aggregated model name to look up.
            provider: Optional. Filter by provider name. If provided, configs with this provider will be prioritized.
            model_value: Optional. Filter by actual model name (real API model name).
            llm_token: Optional. Filter by associated LLM token name.
            model_control: Optional. ModelControlParams for dynamic weight/enable/disable control.
            
        Returns:
            list[ModelConfig]: Ordered list of model configurations sorted by priority, or None if not found.
                              If provider is specified, matching providers come first.
                              Within each group, configs are sorted by weight (highest first).
        """
        for loader in self._model_loaders:
            try:
                return loader.get_model_config(model_name, provider, model_value, llm_token, model_control)
            except ModelConfigNotFoundError:
                continue
        return None

    def _merge_model_configs(self, input_cfg: ModelConfig | None, template_cfg: ModelConfig | None,
                             variant_strategy: dict[str, Any] | None = None,
                             input_strategy: dict[str, Any] | None = None) -> list[ModelConfig]:
        """Merge model configurations and return ordered list for sequential attempt.
        
        Args:
            input_cfg: Configuration passed to run method
            template_cfg: Configuration from template variant  
            variant_strategy: Model strategy from template variant
            input_strategy: Model strategy passed to run method (highest priority)
            
        Returns:
            List of ModelConfig objects ordered by priority for sequential attempt
            
        Raises:
            ValueError: If no valid configuration could be determined
        """
        # Priority order: input_strategy > input_cfg > variant_strategy > template_cfg > global_cfg
        
        # 1. If input has model_strategy, use it directly (highest priority)
        if input_strategy is not None:
            return self._parse_model_strategy(input_strategy)
        
        # 2. If input has model_cfg, use it (second highest priority)
        if input_cfg is not None:
            # Get model name from input config for registry lookup
            if not input_cfg.model:
                raise ValueError("Model name is required in model configuration")
            
            lookup_model_name = input_cfg.get_aggregated_model_name()
            
            # Get ordered list of model configs from registry
            registry_cfgs = self.get_model_config(
                model_name=lookup_model_name, 
                provider=input_cfg.provider, 
                model_value=input_cfg.model_value
            )
            logger.debug(f"registry_cfgs: {registry_cfgs}")
            
            if registry_cfgs is None or len(registry_cfgs) == 0:
                # No registry configs found, create single config from input_cfg
                merged_cfg = input_cfg.model_copy()
                
                # Apply fallback logic: fill missing fields from template_cfg or variant_strategy
                fallback_cfgs = []
                if variant_strategy is not None:
                    fallback_cfgs.extend(self._parse_model_strategy(variant_strategy))
                if template_cfg is not None:
                    fallback_cfgs.append(template_cfg)
                if self._global_cfg is not None:
                    fallback_cfgs.append(self._global_cfg)
                
                for fallback_cfg in fallback_cfgs:
                    for field_name, field_info in ModelConfig.model_fields.items():
                        src_value = getattr(merged_cfg, field_name)
                        if src_value is None or (isinstance(src_value, str) and src_value == ""):
                            fallback_value = getattr(fallback_cfg, field_name)
                            if fallback_value is not None:
                                setattr(merged_cfg, field_name, fallback_value)
                                break
                
                # Ensure configuration completeness
                if not merged_cfg.provider:
                    raise ValueError("Provider is required in model configuration")
                
                return [merged_cfg]
            
            # Registry configs found, merge with input config
            result_configs = []
            for registry_cfg in registry_cfgs:
                merged_cfg = registry_cfg.model_copy()
                # Override with input config fields if provided
                for field_name, field_info in ModelConfig.model_fields.items():
                    input_value = getattr(input_cfg, field_name)
                    if input_value:
                        setattr(merged_cfg, field_name, input_value)
                
                result_configs.append(merged_cfg)
            
            if len(result_configs) == 1:
                result_configs.append(result_configs[0])
            logger.debug(f"result_configs: {result_configs}")
            return result_configs
        
        # 3. If variant has model_strategy, use it (third highest priority)
        if variant_strategy is not None:
            return self._parse_model_strategy(variant_strategy)
        
        # 4. Otherwise use template_cfg or global_cfg
        # Determine the base configuration to use
        base_cfg = template_cfg or self._global_cfg
        if base_cfg is None:
            raise ValueError("ModelConfig required but not provided in template or globally")

        # Get the model name for registry lookup
        if not base_cfg.model:
            raise ValueError("Model name is required in model configuration")

        lookup_model_name = base_cfg.get_aggregated_model_name()
        
        # Get ordered list of model configs from registry
        registry_cfgs = self.get_model_config(
            model_name=lookup_model_name, 
            provider=base_cfg.provider, 
            model_value=base_cfg.model_value
        )
        logger.debug(f"registry_cfgs: {registry_cfgs}")
        if registry_cfgs is None or len(registry_cfgs) == 0:
            # No registry configs found, create a single config from base
            merged_cfg = base_cfg.model_copy()
            
            # Apply fallback logic for missing fields
            if input_cfg is not None:
                # If input_cfg provided, supplement from template_cfg or global_cfg
                fallback_cfg = template_cfg or self._global_cfg
                if fallback_cfg is not None:
                    for field_name, field_info in ModelConfig.model_fields.items():
                        src_value = getattr(merged_cfg, field_name)
                        if src_value is None or (isinstance(src_value, str) and src_value == ""):
                            fallback_value = getattr(fallback_cfg, field_name)
                            if fallback_value is not None:
                                setattr(merged_cfg, field_name, fallback_value)
            
            # Ensure configuration completeness
            if not merged_cfg.provider:
                raise ValueError("Provider is required in model configuration")
            
            return [merged_cfg]
        
        # Registry configs found, create merged configs for each
        result_configs = []
        for registry_cfg in registry_cfgs:
            # Start with registry config as base
            merged_cfg = registry_cfg.model_copy()
            # Override with input config fields if provided
            if input_cfg is not None:
                for field_name, field_info in ModelConfig.model_fields.items():
                    input_value = getattr(input_cfg, field_name)
                    if input_value:
                        setattr(merged_cfg, field_name, input_value)
            
            # Fill missing fields from template config if needed
            if template_cfg is not None:
                for field_name, field_info in ModelConfig.model_fields.items():
                    current_value = getattr(merged_cfg, field_name)
                    if not current_value and (template_value := getattr(template_cfg, field_name)):
                        setattr(merged_cfg, field_name, template_value)
            
            # Fill missing fields from global config if needed
            if self._global_cfg is not None:
                for field_name, field_info in ModelConfig.model_fields.items():
                    current_value = getattr(merged_cfg, field_name)
                    if not current_value and (global_value := getattr(self._global_cfg, field_name)):
                        setattr(merged_cfg, field_name, global_value)
            
            # Ensure configuration completeness
            if not merged_cfg.provider:
                raise ValueError("Provider is required in model configuration")
            if not merged_cfg.model:
                raise ValueError("Model name is required in model configuration")
            
            result_configs.append(merged_cfg)
        if len(result_configs) == 1:
            result_configs.append(result_configs[0])
        logger.debug(f"result_configs: {result_configs}")
        return result_configs

    def _apply_private_llm_provider(self, cfg_list: list[ModelConfig]) -> None:
        """如果设置了 CHOOSE_PRIVATE_LLM 环境变量，对于 claude 模型固定使用 wanshi-openai provider.

        Args:
            cfg_list: 模型配置列表，将被就地修改
        """
        env_value = os.getenv("CHOOSE_PRIVATE_LLM", "").lower()
        if env_value not in ("true", "1", "yes"):
            return

        for cfg in cfg_list:
            if cfg.model and "claude" in cfg.model.lower():
                logger.debug(f"CHOOSE_PRIVATE_LLM enabled, switching claude model {cfg.model}"
                             f" provider from {cfg.provider} to wanshi-openai")
                # 从配置中获取 wanshi-openai 的完整配置
                wanshi_configs = self.get_model_config(model_name=cfg.get_aggregated_model_name(),
                                                       provider="wanshi-openai")
                if wanshi_configs and len(wanshi_configs) > 0:
                    # 使用找到的 wanshi-openai 配置
                    wanshi_cfg = wanshi_configs[0]
                    cfg.provider = wanshi_cfg.provider
                    if wanshi_cfg.api_url:
                        cfg.api_url = wanshi_cfg.api_url
                    if wanshi_cfg.api_key:
                        cfg.api_key = wanshi_cfg.api_key
                    logger.debug(f"Applied wanshi-openai config: api_url={cfg.api_url}")
                else:
                    # 如果没有找到配置，只修改 provider
                    cfg.provider = "wanshi-openai"
                    logger.warning(f"No wanshi-openai config found for model {cfg.model}, only changed provider")

    def _parse_model_strategy(self, strategy: dict[str, Any]) -> list[ModelConfig]:
        """Parse model strategy configuration and return ordered ModelConfig list.

        Args:
            strategy: Model strategy configuration dict

        Returns:
            List of ModelConfig objects ordered by priority for fallback

        Raises:
            ValueError: If strategy configuration is invalid
        """
        strategy_type = strategy.get("type")

        if strategy_type == "fallback_chain":
            return self._parse_fallback_chain_strategy(strategy)
        else:
            # For future strategy types, we can add them here
            raise ValueError(f"Unsupported model strategy type: {strategy_type}")
    
    def _parse_fallback_chain_strategy(self, strategy: dict[str, Any]) -> list[ModelConfig]:
        """Parse fallback_chain strategy and return ordered ModelConfig list.

        Args:
            strategy: Strategy config with type="fallback_chain"
                     Expected format:
                     {
                         "type": "fallback_chain",
                         "models": [
                             {
                                 "model": "gpt-4",                    # required: aggregated model name
                                 "provider": "openai",               # optional: filter by provider
                                 "priority": 1,                      # required: fallback priority
                                 "weight": 100,                      # optional: override weight
                                 "api_url": "https://...",           # optional: override api_url
                                 "api_key": "sk-xxx",                # optional: override api_key
                                 "model_value": "gpt-4-turbo",       # optional: override model_value
                                 "ext": {                            # optional: extension fields
                                     "bedrock_model_tag": "..."      # e.g., for AWS Bedrock model_id
                                 }
                             },
                             ...
                         ]
                     }

        Returns:
            List of ModelConfig objects sorted by priority (lowest priority number first)
        """
        models = strategy.get("models", [])
        if not models:
            raise ValueError("fallback_chain strategy requires 'models' list")
        
        model_configs = []
        for model_data in models:
            if not isinstance(model_data, dict):
                raise ValueError("Each model in fallback_chain must be a dict")
            
            # Required fields
            model = model_data.get("model")
            if not model:
                raise ValueError("Each model must specify 'model' field")
            
            priority = model_data.get("priority")
            if priority is None:
                raise ValueError("Each model must specify 'priority' field")
            
            # Optional fields for filtering and overrides
            provider_filter = model_data.get("provider")
            weight_override = model_data.get("weight")
            api_url_override = model_data.get("api_url")
            api_key_override = model_data.get("api_key")
            model_value_override = model_data.get("model_value")
            temperature_override = model_data.get("temperature")
            top_p_override = model_data.get("top_p")
            max_tokens_override = model_data.get("max_tokens")
            ext_override = model_data.get("ext")

            # 如果设置了 CHOOSE_PRIVATE_LLM 环境变量，对于 claude 模型使用 wanshi-openai provider
            env_value = os.getenv("CHOOSE_PRIVATE_LLM", "").lower()
            if env_value in ("true", "1", "yes") and "claude" in model.lower():
                logger.debug(f"CHOOSE_PRIVATE_LLM enabled in strategy, using wanshi-openai for claude model {model}")
                provider_filter = "wanshi-openai"

            # Try to get model configs from registry first
            registry_configs = self.get_model_config(
                model_name=model,
                provider=provider_filter,
                model_value=model_value_override
            )
            
            if registry_configs and len(registry_configs) > 0:
                # Use the first (highest priority) config from registry as base
                base_config = registry_configs[0]

                # Merge ext fields: start with base_config.ext, then override with ext_override
                merged_ext = base_config.ext.copy() if base_config.ext else {}
                if ext_override:
                    merged_ext.update(ext_override)

                # Create new config with overrides
                config = ModelConfig(
                    provider=base_config.provider,
                    model=base_config.model,
                    model_value=model_value_override or base_config.model_value,
                    api_url=api_url_override or base_config.api_url,
                    api_key=api_key_override or base_config.api_key,
                    weight=weight_override if weight_override is not None else base_config.weight,
                    temperature=temperature_override if temperature_override is not None else base_config.temperature,
                    top_p=top_p_override if top_p_override is not None else base_config.top_p,
                    max_tokens=max_tokens_override if max_tokens_override is not None else base_config.max_tokens,
                    ext=merged_ext,
                )
            else:
                # Registry config not found, create from provided data
                if not provider_filter:
                    raise ValueError(f"Model '{model}' not found in registry and no 'provider' specified")

                config = ModelConfig(
                    provider=provider_filter,
                    model=model,
                    model_value=model_value_override or model,
                    api_url=api_url_override or "",
                    api_key=api_key_override or "",
                    weight=weight_override if weight_override is not None else 50,
                    temperature=temperature_override if temperature_override is not None else None,
                    top_p=top_p_override if top_p_override is not None else None,
                    max_tokens=max_tokens_override if max_tokens_override is not None else None,
                    ext=ext_override or {},
                )
            
            model_configs.append((priority, config))
        
        # Sort by priority (lowest number = highest priority)
        model_configs.sort(key=lambda x: x[0])
        
        # Return only the ModelConfig objects
        return [config for _, config in model_configs]

    def list_available_models(self) -> list[str]:
        """List all available models from all model loaders."""
        models = []
        for loader in self._model_loaders:
            models.extend(loader.list_models())
        return list(set(models))  # Remove duplicates

    def load_model_configs(self):
        """Load all model configurations from model loaders."""
        for loader in self._model_loaders:
            loader.load()

    def _load_global_config(self, base_url: str, auth_token: str):
        """
        Load global configuration from PromptStore

        Args:
            base_url: PromptStore API base URL
            auth_token: Authentication token
        """
        try:
            from .loader.global_config_loader import GlobalConfigLoader

            logger.debug("Loading global configuration from PromptStore...")

            # 创建配置加载器
            self._global_config_loader = GlobalConfigLoader(
                base_url=base_url,
                auth_token=auth_token,
                max_retries=3,
                timeout=10,
                enable_cache=True,
                cache_ttl=300,  # 5分钟缓存
            )

            # 获取配置 (自动解密)
            config = self._global_config_loader.get_config(decrypt=True)

            if config:
                self._global_config = config
                logger.debug(f"Successfully loaded global config: {list(config.keys())}")

                # 日志输出配置信息 (隐藏敏感信息)
                if "redis" in config:
                    redis_config = config["redis"]
                    logger.debug(f"Redis config loaded: enabled={redis_config.get('enabled')}, "
                               f"host={redis_config.get('host')}, port={redis_config.get('port')},"
                                f" db={redis_config.get('db')}")

                if "sticky_routing" in config:
                    sticky_config = config["sticky_routing"]
                    logger.info(f"Sticky routing config loaded: enabled={sticky_config.get('enabled')}, "
                               f"ttl_days={sticky_config.get('ttl_days')}")

            else:
                logger.warning("No global config available, sticky routing will be disabled")

        except ImportError as e:
            logger.warning(f"Cannot load global config loader: {e}")
        except Exception as e:
            logger.error(f"Failed to load global config: {e}")
            # 失败时不抛出异常,允许 Engine 继续运行

    def get_global_config(self, key: str = None) -> dict[str, Any]:
        """
        Get global configuration

        Args:
            key: Optional key to get specific config section (e.g., 'redis', 'sticky_routing')
                 If None, returns entire config

        Returns:
            Global configuration dict or specific section
        """
        if key:
            return self._global_config.get(key, {})
        return self._global_config.copy()

    def get_redis_client(self):
        """Get or create a cached Redis client for sticky routing.

        Returns:
            Redis client instance, or None if Redis is not available

        Note:
            This method reuses the same Redis connection to avoid connection leaks.
            The connection is created lazily on first use.
        """
        # Fast path: return cached client if available
        if self._redis_client is not None:
            return self._redis_client

        # Slow path: create new client (thread-safe)
        with self._redis_client_lock:
            # Double-check after acquiring lock
            if self._redis_client is not None:
                return self._redis_client

            # Get Redis config from global config
            redis_config = self.get_global_config("redis")
            if not redis_config or not redis_config.get("enabled"):
                logger.debug("Redis not enabled in global config")
                return None

            try:
                import redis

                # Create connection pool for Redis
                pool = redis.ConnectionPool(
                    host=redis_config.get("host", "localhost"),
                    port=redis_config.get("port", 6379),
                    db=redis_config.get("db", 0),
                    password=redis_config.get("password"),
                    decode_responses=True,
                    socket_connect_timeout=1,
                    socket_timeout=1,
                    max_connections=redis_config.get("max_connections", 10),
                    # Connection pool settings
                    socket_keepalive=True,
                    socket_keepalive_options={},
                    health_check_interval=30,  # Health check every 30s
                )

                # Create Redis client from pool
                self._redis_client = redis.Redis(connection_pool=pool)

                # Test connection
                self._redis_client.ping()
                logger.info(
                    f"Redis connection pool initialized: "
                    f"{redis_config.get('host')}:{redis_config.get('port')}, "
                    f"db={redis_config.get('db', 0)}, "
                    f"max_connections={redis_config.get('max_connections', 10)}"
                )

                return self._redis_client

            except ImportError:
                logger.warning("redis-py not installed, sticky routing disabled")
                return None
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}")
                return None

    def close_redis_client(self):
        """Close the cached Redis client connection.

        Should be called when the Engine is destroyed or no longer needed.
        """
        if self._redis_client is not None:
            with self._redis_client_lock:
                if self._redis_client is not None:
                    try:
                        self._redis_client.close()
                        logger.debug("Redis client closed")
                    except Exception as e:
                        logger.warning(f"Error closing Redis client: {e}")
                    finally:
                        self._redis_client = None

    async def aformat(
        self,
        template_name: str,
        variables: dict[str, Any],
        *,
        variant: str | None = None,
        version: str | None = None,
        selector: dict[str, Any] | None = None
    ) -> list[Message] | list[dict]:
        """Return formatted messages for ``template_name`` in ``format``."""
        tmpl = await self._resolve(template_name, version)
        msgs, _ = tmpl.format(
            variables,
            variant=variant,
            selector=selector
        )
        return msgs

    async def acompletion(
        self,
        template_name: str,
        variables: dict[str, Any],
        model_cfg: ModelConfig | dict[str, Any] | None = None,
        *,
        model_strategy: dict[str, Any] | None = None,
        variant_router: dict[str, Any] | None = None,
        version: str | None = None,
        variant: str | None = None,
        ctx: dict[str, Any] | None = None,
        tool_params: ToolParams | list[ToolSpec] | list[dict] | dict[str, Any] | None = None,
        messages: list[Message] | list[dict] | None = None,
        template: PromptTemplate | None = None,
        hook_configs: dict[str, Any] | None = None,
        timeout: float | None = None,
        **run_params: Any,
    ) -> AsyncGenerator[Union[ModelResponse, StreamingModelResponse], None]:
        """Stream messages produced by running the template.

        Args:
            template_name: Name of the template to run
            variables: Variables to use for template rendering
            model_cfg: Optional ModelConfig to use or override template's config.
                      Can be a ModelConfig object or a dict with config parameters.
                      If None, will use the template's model configuration.
                      If provided, will merge with template's config (with this taking precedence).
            model_strategy: Optional model strategy configuration for multi-model fallback.
                           Takes highest priority over variant and input model configs.
                           Format: {"type": "fallback_chain", "models": [...]}
            variant_router: Optional variant router configuration for variant selection.
                           Dict with two fields:
                           - "rules": List of routing rules that define conditions and candidate variants
                           - "context": Dict containing fields used by routing conditions
                           Format: {"rules": [...], "context": {...}}
            variant: Optional variant name to use (highest priority, overrides routing)
            ctx: Optional context for variant selection (defaults to variables)
            tool_params: Optional tool parameters for model calls.
                        Can be a ToolParams object, list of ToolSpec objects,
                        list of dicts (converted to ToolSpec), or dict with ToolParams fields.
            messages: Optional direct messages to use instead of template.
                     Can be a list of Message objects or list of dicts (converted to Message).
                     If provided, template_name and variables will be ignored.
            hook_configs: Optional hook configurations dict to override default hooks.
                         Format: {"desensitization": {"type": "wordlist", "wordlist": {...}}}
            timeout: Optional timeout in seconds for the model API call
            **run_params: Additional parameters passed to model run
                         (e.g., stream, temperature, max_tokens, user_id,
                         request_id, conversation_id, session_id, etc.)

        Returns:
            AsyncGenerator yielding ModelResponse or StreamingResponse objects
        """
        # Convert basic types to objects
        converted_model_cfg = self._convert_model_cfg(model_cfg) if model_cfg is not None else None
        converted_tool_params = self._convert_tool_params(tool_params) if tool_params is not None else None
        if messages is not None:
            filtered_messages = self._filter_empty_assistant_messages(messages)
            converted_messages = self._convert_messages(filtered_messages)
        else:
            converted_messages = None

        # 解析模板
        tmpl_name = template_name
        tmpl = await self._resolve(template_name, version) if template is None else template

        # 使用模板解析获取 variant 配置
        ctx = ctx or variables

        # Variant selection priority:
        # 1. Explicit variant parameter (highest priority)
        # 2. variant_router routing (runtime or template-level)
        # 3. Template auto-selection (lowest priority)
        if variant is None:
            variant = self._select_variant_by_router(
                variant_router,
                tmpl,
                ctx,
                template_name=tmpl_name,
                version=version
            )

        logger.debug(f"use variant: {variant}")

        # 如果直接提供了messages，使用提供的messages；否则使用模板生成
        if converted_messages is not None:
            # 直接使用提供的messages，但仍需要获取 variant 配置
            _, var = tmpl.format(
                variables,
                variant=variant,
                selector=ctx,
            )
            params = RunParams(messages=converted_messages, tool_params=converted_tool_params, timeout=timeout,
                                **run_params)
        else:
            # 使用模板生成 messages
            messages, var = tmpl.format(
                variables,
                variant=variant,
                selector=ctx,
            )
            # Filter out empty assistant messages from template-generated messages
            filtered_template_messages = self._filter_empty_assistant_messages(messages)
            params = RunParams(messages=cast(list[Message], filtered_template_messages),
                               tool_params=converted_tool_params, timeout=timeout, **run_params)

        # 设置跟踪属性
        span_attrs = {
            "template.name": tmpl_name,
        }

        if var is not None:
            # 只有使用模板时才有这些属性
            span_attrs["template.version"] = getattr(var, "version", None) or ""
            span_attrs["variant"] = variant or ""

        with _tracer.start_as_current_span(
            "prompt.run",
            attributes=span_attrs,
        ):
            # 合并配置：传入的model_strategy > 传入的model_cfg > 模板的var.model_strategy > 模板的var.model_cfg > 全局配置
            template_cfg = var.model_cfg if var is not None else None
            variant_strategy = var.model_strategy if var is not None else None

            cfg_list = self._merge_model_configs(input_cfg=converted_model_cfg, template_cfg=template_cfg,
                                                 variant_strategy=variant_strategy, input_strategy=model_strategy)
            fallback_chain_enabled = (
                (isinstance(model_strategy, dict) and model_strategy.get("type") == "fallback_chain")
                or (isinstance(variant_strategy, dict) and variant_strategy.get("type") == "fallback_chain")
            )

            # 应用私有 LLM provider 配置（如果需要）
            self._apply_private_llm_provider(cfg_list)

            logger.info(f"get model_cfg list: {self._mask_sensitive_fields(cfg_list)}")

            # 构建 completion 请求参数并添加到 params 中
            completion_request_params = {
                "template_name": template_name,
                "variables": variables,
                "model_cfg": converted_model_cfg.model_dump() if converted_model_cfg else None,
                "model_strategy": model_strategy,
                "variant_router": variant_router,
                "version": version,
                "variant": variant,
                "ctx": ctx,
                "tool_params": converted_tool_params.model_dump() if converted_tool_params else None,
                "messages": [msg.model_dump() for msg in converted_messages] if converted_messages else None,
                "hook_configs": hook_configs,
                "timeout": timeout,
                "run_params": run_params,
            }

            # Try each model config in order until one succeeds
            for i, cfg in enumerate(cfg_list):
                is_last_model = (i == len(cfg_list) - 1)
                try:
                    async for response in self._try_single_model_async(
                        cfg, params, tmpl_name, tmpl, variant, var, variables, hook_configs,
                            completion_request_params, is_last_model, fallback_chain_enabled
                    ):
                        yield response
                    # If we get here, the model succeeded, so break the loop
                    return
                except Exception as e:
                    logger.warning(f"Model {cfg.get_aggregated_model_name()} (provider: {cfg.provider}) failed: {e}")
                    # If this is the last model, return error response instead of raising exception
                    if is_last_model:
                        logger.error(f"All {len(cfg_list)} models failed, returning error response")
                        # Create error response
                        error_response = ModelResponse(
                            id=str(uuid.uuid4()),
                            model=cfg.get_aggregated_model_name(),
                            created=int(time.time()),
                            error={
                                "message": f"All models failed. Last error: {str(e)}",
                                "type": "fallback_failed",
                                "code": "all_models_failed"
                            }
                        )
                        yield error_response
                        return
                    # Otherwise, continue to next model
                    continue

    async def _try_single_model_async(self, cfg: ModelConfig, params: RunParams, tmpl_name: str,
                                      tmpl: PromptTemplate, variant: str, var, variables: dict,
                                      hook_configs: dict, completion_request_params: dict,
                                      is_last_model: bool = False, fallback_chain_enabled: bool = False):
        """Try to execute with a single model configuration."""
        # 创建model client
        model_client = create_client(cfg)

        # 记录开始时间用于计算请求持续时间
        start_time = time.time()
        responses = []
        # Extract query for trace reporting
        query = _extract_query_from_params(params.messages, variables)
        
        event = TraceEvent(
            template_name=tmpl_name if tmpl else "",
            template_version=tmpl.version if tmpl else "",
            template_id=tmpl.id if tmpl else "",
            variant=variant,
            model=cfg.get_aggregated_model_name(),  # Use aggregated model name for tracking
            messages_template=var.messages if var else "",
            timestamp=start_time,
            variables=variables,
            query=query,
            user_id=params.user_id,
            request_id=params.request_id,
            app_id=params.app_id,
            conversation_id=params.conversation_id or params.session_id,
            span_id=params.span_id,
            parent_span_id=params.parent_span_id,
            source=params.source,
            ext={"completion_params": completion_request_params},
        )
        
        # Initialize OTEL metrics tracking
        llm_metrics = get_llm_metrics()
        otel_context = None
        first_token_time = None  # Track first token/chunk time
        if llm_metrics:
            otel_context = llm_metrics.record_request_start(
                template_name=tmpl_name,
                model=cfg.get_aggregated_model_name(),
                provider=cfg.provider,
                variant=variant or "",
                streaming=str(params.stream).lower() if hasattr(params, 'stream') else "false"
            )
        try:
            # 保存原始参数用于trace上报
            original_params = params
            original_responses = []

            # 处理hook_configs并创建对应的hooks
            effective_before_hooks, effective_after_hooks = self._create_hooks_from_configs(hook_configs, cfg)

            # 执行before run hooks
            processed_params = params
            hook_metadata = {}
            for hook in effective_before_hooks:
                hook_result = await hook.aprocess(processed_params)
                processed_params = hook_result.data
                hook_metadata.update(hook_result.metadata)

            try:
                hook_debug_content = {}
                hook_response_data = {}  # 新增：存储每个hook的完整响应数据
                for hook in effective_after_hooks:
                    # 将类名转换为下划线格式
                    hook_name = self._class_name_to_snake_case(hook.__class__.__name__)
                    hook_debug_content[hook_name] = ""
                    hook_response_data[hook_name] = []  # 存储该hook的所有响应数据
                # 根据参数判断是否为流式响应
                is_streaming = processed_params.stream
                
                # 生成唯一的会话ID用于流式处理
                session_id = f"{params.request_id or uuid.uuid4().hex}_{start_time}"
                
                # 如果是流式响应，初始化hook会话
                if is_streaming:
                    for hook in effective_after_hooks:
                        await hook.astart_streaming_session(session_id, hook_metadata)
                
                # 正常处理模型响应
                async for response in model_client.arun(processed_params):
                    # 保存原始响应（未进行after hook处理）
                    original_responses.append(response.model_dump(exclude_none=True))
                    # 如果响应包含错误，检查是否应该触发provider fallback
                    if hasattr(response, 'error') and response.error:
                        error_msg = response.error.get('message', '') if isinstance(response.error, dict)\
                            else str(response.error)
                        # 创建一个异常对象来检查是否应该重试/fallback
                        error_exception = Exception(error_msg)
                        should_fallback = fallback_chain_enabled or should_retry_error(error_exception)
                        if should_fallback:
                            # 如果是最后一个模型且需要fallback，返回错误响应而不是抛出异常
                            if is_last_model:
                                logger.warning(f"Last model failed with fallback error,"
                                               f" returning error response: {error_msg}")
                                # 将当前response标记为最终错误响应并返回
                                responses.append(response.model_dump(exclude_none=True))
                                # 错误响应：在yield前上报
                                asyncio.create_task(self._async_report_error_trace(
                                    event, Exception(error_msg), params, cfg, original_params,
                                    effective_before_hooks, effective_after_hooks
                                ))
                                # Record OTEL metrics error - 上报错误指标
                                if llm_metrics and otel_context:
                                    error_type = response.error.get('type', 'unknown_error')\
                                        if isinstance(response.error, dict) else 'error'
                                    asyncio.create_task(
                                        self._async_report_otel_error(llm_metrics, otel_context, error_type))
                                logger.info(f"Async yielding final error response: {response}")
                                yield response
                                return
                            else:
                                # 对于可以fallback的错误（包括限流），抛出异常以触发下一个provider
                                logger.warning(f"Error detected, triggering provider fallback: {error_msg}")
                                raise Exception(f"Provider fallback required: {error_msg}")
                        else:
                            # 其他错误直接返回给用户
                            responses.append(response.model_dump(exclude_none=True))
                            # 错误响应：在yield前上报
                            asyncio.create_task(self._async_report_error_trace(
                                event, Exception(error_msg), processed_params, cfg, original_params,
                                effective_before_hooks, effective_after_hooks
                            ))
                            # Record OTEL metrics error - 上报错误指标
                            if llm_metrics and otel_context:
                                error_type = response.error.get('type', 'unknown_error') \
                                    if isinstance(response.error, dict) else 'error'
                                asyncio.create_task(
                                    self._async_report_otel_error(llm_metrics, otel_context, error_type))
                            logger.info(f"Async yielding error response: {response}")
                            yield response
                            continue

                    if is_streaming:
                        # 流式响应：实现正确的管道式Hook处理
                        # 初始化管道：原始响应作为第一个输入
                        current_chunks = [response]
                        
                        # 依次通过每个hook处理，实现管道传递
                        for hook in effective_after_hooks:
                            hook_name = self._class_name_to_snake_case(hook.__class__.__name__)
                            next_chunks = []
                            
                            # 处理当前所有chunks
                            for chunk in current_chunks:
                                # 每个chunk可能产生多个输出chunk
                                async for hook_result in hook.aprocess_streaming_chunk(
                                        chunk, session_id, is_final=False):
                                    if hook_result.data:
                                        next_chunks.append(hook_result.data)
                                        # 记录hook处理结果用于调试
                                        if hook_result.data.get_text_content():
                                            hook_debug_content[hook_name] += hook_result.data.get_text_content()
                                        hook_response_data[hook_name].append(
                                            hook_result.data.model_dump(exclude_none=True))
                            
                            # 更新当前chunks为这个hook的输出
                            current_chunks = next_chunks
                        
                        # 输出最终处理后的chunks
                        for final_chunk in current_chunks:
                            if final_chunk and not self._is_completely_empty_response(final_chunk):
                                # 记录首token时间（第一次真正yield给用户时，经过hook处理后）
                                if first_token_time is None:
                                    first_token_time = time.time()
                                logger.info(f"Async yielding streaming chunk: {final_chunk}")
                                yield final_chunk
                                responses.append(final_chunk.model_dump(exclude_none=True))
                    else:
                        # 非流式响应：使用原有的方法
                        processed_response = response
                        for hook in effective_after_hooks:
                            hook_result = await hook.aprocess_non_streaming_response(processed_response, hook_metadata)
                            hook_name = self._class_name_to_snake_case(hook.__class__.__name__)
                            hook_debug_content[hook_name] += hook_result.data.get_text_content() or ""
                            hook_response_data[hook_name].append(hook_result.data.model_dump(exclude_none=True))
                            processed_response = hook_result.data
                        
                        # 非流式响应直接输出
                        # 记录首token时间（非流式响应只有一次，经过hook处理后）
                        if first_token_time is None:
                            first_token_time = time.time()
                        logger.info(f"Async yielding non-streaming response: {processed_response}")
                        yield processed_response
                        responses.append(processed_response.model_dump(exclude_none=True))

                # 只对流式响应进行finish处理
                if is_streaming:
                    # 流式响应结束，结束所有hook会话并获取最终剩余内容
                    # 初始化管道：从空的响应列表开始（因为这是finish阶段）
                    current_chunks = [None]  # 开始时传入None表示finish阶段
                    
                    # 依次通过每个hook处理，实现管道传递
                    for hook in effective_after_hooks:
                        hook_name = self._class_name_to_snake_case(hook.__class__.__name__)
                        next_chunks = []
                        
                        # 处理当前所有chunks
                        for chunk in current_chunks:
                            # 每个chunk可能产生多个输出chunk
                            async for hook_result in hook.aprocess_streaming_chunk(chunk, session_id, is_final=True):
                                if hook_result.data:
                                    next_chunks.append(hook_result.data)
                                    # 记录hook处理结果用于调试
                                    if hook_result.data.get_text_content():
                                        hook_debug_content[hook_name] += hook_result.data.get_text_content()
                                    hook_response_data[hook_name].append(hook_result.data.model_dump(exclude_none=True))
                                    responses.append(hook_result.data.model_dump(exclude_none=True))
                        
                        # 更新当前chunks为这个hook的输出
                        current_chunks = next_chunks
                    
                    # 输出所有最终处理后的chunks
                    for final_chunk in current_chunks:
                        if final_chunk and not self._is_completely_empty_response(final_chunk):
                            logger.info(f"Async yielding final response: {final_chunk}")
                            yield final_chunk
                            responses.append(final_chunk.model_dump(exclude_none=True))
                    logger.debug(hook_debug_content)

            except Exception as safety_exception:
                # 统一处理安全分类异常（无论是在响应处理阶段还是flush阶段）
                if (hasattr(safety_exception, '__class__')
                        and safety_exception.__class__.__name__ == 'SafetyClassificationException'):
                    # 获取最后一个响应作为原始响应
                    last_response = None
                    if original_responses:
                        # 从原始响应数据重建对象
                        last_response_data = original_responses[-1]
                        if last_response_data.get('object') == 'chat.completion.chunk':
                            last_response = StreamingModelResponse(**last_response_data)
                        else:
                            last_response = ModelResponse(**last_response_data)
                    
                    blocked_response = self._handle_safety_exception(
                        safety_exception, last_response, cfg, original_responses)
                    responses.append(blocked_response.model_dump(exclude_none=True))
                    # 阻止响应：在yield前上报
                    asyncio.create_task(self._async_report_error_trace(
                        event, safety_exception, processed_params, cfg, original_params, 
                        effective_before_hooks, effective_after_hooks
                    ))
                    logger.info(f"Async yielding blocked response: {blocked_response}")
                    yield blocked_response
                    
                else:
                    # 其他异常继续抛出
                    raise

            # 成功完成流式响应后，异步上报数据
            # 检查 responses 中是否有错误，如果有错误则不再上报正常trace（错误已在前面上报）
            if self._trace_service and responses:
                has_error = any(
                    isinstance(resp, dict) and resp.get('error')
                    for resp in responses
                )
                if not has_error:
                    asyncio.create_task(self._async_report_trace(
                        event, processed_params, cfg, effective_before_hooks, effective_after_hooks,
                        original_params, original_responses, responses, hook_response_data
                    ))

            # Record OTEL metrics completion - 异步处理
            # 检查 responses 中是否有错误，如果有错误则不再上报成功（错误已在前面上报）
            if llm_metrics and otel_context:
                has_error = any(
                    isinstance(resp, dict) and resp.get('error')
                    for resp in responses
                )
                if not has_error:
                    asyncio.create_task(self._async_report_otel_metrics(
                        llm_metrics, otel_context, first_token_time, start_time, responses, True
                    ))
        except Exception as e:
            # 如果启用了trace服务，异步报告错误
            if self._trace_service:
                asyncio.create_task(self._async_report_error_trace(
                    event, e, params, cfg, original_params, effective_before_hooks, effective_after_hooks
                ))

            # Record OTEL metrics error - 也异步处理
            if llm_metrics and otel_context:
                asyncio.create_task(self._async_report_otel_error(
                    llm_metrics, otel_context, type(e).__name__
                ))

            # 重新抛出异常
            raise
        finally:
            # 确保关闭客户端连接，防止出现未关闭client_session的警告
            await model_client.aclose()

    def _try_single_model_sync(self, cfg: ModelConfig, params: RunParams, tmpl_name: str,
                               tmpl: PromptTemplate, variant: str, var, variables: dict,
                               hook_configs: dict, completion_request_params: dict,
                               is_last_model: bool = False, fallback_chain_enabled: bool = False):
        """Try to execute with a single model configuration (synchronous version)."""
        # 创建model client
        from .model_client.factory import create_sync_client
        model_client = create_sync_client(cfg)

        # 记录开始时间用于计算请求持续时间
        start_time = time.time()
        responses = []
        # Extract query for trace reporting
        query = _extract_query_from_params(params.messages, variables)
        
        event = TraceEvent(
            template_name=tmpl_name if tmpl else "",
            template_version=tmpl.version if tmpl else "",
            template_id=tmpl.id if tmpl else "",
            variant=variant,
            model=cfg.get_aggregated_model_name(),  # Use aggregated model name for tracking
            messages_template=var.messages if var else "",
            timestamp=start_time,
            variables=variables,
            query=query,
            user_id=params.user_id,
            request_id=params.request_id,
            app_id=params.app_id,
            conversation_id=params.conversation_id or params.session_id,
            span_id=params.span_id,
            parent_span_id=params.parent_span_id,
            source=params.source,
            ext={"completion_params": completion_request_params},
        )
        
        # Initialize OTEL metrics tracking
        llm_metrics = get_llm_metrics()
        otel_context = None
        first_token_time = None  # Track first token/chunk time
        if llm_metrics:
            otel_context = llm_metrics.record_request_start(
                template_name=tmpl_name,
                model=cfg.get_aggregated_model_name(),
                provider=cfg.provider,
                variant=variant or "",
                streaming=str(params.stream).lower() if hasattr(params, 'stream') else "false"
            )
        try:
            # 保存原始参数用于trace上报
            original_params = params
            original_responses = []

            # 处理hook_configs并创建对应的hooks
            effective_before_hooks, effective_after_hooks = self._create_hooks_from_configs(hook_configs, cfg)

            # 执行before run hooks
            processed_params = params
            hook_metadata = {}
            for hook in effective_before_hooks:
                hook_result = hook.process(processed_params)
                processed_params = hook_result.data
                hook_metadata.update(hook_result.metadata)

            try:
                hook_debug_content = {}
                hook_response_data = {}  # 新增：存储每个hook的完整响应数据
                for hook in effective_after_hooks:
                    # 将类名转换为下划线格式
                    hook_name = self._class_name_to_snake_case(hook.__class__.__name__)
                    hook_debug_content[hook_name] = ""
                    hook_response_data[hook_name] = []  # 存储该hook的所有响应数据
                
                # 根据参数判断是否为流式响应
                is_streaming = processed_params.stream
                
                # 生成唯一的会话ID用于流式处理
                session_id = f"{params.request_id or uuid.uuid4().hex}_{start_time}"
                
                # 如果是流式响应，初始化hook会话
                if is_streaming:
                    for hook in effective_after_hooks:
                        hook.start_streaming_session(session_id, hook_metadata)
                
                # 正常处理模型响应
                for response in model_client.run(processed_params):
                    # 保存原始响应（未进行after hook处理）
                    original_responses.append(response.model_dump(exclude_none=True))
                    # 如果响应包含错误，检查是否应该触发provider fallback
                    if hasattr(response, 'error') and response.error:
                        error_msg = response.error.get('message', '') if isinstance(response.error, dict) else str(
                            response.error)
                        # 创建一个异常对象来检查是否应该重试/fallback
                        error_exception = Exception(error_msg)
                        should_fallback = fallback_chain_enabled or should_retry_error(error_exception)
                        if should_fallback:
                            # 如果是最后一个模型且需要fallback，返回错误响应而不是抛出异常
                            if is_last_model:
                                logger.warning(f"Last model failed with fallback error,"
                                               f" returning error response: {error_msg}")
                                # 将当前response标记为最终错误响应并返回
                                responses.append(response.model_dump(exclude_none=True))
                                # 错误响应：在yield前上报
                                self._start_background_error_report_sync(
                                    event, Exception(error_msg), params, cfg, original_params,
                                    effective_before_hooks, effective_after_hooks
                                )
                                # Record OTEL metrics error - 上报错误指标
                                if llm_metrics and otel_context:
                                    error_type = response.error.get('type', 'unknown_error') \
                                        if isinstance(response.error, dict) else 'error'
                                    self._start_background_otel_error_report_sync(llm_metrics, otel_context, error_type)
                                logger.info(f"Sync yielding final error response: {response}")
                                yield response
                                return
                            else:
                                # 对于可以fallback的错误（包括限流），抛出异常以触发下一个provider
                                logger.warning(f"Error detected, triggering provider fallback: {error_msg}")
                                raise Exception(f"Provider fallback required: {error_msg}")
                        else:
                            # 其他错误直接返回给用户
                            responses.append(response.model_dump(exclude_none=True))
                            # 错误响应：在yield前上报
                            self._start_background_error_report_sync(
                                event, Exception(error_msg), params, cfg, original_params,
                                effective_before_hooks, effective_after_hooks
                            )
                            # Record OTEL metrics error - 上报错误指标
                            if llm_metrics and otel_context:
                                error_type = response.error.get('type', 'unknown_error')\
                                    if isinstance(response.error, dict) else 'error'
                                self._start_background_otel_error_report_sync(llm_metrics, otel_context, error_type)
                            logger.info(f"Sync yielding error response: {response}")
                            yield response
                            continue

                    logger.debug(f"model original response: {response}")
                    
                    if is_streaming:
                        # 流式响应：实现正确的管道式Hook处理
                        # 初始化管道：原始响应作为第一个输入
                        current_chunks = [response]
                        
                        # 依次通过每个hook处理，实现管道传递
                        for hook in effective_after_hooks:
                            hook_name = self._class_name_to_snake_case(hook.__class__.__name__)
                            next_chunks = []
                            
                            # 处理当前所有chunks
                            for chunk in current_chunks:
                                # 每个chunk可能产生多个输出chunk
                                for hook_result in hook.process_streaming_chunk(chunk, session_id, is_final=False):
                                    if hook_result.data:
                                        next_chunks.append(hook_result.data)
                                        # 记录hook处理结果用于调试
                                        if hook_result.data.get_text_content():
                                            hook_debug_content[hook_name] += hook_result.data.get_text_content()
                                        hook_response_data[hook_name].append(
                                            hook_result.data.model_dump(exclude_none=True))
                                        logger.debug(f"{hook.__class__}, processed chunk: {hook_result.data}")
                            
                            # 更新当前chunks为这个hook的输出
                            current_chunks = next_chunks
                        
                        # 输出最终处理后的chunks
                        for final_chunk in current_chunks:
                            if final_chunk and not self._is_completely_empty_response(final_chunk):
                                # 记录首token时间（第一次真正yield给用户时，经过hook处理后）
                                if first_token_time is None:
                                    first_token_time = time.time()
                                logger.debug(f"Sync yielding streaming chunk: {final_chunk}")
                                yield final_chunk
                                responses.append(final_chunk.model_dump(exclude_none=True))
                    else:
                        # 非流式响应：使用原有的方法
                        processed_response = response
                        for hook in effective_after_hooks:
                            hook_result = hook.process_non_streaming_response(processed_response, hook_metadata)
                            hook_name = self._class_name_to_snake_case(hook.__class__.__name__)
                            hook_debug_content[hook_name] += hook_result.data.get_text_content() or ""
                            hook_response_data[hook_name].append(hook_result.data.model_dump(exclude_none=True))
                            processed_response = hook_result.data
                            logger.info(f"{hook.__class__}, {hook_result.data}")
                        
                        # 非流式响应直接输出
                        # 记录首token时间（非流式响应只有一次，经过hook处理后）
                        if first_token_time is None:
                            first_token_time = time.time()
                        logger.info(f"Sync yielding non-streaming response: {processed_response}")
                        yield processed_response
                        responses.append(processed_response.model_dump(exclude_none=True))

                # 只对流式响应进行finish处理
                if is_streaming:
                    logger.debug("Processing streaming response - executing finish stage")
                    # 流式响应结束，结束所有hook会话并获取最终剩余内容
                    # 初始化管道：从空的响应列表开始（因为这是finish阶段）
                    current_chunks = [None]  # 开始时传入None表示finish阶段
                    
                    # 依次通过每个hook处理，实现管道传递
                    for hook in effective_after_hooks:
                        hook_name = self._class_name_to_snake_case(hook.__class__.__name__)
                        next_chunks = []
                        
                        # 处理当前所有chunks
                        for chunk in current_chunks:
                            # 每个chunk可能产生多个输出chunk
                            for hook_result in hook.process_streaming_chunk(chunk, session_id, is_final=True):
                                if hook_result.data:
                                    next_chunks.append(hook_result.data)
                                    # 记录hook处理结果用于调试
                                    if hook_result.data.get_text_content():
                                        hook_debug_content[hook_name] += hook_result.data.get_text_content()
                                    hook_response_data[hook_name].append(hook_result.data.model_dump(exclude_none=True))
                                    logger.debug(f"finish stage {hook.__class__}, processed chunk: {hook_result.data}")
                        
                        # 更新当前chunks为这个hook的输出
                        current_chunks = next_chunks
                    
                    # 输出所有最终处理后的chunks
                    for final_chunk in current_chunks:
                        if final_chunk and not self._is_completely_empty_response(final_chunk):
                            logger.debug(f"Sync yielding final response: {final_chunk}")
                            yield final_chunk
                            responses.append(final_chunk.model_dump(exclude_none=True))
                    
                    logger.debug(hook_debug_content)
                    logger.debug("\n\n\n")
                else:
                    logger.debug("Processing non-streaming response - skipping finish stage")

            except Exception as safety_exception:
                # 统一处理安全分类异常（无论是在响应处理阶段还是flush阶段）
                if (hasattr(safety_exception, '__class__')
                        and safety_exception.__class__.__name__ == 'SafetyClassificationException'):
                    # 获取最后一个响应作为原始响应
                    last_response = None
                    if original_responses:
                        # 从原始响应数据重建对象
                        last_response_data = original_responses[-1]
                        if is_streaming:
                            last_response = StreamingModelResponse(**last_response_data)
                        else:
                            last_response = ModelResponse(**last_response_data)
                    
                    blocked_response = self._handle_safety_exception(
                        safety_exception, last_response, cfg, original_responses)
                    responses.append(blocked_response.model_dump(exclude_none=True))
                    # 阻止响应：在yield前上报
                    self._start_background_error_report_sync(
                        event, safety_exception, params, cfg, original_params, 
                        effective_before_hooks, effective_after_hooks
                    )
                    logger.info(f"Sync yielding blocked response: {blocked_response}")
                    yield blocked_response
                    
                else:
                    # 其他异常继续抛出
                    raise

            # 成功完成同步响应后，使用线程池异步上报数据
            # 检查 responses 中是否有错误，如果有错误则不再上报正常trace（错误已在前面上报）
            if self._trace_service and responses:
                has_error = any(
                    isinstance(resp, dict) and resp.get('error')
                    for resp in responses
                )
                if not has_error:
                    self._start_background_report_sync(
                        event, processed_params, cfg, effective_before_hooks, effective_after_hooks,
                        original_params, original_responses, responses, hook_response_data
                    )

            # Record OTEL metrics completion - 使用线程池异步处理
            # 检查 responses 中是否有错误，如果有错误则不再上报成功（错误已在前面上报）
            if llm_metrics and otel_context:
                has_error = any(
                    isinstance(resp, dict) and resp.get('error')
                    for resp in responses
                )
                if not has_error:
                    self._start_background_otel_report_sync(
                        llm_metrics, otel_context, first_token_time, start_time, responses, True)

        except Exception as e:
            # Handle errors in sync context - 使用线程池异步处理
            if self._trace_service:
                self._start_background_error_report_sync(
                    event, e, params, cfg, original_params, effective_before_hooks, effective_after_hooks
                )

            # Record OTEL metrics error - 使用线程池异步处理
            if llm_metrics and otel_context:
                self._start_background_otel_error_report_sync(llm_metrics, otel_context, type(e).__name__)

            # 重新抛出异常
            raise
        finally:
            # 确保关闭客户端连接
            model_client.close()

    def completion(
        self,
        template_name: str,
        variables: dict[str, Any],
        model_cfg: ModelConfig | dict[str, Any] | None = None,
        *,
        model_strategy: dict[str, Any] | None = None,
        variant_router: dict[str, Any] | None = None,
        version: str | None = None,
        variant: str | None = None,
        ctx: dict[str, Any] | None = None,
        tool_params: ToolParams | list[ToolSpec] | list[dict] | dict[str, Any] | None = None,
        messages: list[Message] | list[dict] | None = None,
        template: PromptTemplate | None = None,
        hook_configs: dict[str, Any] | None = None,
        timeout: float | None = None,
        **run_params: Any,
    ) -> Generator[Union[ModelResponse, StreamingModelResponse], None, None]:
        """Synchronous version: Stream messages produced by running the template.

        Args:
            template_name: Name of the template to run
            variables: Variables to use for template rendering
            model_cfg: Optional ModelConfig to use or override template's config
            model_strategy: Optional model strategy configuration for multi-model fallback.
                           Takes highest priority over variant and input model configs.
                           Format: {"type": "fallback_chain", "models": [...]}
            variant_router: Optional variant router configuration for variant selection.
                           Dict with two fields:
                           - "rules": List of routing rules that define conditions and candidate variants
                           - "context": Dict containing fields used by routing conditions
                           Format: {"rules": [...], "context": {...}}
            variant: Optional variant name to use (highest priority, overrides routing)
            ctx: Optional context for variant selection (defaults to variables)
            tool_params: Optional tool parameters for model calls
            messages: Optional direct messages to use instead of template
            hook_configs: Optional hook configurations dict to override default hooks.
                         Format: {"desensitization": {"type": "wordlist", "wordlist": {...}}}
            timeout: Optional timeout in seconds for the model API call
            **run_params: Additional parameters passed to model run
                         (e.g., stream, temperature, max_tokens, user_id,
                         request_id, conversation_id, session_id, etc.)

        Returns:
            Generator yielding ModelResponse or StreamingResponse objects
        """
        if os.getenv("CHOOSE_PRIVATE_LLM") == "true":
            model_cfg = ModelConfig(
                provider="wanshi-openai",
                model="claude-sonnet-4-20250514",
            )

        # Convert basic types to objects
        converted_model_cfg = self._convert_model_cfg(model_cfg) if model_cfg is not None else None
        converted_tool_params = self._convert_tool_params(tool_params) if tool_params is not None else None
        if messages is not None:
            filtered_messages = self._filter_empty_assistant_messages(messages)
            converted_messages = self._convert_messages(filtered_messages)
        else:
            converted_messages = None

        # Resolve template synchronously
        tmpl_name = template_name
        tmpl = self._sync_resolve(template_name, version) if template is None else template

        # 使用模板解析获取 variant 配置
        ctx = ctx or variables

        if variant is None:
            variant = self._select_variant_by_router(
                variant_router,
                tmpl,
                ctx,
                template_name=tmpl_name,
                version=version
            )
            if variant is None:
                raise ValueError(f"Template {template_name} has no variants")

        logger.debug(f"use variant: {variant}")
        # 如果直接提供了messages，使用提供的messages；否则使用模板生成
        if converted_messages is not None:
            # 直接使用提供的messages，但仍需要获取 variant 配置
            _, var = tmpl.format(
                variables,
                variant=variant,
                selector=ctx,
            )
            params = RunParams(messages=converted_messages, tool_params=converted_tool_params, timeout=timeout,
                                **run_params)
        else:
            # 使用模板生成 messages
            messages, var = tmpl.format(
                variables,
                variant=variant,
                selector=ctx,
            )
            # Filter out empty assistant messages from template-generated messages
            filtered_template_messages = self._filter_empty_assistant_messages(messages)
            params = RunParams(messages=cast(list[Message], filtered_template_messages),
                               tool_params=converted_tool_params, timeout=timeout, **run_params)
        # 根据参数判断是否为流式响应
        is_streaming = params.stream

        # Set trace attributes
        span_attrs = {
            "template.name": tmpl_name,
        }

        if var is not None:
            span_attrs["template.version"] = getattr(var, "version", None) or ""
            span_attrs["variant"] = variant or ""

        with _tracer.start_as_current_span(
            "prompt.run",
            attributes=span_attrs,
        ):
            # 合并配置：传入的model_strategy > 传入的model_cfg > 模板的var.model_strategy > 模板的var.model_cfg > 全局配置
            template_cfg = var.model_cfg if var is not None else None
            variant_strategy = var.model_strategy if var is not None else None
            cfg_list = self._merge_model_configs(input_cfg=converted_model_cfg, template_cfg=template_cfg,
                                                 variant_strategy=variant_strategy, input_strategy=model_strategy)
            fallback_chain_enabled = (
                (isinstance(model_strategy, dict) and model_strategy.get("type") == "fallback_chain")
                or (isinstance(variant_strategy, dict) and variant_strategy.get("type") == "fallback_chain")
            )

            # 应用私有 LLM provider 配置（如果需要）
            self._apply_private_llm_provider(cfg_list)

            logger.info(f"get model_cfg list: {self._mask_sensitive_fields(cfg_list)}")

            # 构建 completion 请求参数并添加到 params 中
            completion_request_params = {
                "template_name": template_name,
                "variables": variables,
                "model_cfg": converted_model_cfg.model_dump() if converted_model_cfg else None,
                "model_strategy": model_strategy,
                "variant_router": variant_router,
                "version": version,
                "variant": variant,
                "ctx": ctx,
                "tool_params": converted_tool_params.model_dump() if converted_tool_params else None,
                "messages": [msg.model_dump() for msg in converted_messages] if converted_messages else None,
                "hook_configs": hook_configs,
                "timeout": timeout,
                "run_params": run_params,
            }

            # Try each model config in order until one succeeds
            last_exception = None
            for i, cfg in enumerate(cfg_list):
                is_last_model = (i == len(cfg_list) - 1)
                try:
                    for response in self._try_single_model_sync(
                        cfg, params, tmpl_name, tmpl, variant, var, variables, hook_configs,
                            completion_request_params, is_last_model, fallback_chain_enabled
                    ):
                        yield response
                    # If we get here, the model succeeded, so break the loop
                    return
                except Exception as e:
                    last_exception = e
                    logger.warning(f"Model {cfg.get_aggregated_model_name()} (provider: {cfg.provider}) failed: {e}")
                    # If this is the last model, return error response instead of raising exception
                    if is_last_model:
                        logger.error(f"All {len(cfg_list)} models failed, returning error response")
                        # Create error response
                        error_response = ModelResponse(
                            id=str(uuid.uuid4()),
                            model=cfg.get_aggregated_model_name(),
                            created=int(time.time()),
                            error={
                                "message": f"All models failed. Last error: {str(e)}",
                                "type": "fallback_failed",
                                "code": "all_models_failed"
                            }
                        )
                        yield error_response
                        return
                    # Otherwise, continue to next model
                    continue
    def _convert_model_cfg(self, model_cfg: ModelConfig | dict[str, Any]) -> ModelConfig:
        """Convert dict to ModelConfig object if needed."""
        if isinstance(model_cfg, dict):
            return ModelConfig(**model_cfg)
        return model_cfg

    def _convert_tool_params(self,
                             tool_params: ToolParams | list[ToolSpec] | list[dict] | dict[str, Any]) -> ToolParams:
        """Convert dict/list to ToolParams object if needed."""
        if isinstance(tool_params, dict):
            # If it's a dict, assume it contains ToolParams fields
            tools = tool_params.get('tools', [])
            if isinstance(tools, list) and tools:
                # Convert tool dicts to ToolSpec objects
                converted_tools = []
                for tool in tools:
                    if isinstance(tool, dict):
                        converted_tools.append(ToolSpec(**tool))
                    else:
                        converted_tools.append(tool)
                tool_params = tool_params.copy()
                tool_params['tools'] = converted_tools
            return ToolParams(**tool_params)
        elif isinstance(tool_params, list):
            # Convert list of dicts/ToolSpecs to ToolParams
            converted_tools = []
            for tool in tool_params:
                if isinstance(tool, dict):
                    converted_tools.append(ToolSpec(**tool))
                else:
                    converted_tools.append(tool)
            return ToolParams(tools=converted_tools)
        return tool_params

    def _filter_empty_assistant_messages(self, messages: list[Message] | list[dict]) -> list[Message] | list[dict]:
        """Filter out empty assistant messages from the messages list, but keep those with tool calls."""
        filtered_messages = []

        for msg in messages:
            # Extract role, content and tool_calls from Message object or dict
            if isinstance(msg, dict):
                role = msg.get("role", "")
                content = msg.get("content", "")
                tool_calls = msg.get("tool_calls", None)
            else:
                role = getattr(msg, "role", "")
                content = getattr(msg, "content", "")
                tool_calls = getattr(msg, "tool_calls", None)

            # Skip assistant messages with empty content only if they don't have tool calls
            if role == "assistant":
                # If message has tool calls, keep it even if content is empty
                if tool_calls is not None and tool_calls:
                    filtered_messages.append(msg)
                    continue

                # Handle different content formats for messages without tool calls
                if content is None or content == "":
                    continue
                elif isinstance(content, list):
                    # Handle array format like [{"type": "text", "text": ""}]
                    if not content or all(
                        item.get("text", "") == "" if isinstance(item, dict) and item.get("type") == "text"
                        else False
                        for item in content
                    ):
                        continue
                elif isinstance(content, str) and content.strip() == "":
                    continue

            filtered_messages.append(msg)

        return filtered_messages

    def _is_streaming_response(self, response: Union["ModelResponse", "StreamingModelResponse"]) -> bool:
        """判断是否为流式响应。
        
        Args:
            response: 响应对象
            
        Returns:
            bool: True表示流式响应，False表示非流式响应
        """
        # 根据响应对象类型和字段来判断
        if hasattr(response, 'object'):
            return response.object == "chat.completion.chunk"
        elif hasattr(response, 'choices') and response.choices:
            # 检查choices中是否包含delta字段（流式）还是message字段（非流式）
            first_choice = response.choices[0]
            return hasattr(first_choice, 'delta')
        else:
            # 默认假设为非流式
            return False

    def _convert_messages(self, messages: list[Message] | list[dict]) -> list[Message]:
        """Convert list of dicts to list of Message objects if needed."""
        if messages and isinstance(messages[0], dict):
            # Convert list of dicts to Message objects
            return [Message(**msg) if isinstance(msg, dict) else msg for msg in messages]
        return messages

    def _create_hooks_from_configs(self, hook_configs: dict[str, Any] | None, model_cfg: ModelConfig | None = None) \
            -> tuple[list[BeforeRunHook], list[AfterRunHook]]:
        """根据hook_configs创建动态hooks。
        
        Args:
            hook_configs: hook配置字典，格式为 {"desensitization": {"type": "wordlist", "wordlist": {...}}}
            model_cfg: 模型配置，用于策略判断
            
        Returns:
            tuple: (before_hooks, after_hooks)
        """
        before_hooks = []
        after_hooks = []
        
        # 合并默认配置和传入配置，传入配置优先
        effective_hook_configs = {}
        if hasattr(self, '_default_hook_configs') and self._default_hook_configs:
            effective_hook_configs.update(self._default_hook_configs)
        if hook_configs:
            effective_hook_configs.update(hook_configs)
            
        if not effective_hook_configs:
            return before_hooks, after_hooks
            
        # Process anonymization configuration (support backward compatibility)
        anon_config = effective_hook_configs.get("anonymization")
        if anon_config:
            hook_type = anon_config.get("type", "").lower()
            
            if hook_type == "wordlist":
                # Create wordlist anonymization hook
                wordlist = anon_config.get("wordlist", {})
                if wordlist:
                    from .hooks.wordlist_anonymization_hook import WordlistAnonymizationHook
                    hook = WordlistAnonymizationHook(wordlist=wordlist)
                    before_hooks.append(hook)
                    after_hooks.append(hook)
                    logger.info(f"WordlistAnonymizationHook: Created hook with {len(wordlist)} word mappings")
                else:
                    logger.info(f"WordlistAnonymizationHook: Skipped hook creation - empty wordlist")
            else:
                logger.info(f"WordlistAnonymizationHook: Skipped hook creation - unsupported type '{hook_type}'")
        
        # Process safety classification configuration
        safety_config = effective_hook_configs.get("safety_classification")
        if safety_config:
            strategy = safety_config.get("strategy", "disabled")
            should_enable = self._should_enable_safety_hook_by_strategy(strategy, model_cfg)
            
            if should_enable:
                from .hooks.safety_classification_hook import SafetyClassificationHook
                # 从配置中获取所有参数
                hook_kwargs = {
                    'api_url': safety_config.get(
                        "api_url", "https://aisecurity.baidu-int.com/output_safety_multi_classification_service"),
                    'blocked_message': safety_config.get("blocked_message", ""),
                    'timeout': safety_config.get("timeout", 5),
                    'max_retries': safety_config.get("max_retries", 3),
                    'retry_delay': safety_config.get("retry_delay", 1.0),
                    'max_concurrent_checks': safety_config.get("max_concurrent_checks", 10)
                }
                # 过滤掉None值的参数
                hook_kwargs = {k: v for k, v in hook_kwargs.items() if v is not None}
                hook = SafetyClassificationHook(**hook_kwargs)
                after_hooks.append(hook)
                logger.info(f"SafetyClassificationHook: Created hook for strategy"
                            f" '{strategy}' with model '{model_cfg.model if model_cfg else 'unknown'}'")
            else:
                logger.info(f"SafetyClassificationHook: Skipped hook creation for strategy"
                            f" '{strategy}' with model '{model_cfg.model if model_cfg else 'unknown'}'")
        return before_hooks, after_hooks
    
    def _should_enable_safety_hook_by_strategy(self, strategy: str, model_cfg: ModelConfig | None) -> bool:
        """根据策略和模型配置判断是否应该启用安全分类钩子。
        
        Args:
            strategy: 安全检查策略
            model_cfg: 模型配置
            
        Returns:
            bool: True表示应该启用安全钩子，False表示不应该启用
        """
        if strategy == "all":
            return True
        elif strategy == "check_non_domestic":
            if not model_cfg:
                return False
            model_name = model_cfg.model.lower() if model_cfg.model else ""
            # 如果模型名称包含"claude"，则认为是非国产模型，启用安全检查
            return "claude" in model_name
        else:
            # 其他策略值都不启用（包括 "disabled"）
            return False
    

    def _create_streaming_response(
            self, content: str, original_response: Union["ModelResponse", "StreamingModelResponse"],
            cfg: "ModelConfig", finish_reason: str | None = None) -> "StreamingModelResponse":
        """创建流式响应，确保元信息与原始响应一致。
        
        Args:
            content: 响应内容
            original_response: 原始响应，用于获取元信息
            cfg: 模型配置
            finish_reason: 完成原因
            
        Returns:
            StreamingModelResponse: 新的流式响应
        """
        from .message import StreamingChoice, Message
        
        delta_message = Message(role="assistant", content=content)
        choice = StreamingChoice(index=0, delta=delta_message, finish_reason=finish_reason)
        
        return StreamingModelResponse(
            id=getattr(original_response, 'id', None) or f"chatcmpl-{int(time.time())}",
            object=getattr(original_response, 'object', None) or "chat.completion.chunk",
            created=getattr(original_response, 'created', None) or int(time.time()),
            model=getattr(original_response, 'model', None) or cfg.model,
            choices=[choice],
            usage=getattr(original_response, 'usage', None)
        )

    def _create_blocked_response(
            self, blocked_message: str, original_response: Union["ModelResponse", "StreamingModelResponse"],
            cfg: "ModelConfig") -> Union["ModelResponse", "StreamingModelResponse"]:
        """创建阻止响应，确保元信息与原始响应一致。
        
        Args:
            blocked_message: 阻止消息内容
            original_response: 原始响应，用于获取元信息
            cfg: 模型配置
            
        Returns:
            Union[ModelResponse, StreamingModelResponse]: 阻止响应
        """

        # 根据原始响应类型创建相应的阻止响应
        if isinstance(original_response, StreamingModelResponse):
            delta_message = Message(role="assistant", content=blocked_message)
            choice = StreamingChoice(index=0, delta=delta_message, finish_reason="content_filter")
            return StreamingModelResponse(
                id=original_response.id or f"chatcmpl-blocked-{int(time.time())}",
                object=original_response.object or "chat.completion.chunk",
                created=original_response.created or int(time.time()),
                model=original_response.model or cfg.model,
                choices=[choice],
                usage=original_response.usage
            )
        else:
            # 非流式响应
            message = Message(role="assistant", content=blocked_message)
            choice = Choice(index=0, message=message, finish_reason="content_filter")
            return ModelResponse(
                id=original_response.id or f"chatcmpl-blocked-{int(time.time())}",
                object=original_response.object or "chat.completion",
                created=original_response.created or int(time.time()),
                model=original_response.model or cfg.model,
                choices=[choice],
                usage=original_response.usage
            )

    def _class_name_to_snake_case(self, class_name: str) -> str:
        """将类名转换为下划线格式。
        
        Args:
            class_name: Class name, like 'WordlistAnonymizationHook'
            
        Returns:
            下划线格式的字符串，如 'wordlist_desensitization_hook'
        """
        import re
        # 在大写字母前插入下划线，然后转为小写
        snake_case = re.sub(r'(?<!^)(?=[A-Z])', '_', class_name).lower()
        return snake_case

    def _is_completely_empty_response(self, response: Union["ModelResponse", "StreamingModelResponse"]) -> bool:
        """检查响应是否完全为空（没有content、tool_calls、usage、finish_reason等任何有效内容）。
        
        Args:
            response: 要检查的响应对象
            
        Returns:
            bool: True表示响应完全为空，False表示响应有有效内容
        """
        # 检查是否有错误信息
        if hasattr(response, 'error') and response.error:
            return False
            
        # 检查是否有usage信息
        if hasattr(response, 'usage') and response.usage:
            return False
            
        # 检查choices
        if not hasattr(response, 'choices') or not response.choices:
            return True
            
        for choice in response.choices:
            # 检查finish_reason
            if hasattr(choice, 'finish_reason') and choice.finish_reason:
                return False
                
            # 检查message（非流式响应）
            if hasattr(choice, 'message') and choice.message:
                message = choice.message
                # 检查content
                if hasattr(message, 'content') and message.content:
                    if isinstance(message.content, str) and message.content.strip():
                        return False
                    elif isinstance(message.content, list) and message.content:
                        # 检查列表格式的content是否有非空文本
                        for item in message.content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                if item.get("text", "").strip():
                                    return False
                            elif isinstance(item, dict) and item.get("type") != "text":
                                # 非文本类型的内容（如图片）也算有效内容
                                return False
                # 检查tool_calls
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    return False
                    
            # 检查delta（流式响应）
            if hasattr(choice, 'delta') and choice.delta:
                delta = choice.delta
                # 检查content
                if hasattr(delta, 'content') and delta.content:
                    if isinstance(delta.content, str) and delta.content.strip():
                        return False
                    elif isinstance(delta.content, list) and delta.content:
                        # 检查列表格式的content是否有非空文本
                        for item in delta.content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                if item.get("text", "").strip():
                                    return False
                            elif isinstance(item, dict) and item.get("type") != "text":
                                # 非文本类型的内容也算有效内容
                                return False
                # 检查tool_calls
                if hasattr(delta, 'tool_calls') and delta.tool_calls:
                    return False
                # 检查role（流式响应中role的变化也算有效内容）
                if hasattr(delta, 'role') and delta.role:
                    return False
                    
        # 所有检查都通过，说明响应完全为空
        return True

    def _handle_safety_exception(
            self, exception: Exception, original_response: Union["ModelResponse", "StreamingModelResponse"] | None,
            cfg: "ModelConfig", original_responses: list) -> Union["ModelResponse", "StreamingModelResponse"]:
        """统一处理安全分类异常。
        
        Args:
            exception: 安全异常对象
            original_response: 原始响应对象（可能为None，在flush阶段）
            cfg: 模型配置
            original_responses: 原始响应列表，用于获取元信息
            
        Returns:
            Union[ModelResponse, StreamingModelResponse]: 阻止响应
        """
        blocked_message = getattr(exception, 'blocked_message', '')
        
        if original_response is not None:
            # 正常处理阶段，有原始响应
            return self._create_blocked_response(blocked_message, original_response, cfg)
        else:
            # Flush阶段，没有原始响应，尝试从历史响应获取元信息
            from .message import StreamingChoice, Message
            
            last_response_id = None
            last_response_usage = None
            if original_responses:
                last_original = original_responses[-1]
                last_response_id = last_original.get('id')
                last_response_usage = last_original.get('usage')
            
            # 对于flush阶段，总是创建流式响应
            delta_message = Message(role="assistant", content=blocked_message)
            choice = StreamingChoice(index=0, delta=delta_message, finish_reason="content_filter")
            return StreamingModelResponse(
                id=last_response_id or f"chatcmpl-blocked-{int(time.time())}",
                object="chat.completion.chunk",
                created=int(time.time()),
                model=cfg.get_aggregated_model_name(),  # Use aggregated model name for tracking
                choices=[choice],
                usage=last_response_usage
            )

    async def aclose(self):
        """Close all resources including trace service and OTEL metrics."""
        if self._trace_service:
            await self._trace_service.aclose()
        # Shutdown OTEL metrics queue
        shutdown_llm_metrics()

    # Backward compatibility alias
    async def close(self):
        """Deprecated: use aclose() instead."""
        import warnings
        warnings.warn("close() is deprecated, use aclose() instead", DeprecationWarning, stacklevel=2)
        await self.aclose()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.aclose()

    async def _async_report_trace(self, event, processed_params, cfg, effective_before_hooks, 
                                  effective_after_hooks, original_params, original_responses, 
                                  responses, hook_response_data):
        """异步上报trace数据，不阻塞主流程"""
        try:
            # 构建llm_request_body
            request_body = {}

            # 从model_client传递过来的基础请求数据
            if hasattr(processed_params, "trace_context") and processed_params.trace_context:
                if "llm_request_body" in processed_params.trace_context:
                    request_body["call_llm_params"] = processed_params.trace_context["llm_request_body"]

            # 添加大模型调用URL
            if cfg.api_url:
                request_body["api_url"] = cfg.api_url

            # Add anonymization related data
            if effective_before_hooks or effective_after_hooks:
                # Collect anonymization mapping relationships
                combined_mapping = {}
                for hook in effective_before_hooks:
                    # Only consider WordlistAnonymizationHook
                    if hasattr(hook, 'wordlist') and hook.wordlist:
                        combined_mapping.update(hook.wordlist)

                # Add anonymization related fields to request_body
                original_messages = [msg.model_dump() for msg in original_params.messages]
                request_body["original_messages"] = restore_original_urls_in_messages(original_messages)
                request_body["anonymization_mapping"] = combined_mapping  # Anonymization mapping

            event.llm_request_body = request_body

            # 构建llm_response_body
            concatenated_data = _concatenate_streaming_responses(original_responses, responses, hook_response_data)
            response_body = {
                "responses": original_responses,  # 大模型原始返回的数据
                "final_responses": responses,  # 经过hooks处理后的最终数据
                "concatenated_response": concatenated_data  # 拼接后的完整响应数据
            }
            event.llm_response_body = response_body

            event.perf_metrics = processed_params.trace_context["perf_metrics"] if hasattr(processed_params,
                                                                                           "trace_context") else {}
            event.token_usage = responses[-1].get("usage", {}) if responses else {}
            if responses and "error" in responses[-1]:
                event.error = json.dumps(responses[-1]["error"], ensure_ascii=False)

            # 异步上报trace事件
            await self._trace_service.areport(event)
        except Exception as e:
            logger.error(f"Failed to report trace asynchronously: {e}")

    async def _async_report_trace_sync(self, event, processed_params, cfg, effective_before_hooks, 
                                       effective_after_hooks, original_params, original_responses, 
                                       responses, hook_response_data):
        """同步模式下的异步上报trace数据"""
        try:
            # 构建llm_request_body
            request_body = {}

            # 从model_client传递过来的基础请求数据
            if hasattr(processed_params, "trace_context") and processed_params.trace_context:
                if "llm_request_body" in processed_params.trace_context:
                    request_body["call_llm_params"] = processed_params.trace_context["llm_request_body"]

            # 添加大模型调用URL
            if cfg.api_url:
                request_body["api_url"] = cfg.api_url

            # Add anonymization related data
            if effective_before_hooks or effective_after_hooks:
                # Collect anonymization mapping relationships
                combined_mapping = {}
                for hook in effective_before_hooks:
                    # Only consider WordlistAnonymizationHook
                    if hasattr(hook, 'wordlist') and hook.wordlist:
                        combined_mapping.update(hook.wordlist)

                original_messages = [msg.model_dump() for msg in original_params.messages]
                request_body["original_messages"] = restore_original_urls_in_messages(original_messages)
                request_body["anonymization_mapping"] = combined_mapping  # Anonymization mapping

            event.llm_request_body = request_body

            # 构建llm_response_body
            concatenated_data = _concatenate_streaming_responses(original_responses, responses, hook_response_data)
            response_body = {
                "concatenated_response": concatenated_data,  # 拼接后的完整响应数据
                "responses": original_responses,  # 大模型原始返回的数据
                "final_responses": responses,  # 经过hooks处理后的最终数据
            }
            if hasattr(processed_params, "trace_context") and processed_params.trace_context:
                if "response_extra" in processed_params.trace_context:
                    response_body["response_extra"] = processed_params.trace_context["response_extra"]
            event.llm_response_body = response_body

            event.perf_metrics = processed_params.trace_context.get("perf_metrics") if hasattr(processed_params,
                                                                                           "trace_context") else {}
            event.token_usage = responses[-1].get("usage", {}) if responses else {}
            if responses and "error" in responses[-1]:
                event.error = json.dumps(responses[-1]["error"], ensure_ascii=False)

            # 使用同步方法上报
            await asyncio.get_event_loop().run_in_executor(None, self._trace_service.report, event)
        except Exception as e:
            logger.error(f"Failed to report trace synchronously: {e}")

    async def _async_report_error_trace(self, event, error, params, cfg, original_params, 
                                        effective_before_hooks, effective_after_hooks):
        """异步上报错误trace数据"""
        try:
            # 构建错误trace事件
            event.error = str(error)

            # 构建错误情况下的llm_request_body
            request_body = {}

            # 从params中获取trace_context数据（如果存在）
            if hasattr(params, "trace_context") and params.trace_context:
                if "llm_request_body" in params.trace_context:
                    request_body["call_llm_params"] = params.trace_context["llm_request_body"]
            else:
                # 如果没有trace_context，则创建基本请求信息
                request_body = {
                    "model": getattr(cfg, "model", ""),
                }

            # 添加大模型调用URL
            if cfg.api_url:
                request_body["api_url"] = cfg.api_url

            # 添加messages信息
            if effective_before_hooks or effective_after_hooks:
                # 有hooks的情况下，记录原始和处理后的messages
                original_messages = [msg.model_dump() for msg in original_params.messages]
                request_body["original_messages"] = restore_original_urls_in_messages(original_messages)
                # 尝试获取mapping
                combined_mapping = {}
                for hook in effective_before_hooks:
                    # Only consider WordlistAnonymizationHook
                    if hasattr(hook, 'wordlist') and hook.wordlist:
                        combined_mapping.update(hook.wordlist)
                request_body["desensitization_mapping"] = combined_mapping

            event.llm_request_body = request_body

            # 异步上报错误事件
            await self._trace_service.areport(event)
        except Exception as e:
            logger.error(f"Failed to report error trace asynchronously: {e}")

    async def _async_report_error_trace_sync(self, event, error, params, cfg, original_params, 
                                             effective_before_hooks, effective_after_hooks):
        """同步模式下的异步上报错误trace数据"""
        try:
            event.error = str(error)

            # 构建错误情况下的llm_request_body
            request_body = {}

            if hasattr(params, "trace_context") and params.trace_context:
                if "llm_request_body" in params.trace_context:
                    request_body["call_llm_params"] = params.trace_context["llm_request_body"]
            else:
                request_body = {
                    "model": getattr(cfg, "model", ""),
                }

            # 添加大模型调用URL
            if cfg.api_url:
                request_body["api_url"] = cfg.api_url

            # 添加messages信息
            if effective_before_hooks or effective_after_hooks:
                # 有hooks的情况下，记录原始和处理后的messages
                original_messages = [msg.model_dump() for msg in original_params.messages]
                request_body["original_messages"] = restore_original_urls_in_messages(original_messages)
                # 尝试获取mapping
                combined_mapping = {}
                for hook in effective_before_hooks:
                    # Only consider WordlistAnonymizationHook
                    if hasattr(hook, 'wordlist') and hook.wordlist:
                        combined_mapping.update(hook.wordlist)
                request_body["desensitization_mapping"] = combined_mapping

            event.llm_request_body = request_body

            # 使用线程池执行同步上报
            await asyncio.get_event_loop().run_in_executor(None, self._trace_service.report, event)
        except Exception as e:
            logger.error(f"Failed to report error trace synchronously: {e}")

    async def _async_report_otel_metrics(self, llm_metrics, otel_context, first_token_time, 
                                         start_time, responses, success):
        """异步上报OTEL指标"""
        try:
            # Extract token usage from the last response
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0
            cache_read_input_tokens = 0
            cache_creation_input_tokens = 0

            # Calculate first token latency
            first_token_latency = 0.0
            if first_token_time and start_time:
                first_token_latency = first_token_time - start_time

            if responses:
                last_response = responses[-1]
                usage = last_response.get("usage", {})

                # Extract all token metrics using API field names
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                total_tokens = usage.get("total_tokens", 0)
                cache_read_input_tokens = usage.get("cache_read_input_tokens", 0)
                cache_creation_input_tokens = usage.get("cache_creation_input_tokens", 0)

            # 使用线程池执行OTEL上报
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: llm_metrics.record_request_complete(
                    context=otel_context,
                    first_token_latency_seconds=first_token_latency,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    cache_read_input_tokens=cache_read_input_tokens,
                    cache_creation_input_tokens=cache_creation_input_tokens,
                    success=success
                )
            )
        except Exception as e:
            logger.error(f"Failed to report OTEL metrics asynchronously: {e}")

    async def _async_report_otel_metrics_sync(self, llm_metrics, otel_context, responses, success):
        """同步模式下的异步上报OTEL指标"""
        try:
            # Extract token usage from the last response
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0
            cache_read_input_tokens = 0
            cache_creation_input_tokens = 0

            if responses:
                last_response = responses[-1]
                usage = last_response.get("usage", {})

                # Extract all token metrics using API field names
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                total_tokens = usage.get("total_tokens", 0)
                cache_read_input_tokens = usage.get("cache_read_input_tokens", 0)
                cache_creation_input_tokens = usage.get("cache_creation_input_tokens", 0)

            # 使用线程池执行OTEL上报
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: llm_metrics.record_request_complete(
                    context=otel_context,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    cache_read_input_tokens=cache_read_input_tokens,
                    cache_creation_input_tokens=cache_creation_input_tokens,
                    success=success
                )
            )
        except Exception as e:
            logger.error(f"Failed to report OTEL metrics synchronously: {e}")

    async def _async_report_otel_error(self, llm_metrics, otel_context, error_type):
        """异步上报OTEL错误指标"""
        try:
            # 使用线程池执行OTEL错误上报
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: llm_metrics.record_request_complete(
                    context=otel_context,
                    success=False,
                    error_type=error_type
                )
            )
        except Exception as e:
            logger.error(f"Failed to report OTEL error asynchronously: {e}")

    def _start_background_report_sync(self, event, processed_params, cfg, effective_before_hooks, 
                                      effective_after_hooks, original_params, original_responses, 
                                      responses, hook_response_data):
        """在后台线程中执行同步上报"""
        import threading
        
        def sync_report():
            try:
                # 构建llm_request_body
                request_body = {}

                if hasattr(processed_params, "trace_context") and processed_params.trace_context:
                    if "llm_request_body" in processed_params.trace_context:
                        request_body["call_llm_params"] = processed_params.trace_context["llm_request_body"]

                if cfg.api_url:
                    request_body["api_url"] = cfg.api_url

                if effective_before_hooks or effective_after_hooks:
                    combined_mapping = {}
                    for hook in effective_before_hooks:
                        if hasattr(hook, 'wordlist') and hook.wordlist:
                            combined_mapping.update(hook.wordlist)

                    original_messages = [msg.model_dump() for msg in original_params.messages]
                    request_body["original_messages"] = restore_original_urls_in_messages(original_messages)
                    request_body["anonymization_mapping"] = combined_mapping

                event.llm_request_body = request_body

                concatenated_data = _concatenate_streaming_responses(original_responses, responses, hook_response_data)
                response_body = {
                    "concatenated_response": concatenated_data,
                    "responses": original_responses,
                    "final_responses": responses,
                }
                if hasattr(processed_params, "trace_context") and processed_params.trace_context:
                    if "response_extra" in processed_params.trace_context:
                        response_body["response_extra"] = processed_params.trace_context["response_extra"]
                event.llm_response_body = response_body

                event.perf_metrics = processed_params.trace_context.get("perf_metrics") \
                    if hasattr(processed_params, "trace_context") else {}
                event.token_usage = responses[-1].get("usage", {}) if responses else {}
                if responses and "error" in responses[-1]:
                    event.error = json.dumps(responses[-1]["error"], ensure_ascii=False)

                # 同步上报
                self._trace_service.report(event)
            except Exception as e:
                logger.error(f"Failed to report trace in background thread: {e}")

        thread = threading.Thread(target=sync_report, daemon=True)
        thread.start()

    def _start_background_error_report_sync(self, event, error, params, cfg, original_params, 
                                            effective_before_hooks, effective_after_hooks):
        """在后台线程中执行错误上报"""
        import threading
        
        def sync_error_report():
            try:
                event.error = str(error)
                request_body = {}

                if hasattr(params, "trace_context") and params.trace_context:
                    if "llm_request_body" in params.trace_context:
                        request_body["call_llm_params"] = params.trace_context["llm_request_body"]
                else:
                    request_body = {"model": getattr(cfg, "model", "")}

                if cfg.api_url:
                    request_body["api_url"] = cfg.api_url

                if effective_before_hooks or effective_after_hooks:
                    original_messages = [msg.model_dump() for msg in original_params.messages]
                    request_body["original_messages"] = restore_original_urls_in_messages(original_messages)
                    combined_mapping = {}
                    for hook in effective_before_hooks:
                        if hasattr(hook, 'wordlist') and hook.wordlist:
                            combined_mapping.update(hook.wordlist)
                    request_body["desensitization_mapping"] = combined_mapping

                event.llm_request_body = request_body
                self._trace_service.report(event)
            except Exception as e:
                logger.error(f"Failed to report error trace in background thread: {e}")

        thread = threading.Thread(target=sync_error_report, daemon=True)
        thread.start()

    def _start_background_otel_report_sync(
            self, llm_metrics, otel_context, first_token_time, start_time, responses, success):
        """在后台线程中执行OTEL指标上报"""
        import threading

        def sync_otel_report():
            try:
                prompt_tokens = 0
                completion_tokens = 0
                total_tokens = 0
                cache_read_input_tokens = 0
                cache_creation_input_tokens = 0

                # Calculate first token latency
                first_token_latency = 0.0
                if first_token_time and start_time:
                    first_token_latency = first_token_time - start_time

                if responses:
                    last_response = responses[-1]
                    usage = last_response.get("usage", {})

                    # Extract all token metrics using API field names
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
                    total_tokens = usage.get("total_tokens", 0)
                    cache_read_input_tokens = usage.get("cache_read_input_tokens", 0)
                    cache_creation_input_tokens = usage.get("cache_creation_input_tokens", 0)

                llm_metrics.record_request_complete(
                    context=otel_context,
                    first_token_latency_seconds=first_token_latency,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    cache_read_input_tokens=cache_read_input_tokens,
                    cache_creation_input_tokens=cache_creation_input_tokens,
                    success=success
                )
            except Exception as e:
                logger.error(f"Failed to report OTEL metrics in background thread: {e}")

        thread = threading.Thread(target=sync_otel_report, daemon=True)
        thread.start()

    def _start_background_otel_error_report_sync(self, llm_metrics, otel_context, error_type):
        """在后台线程中执行OTEL错误上报"""
        import threading
        
        def sync_otel_error_report():
            try:
                llm_metrics.record_request_complete(
                    context=otel_context,
                    success=False,
                    error_type=error_type
                )
            except Exception as e:
                logger.error(f"Failed to report OTEL error in background thread: {e}")

        thread = threading.Thread(target=sync_otel_error_report, daemon=True)
        thread.start()


    @classmethod
    def from_setting(cls, setting: Setting) -> PromptEngine:
        """Create an engine instance from a :class:`Setting`.
        
        Args:
            setting: Setting instance. If None, will load from default config file.
            
        Returns:
            Configured PromptEngine instance
        """
        # 如果未提供setting，从配置文件加载
        if setting is None:
            raise ValueError("No setting provided")

        # 创建prompt loaders
        prompt_loaders: list[TemplateLoader] = [FileLoader(Path(p)) for p in setting.template_paths]
        if setting.registry_url:
            # 使用简单缓存版本的HTTP Loader
            prompt_loaders.append(HTTPLoaderSimpleCache(
                base_url=setting.registry_url, 
                auth_token=setting.registry_api_key
            ))
        if setting.memory_templates:
            prompt_loaders.append(MemoryLoader(setting.memory_templates))
        # if setting.config_loader:
        #     prompt_loaders.append(setting.config_loader)

        # 创建model loaders
        model_loaders: list[ModelConfigLoader] = []
        # 按照优先级顺序遍历model_loaders，并尝试加载每个loader中的模型配置
        if setting.memory_model_configs:
            grouped_models = setting.memory_model_configs.get("grouped_models", {})
            tokens = setting.memory_model_configs.get("tokens", [])
            if grouped_models:
                model_loaders.append(MemoryModelConfigLoader(grouped_models=grouped_models, tokens=tokens))
        if setting.registry_url and setting.registry_api_key:
            # 使用简单缓存版本的HTTP Model Config Loader
            model_loaders.append(
                HTTPModelConfigLoaderSimpleCache(
                    url=setting.registry_url,
                    registry_api_key=setting.registry_api_key,
                    reload_interval=setting.model_cache_ttl
                ))
        if setting.model_config_path:
            model_loaders.append(FileModelConfigLoader(setting.model_config_path))

        # 确定global config
        global_cfg = setting.default_model_config
        if setting.global_config_loader:
            global_cfg = setting.global_config_loader.load()

        # 创建trace服务（如果配置了）
        trace_service = None
        if getattr(setting, "registry_url", None):
            trace_service = TraceService(
                endpoint_url=setting.registry_url,
            )
        
        # 创建工具追踪服务（复用registry_url）
        tool_trace_service = None
        if getattr(setting, "registry_url", None):
            tool_trace_service = ToolTraceService(
                endpoint_url=setting.registry_url,
            )

        # 创建hooks(如果配置了)
        before_hooks = getattr(setting, 'before_run_hooks', None) or []
        after_hooks = getattr(setting, 'after_run_hooks', None) or []
        
        # 构建默认hook配置
        default_hook_configs = {}
        if setting.default_hook_configs:
            default_hook_configs.update(setting.default_hook_configs)
        
        # 添加具体的hook配置
        if setting.safety_hook_config:
            default_hook_configs["safety_classification"] = setting.safety_hook_config
        if setting.anonymization_hook_config:
            default_hook_configs["anonymization"] = setting.anonymization_hook_config

        # 创建引擎实例
        engine = cls(
            prompt_loaders=prompt_loaders,
            model_loaders=model_loaders,
            cache_ttl=setting.cache_ttl,
            global_model_config=global_cfg,
            trace_service=trace_service,
            tool_trace_service=tool_trace_service,
            before_run_hooks=before_hooks,
            after_run_hooks=after_hooks,
            default_hook_configs=default_hook_configs
        )

        # 初始化 OTEL 指标（如果配置了）
        if setting.otel_config:
            logger.debug(f"开始初始化OTEL配置: {setting.otel_config}")
            try:
                otel_cfg = setting.otel_config
                logger.debug(f"OTEL配置参数: service_name={otel_cfg.get('service_name', 'prompti')}, "
                           f"endpoint={otel_cfg.get('endpoint', 'http://localhost:4317')}, "
                           f"enabled={otel_cfg.get('enabled', True)}")
                
                init_llm_metrics(
                    service_name=otel_cfg.get("service_name", "prompti"),
                    endpoint=otel_cfg.get("endpoint", "http://localhost:4317"),
                    export_interval_ms=otel_cfg.get("export_interval_ms", 5000),
                    enabled=otel_cfg.get("enabled", True),
                    queue_size=otel_cfg.get("queue_size", 10000)
                )
                logger.debug("OTEL metrics with async queue initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize OTEL metrics: {e}")
                import traceback
                logger.error(f"OTEL初始化异常详情: {traceback.format_exc()}")
        else:
            logger.info("未配置OTEL，跳过OTEL初始化")

        # 加载所有模型配置
        engine.load_model_configs()

        # 拉取全局配置 (Redis等)
        if setting.registry_url and setting.registry_api_key:
            engine._load_global_config(setting.registry_url, setting.registry_api_key)

        return engine

    def report_tool_trace(self, payload: dict[str, Any]) -> bool:
        """
        上报工具调用追踪数据到 promptstore
        
        Args:
            payload: 包含工具调用信息的字典，需要包含以下字段：
                - name: 工具名称 (必须)
                - inputs: 工具输入 (必须)
                - outputs: 工具输出 (可选)
                - error: 错误信息 (可选)
                - start_at: 开始时间 (可选，默认当前时间)
                - end_at: 结束时间 (可选，默认当前时间)
                - request_id: 请求ID (可选)
                - conversation_id: 会话ID (可选)
                - user_id: 用户ID (可选)
                - app_id: 应用ID (可选)
                - span_id: span ID (可选)
                - parent_span_id: 父span ID (可选)
                
        Returns:
            bool: 总是返回True（异步上报，不阻塞）
            
        Example:
            payload = {
                "name": "web_search",
                "inputs": {"query": "Python tutorial", "max_results": 5},
                "outputs": {"results": [...], "count": 2},
                "conversation_id": "conv-123",
                "user_id": "user-456"
            }
            engine.report_tool_trace(payload)
        """
        if self._tool_trace_service is None:
            logger.warning("Tool trace service not configured, skipping report")
            return False
            
        try:
            self._tool_trace_service.report_payload(payload)
            return True
        except Exception as e:
            logger.error(f"Failed to report tool trace: {e}", exc_info=True)
            return False


class Setting(BaseModel):
    """Configuration options for :class:`PromptEngine`.

    Configuration can be loaded from a YAML file with the from_file method.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    template_paths: list[Path] = []
    model_config_path: Path | None = None
    cache_ttl: int = 120
    model_cache_ttl: int = 120
    registry_url: str | None = None
    registry_api_key: str | None = None
    memory_templates: dict[str, Any] | None = None
    memory_model_configs: dict[str, Any] | None = None
    config_loader: TemplateLoader | None = None
    global_config_loader: ModelConfigLoader | None = None
    default_model_config: ModelConfig | None = None
    model_config_loaders: list[ModelConfigLoader] | None = None
    before_run_hooks: list[BeforeRunHook] | None = None
    after_run_hooks: list[AfterRunHook] | None = None
    
    # Hook configurations
    default_hook_configs: dict[str, Any] | None = None
    safety_hook_config: dict[str, Any] | None = None
    anonymization_hook_config: dict[str, Any] | None = None
    
    # OpenTelemetry configuration
    otel_config: dict[str, Any] | None = None

    @classmethod
    def from_file(cls, file_path: str | None = None) -> "Setting":
        """Load settings from a YAML configuration file.
        
        Args:
            file_path: Path to the configuration file. If None, will try default locations.
            
        Returns:
            Loaded Setting instance
            
        Raises:
            FileNotFoundError: If no configuration file could be found
        """
        # 如果未指定文件路径，尝试默认路径
        if file_path is None:
            raise FileNotFoundError(f"No configuration file found: {file_path}")

        # 从文件加载配置
        with open(file_path, "r") as f:
            config_data = yaml.safe_load(f)

        # 处理Path类型字段
        if "template_paths" in config_data and isinstance(config_data["template_paths"], list):
            config_data["template_paths"] = [Path(p) for p in config_data["template_paths"]]

        if "model_config_path" in config_data and isinstance(config_data["model_config_path"], str):
            config_data["model_config_path"] = Path(config_data["model_config_path"])

        # 处理ModelConfig字段
        if "default_model_config" in config_data and config_data["default_model_config"]:
            config_data["default_model_config"] = ModelConfig(**config_data["default_model_config"])

        return cls(**config_data)
