"""
Global Configuration Loader

从 PromptStore 获取全局配置 (如 Redis 配置)
"""

import httpx
import time
from typing import Optional, Dict, Any
from pathlib import Path
import json

from ..logger import get_logger

logger = get_logger(__name__)


class GlobalConfigLoader:
    """
    全局配置加载器

    功能:
    1. 从 PromptStore 获取全局配置
    2. 支持缓存和重试
    3. 自动解密加密字段
    4. 失败时降级 (返回空配置)
    """

    def __init__(
        self,
        base_url: str,
        auth_token: str,
        max_retries: int = 3,
        timeout: int = 10,
        enable_cache: bool = True,
        cache_ttl: int = 300,  # 5分钟
    ):
        """
        Args:
            base_url: PromptStore API 基础 URL
            auth_token: 认证 token
            max_retries: 最大重试次数
            timeout: 请求超时时间(秒)
            enable_cache: 是否启用本地缓存
            cache_ttl: 缓存 TTL (秒)
        """
        self.base_url = base_url.rstrip("/")
        self.auth_token = auth_token
        self.max_retries = max_retries
        self.timeout = timeout
        self.enable_cache = enable_cache
        self.cache_ttl = cache_ttl

        self.headers = {"Authorization": f"Bearer {auth_token}"}
        self.client = httpx.Client(timeout=httpx.Timeout(timeout))

        # 缓存相关
        self.cache_dir = Path(".prompti_cache")
        self.cache_file = self.cache_dir / "global_config.json"
        self._config_cache: Optional[Dict[str, Any]] = None
        self._cache_timestamp: Optional[float] = None

        if enable_cache:
            self.cache_dir.mkdir(exist_ok=True)

    def get_config(self, decrypt: bool = True) -> Dict[str, Any]:
        """
        获取全局配置

        Args:
            decrypt: 是否解密加密字段

        Returns:
            全局配置字典,失败返回空字典 {}
        """
        # 检查内存缓存
        if self._is_cache_valid():
            logger.debug("Using cached global config from memory")
            return self._config_cache.copy()

        # 尝试从 PromptStore 获取
        config = self._fetch_from_promptstore()

        if config:
            # 解密配置
            if decrypt:
                config = self._decrypt_config(config)

            # 缓存配置
            self._save_to_cache(config)
            return config

        # 失败时尝试从文件缓存加载
        cached_config = self._load_from_cache()
        if cached_config:
            logger.info("Using cached global config from file (PromptStore unavailable)")
            if decrypt:
                cached_config = self._decrypt_config(cached_config)
            return cached_config

        # 完全失败,返回空配置
        logger.warning("Failed to get global config, using empty config")
        return {}

    def _fetch_from_promptstore(self) -> Optional[Dict[str, Any]]:
        """从 PromptStore 获取配置"""
        url = f"{self.base_url}/config"

        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Fetching global config from {url} (attempt {attempt + 1})")

                resp = self.client.get(url, headers=self.headers)
                resp.raise_for_status()

                data = resp.json()
                if data.get("code") == 0 and "data" in data:
                    config = data["data"]
                    logger.debug("Successfully fetched global config from PromptStore")
                    return config
                else:
                    logger.warning(f"Unexpected response format: {data}")
                    return None

            except httpx.HTTPStatusError as e:
                logger.warning(f"HTTP error when fetching config (attempt {attempt + 1}): {e.response.status_code}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # 指数退避
                continue

            except httpx.RequestError as e:
                logger.warning(f"Request error when fetching config (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                continue

            except Exception as e:
                logger.error(f"Unexpected error when fetching config: {e}")
                return None

        return None

    def _decrypt_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """解密配置中的加密字段"""
        try:
            from ..utils.encryption import decrypt_global_config, is_encryption_available

            if not is_encryption_available():
                logger.warning("Encryption library not available, returning config as-is")
                return config

            decrypted = decrypt_global_config(config.copy())
            return decrypted

        except Exception as e:
            logger.error(f"Failed to decrypt config: {e}")
            return config

    def _save_to_cache(self, config: Dict[str, Any]):
        """保存配置到缓存"""
        try:
            # 更新内存缓存
            self._config_cache = config.copy()
            self._cache_timestamp = time.time()

            # 保存到文件
            if self.enable_cache:
                with open(self.cache_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "config": config,
                        "timestamp": self._cache_timestamp
                    }, f, ensure_ascii=False, indent=2)
                logger.debug(f"Saved global config to cache: {self.cache_file}")

        except Exception as e:
            logger.warning(f"Failed to save config to cache: {e}")

    def _load_from_cache(self) -> Optional[Dict[str, Any]]:
        """从文件缓存加载配置"""
        if not self.enable_cache or not self.cache_file.exists():
            return None

        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            config = data.get("config")
            timestamp = data.get("timestamp")

            # 检查缓存是否过期 (宽松检查,只在文件缓存时检查)
            if timestamp and time.time() - timestamp < 86400:  # 24小时内有效
                logger.debug("Loaded global config from file cache")
                return config

            logger.debug("File cache expired (> 24 hours)")
            return None

        except Exception as e:
            logger.warning(f"Failed to load config from cache: {e}")
            return None

    def _is_cache_valid(self) -> bool:
        """检查内存缓存是否有效"""
        if self._config_cache is None or self._cache_timestamp is None:
            return False

        age = time.time() - self._cache_timestamp
        return age < self.cache_ttl

    def refresh_config(self, decrypt: bool = True) -> Dict[str, Any]:
        """
        强制刷新配置 (忽略缓存)

        Args:
            decrypt: 是否解密

        Returns:
            最新配置
        """
        # 清除内存缓存
        self._config_cache = None
        self._cache_timestamp = None

        return self.get_config(decrypt=decrypt)

    def close(self):
        """关闭 HTTP 客户端"""
        try:
            self.client.close()
        except Exception as e:
            logger.warning(f"Error closing HTTP client: {e}")


# 异步版本 (可选)
class AsyncGlobalConfigLoader:
    """异步版本的全局配置加载器"""

    def __init__(
        self,
        base_url: str,
        auth_token: str,
        max_retries: int = 3,
        timeout: int = 10,
    ):
        self.base_url = base_url.rstrip("/")
        self.auth_token = auth_token
        self.max_retries = max_retries
        self.timeout = timeout
        self.headers = {"Authorization": f"Bearer {auth_token}"}
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(timeout))

    async def get_config(self, decrypt: bool = True) -> Dict[str, Any]:
        """异步获取配置"""
        config = await self._fetch_from_promptstore()

        if config and decrypt:
            config = self._decrypt_config(config)

        return config or {}

    async def _fetch_from_promptstore(self) -> Optional[Dict[str, Any]]:
        """异步从 PromptStore 获取配置"""
        import asyncio

        url = f"{self.base_url}/config"

        for attempt in range(self.max_retries):
            try:
                resp = await self.client.get(url, headers=self.headers)
                resp.raise_for_status()

                data = resp.json()
                if data.get("code") == 0 and "data" in data:
                    return data["data"]

            except Exception as e:
                logger.warning(f"Error fetching config (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)

        return None

    def _decrypt_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """解密配置"""
        try:
            from ..utils.encryption import decrypt_global_config
            return decrypt_global_config(config.copy())
        except Exception as e:
            logger.error(f"Failed to decrypt config: {e}")
            return config

    async def close(self):
        """关闭客户端"""
        await self.client.aclose()
